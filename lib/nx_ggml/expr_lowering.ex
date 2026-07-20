defmodule NxGgml.ExprLowering do
  @moduledoc """
  Walks a traced `Nx.Defn.Expr` tree and lowers it into a ggml compute
  graph via `NxGgml.Nif`'s `GraphBuilder`/`CompiledGraph` native resources
  (see `native/nx_ggml/graph_builder.{h,cpp}`).

  Op coverage is intentionally partial (see `@binary_ops`/`@unary_ops` and
  `do_lower/6`), f32 only. Anything else raises
  `NxGgml.UnsupportedOpError` / `NxGgml.UnsupportedDTypeError`, which
  `NxGgml.Compiler` catches to fall back to `Nx.Defn.Evaluator` — broader
  op/dtype coverage grows by extending those lists and `do_lower/6`, not by
  touching this module's overall shape.
  """

  alias Nx.Defn.Expr

  @supported_dtypes [{:f, 32}]
  # {:s, 32} is only valid for a defn *parameter* that is exclusively used
  # as :gather's indices operand (see gather_index_positions/1 below) --
  # never for a bare {:s, 32} :constant. A blanket "any {:s, 32} node is an
  # index" rule is unsafe: Nx.mean's sum/n decomposition introduces a plain
  # {:s, 32} :constant divisor (n) that must be treated as ordinary
  # arithmetic (ggml requires matching operand types, so an i32 constant
  # divided against an f32 sum aborts the native compute), not as a gather
  # index just because it happens to share the dtype.
  @index_dtypes [{:s, 32}]

  # Op atom names that map 1:1 onto graph_builder.cpp's add_binary/add_unary
  # dispatch strings (Nx.Defn.Expr's op atoms and ggml's op names happen to
  # coincide for all of these, so no translation table is needed).
  @binary_ops ~w(add subtract multiply divide)a
  @unary_ops ~w(negate abs sign sqrt exp log sigmoid tanh sin cos)a

  @doc """
  Lowers `expr` (the result of tracing a defn function with `vars` as
  inputs) into a compiled ggml graph, allocated on `device` (`"cpu"` or
  `"vulkan"`, per `NxGgml.Device.from_opts/1`).

  Returns `%{compiled: compiled_ref, out_shape: shape, out_type: type}`.
  Raises `NxGgml.UnsupportedOpError` or `NxGgml.UnsupportedDTypeError` if
  `expr` uses anything outside the currently-supported subset.
  """
  def build(vars, expr, device) do
    check_dtype!(expr.type)

    builder = NxGgml.Nif.nx_ggml_builder_new(device)
    index_positions = gather_index_positions(expr)

    # `vars` mirrors the original defn's arity, so a composite argument (a
    # tuple/map, or a Nx.Container-implementing struct like
    # Axon.ModelState -- exactly what Axon.build/2's params argument is)
    # appears as one nested element, not a flat tensor. Flatten first: this
    # is the same leaf ordering `expr`'s `:parameter` node indices (and the
    # runtime `params` list Compiler.run/1 receives) use, per
    # Nx.Defn.Evaluator's own `Enum.fetch!(state.params, i)` convention.
    flat_vars = Nx.Defn.Composite.flatten_list(vars)

    param_indices =
      Enum.with_index(flat_vars)
      |> Enum.map(fn {var, pos} ->
        shape = Tuple.to_list(var.shape)

        if pos in index_positions do
          unless var.type in @index_dtypes do
            raise NxGgml.UnsupportedDTypeError, type: var.type
          end

          NxGgml.Nif.nx_ggml_builder_add_param_i32(builder, shape)
        else
          check_dtype!(var.type)
          NxGgml.Nif.nx_ggml_builder_add_param(builder, shape)
        end
      end)

    # :__device__ lives in the same map threaded through every do_lower
    # clause as the expr-id memo cache, under an atom key that can never
    # collide with a real expr id (those are integers/refs) -- avoids
    # widening every do_lower/6 clause's arity just for :erf's device check.
    {output_index, _memo} = lower(builder, expr, param_indices, %{__device__: device})
    compiled = NxGgml.Nif.nx_ggml_builder_finalize(builder, output_index)

    %{compiled: compiled, out_shape: expr.shape, out_type: expr.type}
  end

  # Pre-scan: collects the positions of top-level parameters that appear
  # directly as some :gather node's indices argument, so build/1 knows
  # which params are safe to create as i32 leaves. A parameter position
  # NOT in this set is always treated as an ordinary f32 compute value
  # (see @index_dtypes' moduledoc note on why this can't be a blanket
  # dtype-based rule).
  defp gather_index_positions(expr) do
    {_visited, positions} = scan_gather_indices(expr, MapSet.new(), MapSet.new())
    positions
  end

  defp scan_gather_indices(%Nx.Tensor{data: %Expr{id: id, op: op, args: args}}, visited, positions) do
    if MapSet.member?(visited, id) do
      {visited, positions}
    else
      visited = MapSet.put(visited, id)

      positions =
        case {op, args} do
          {:gather, [_tensor, %Nx.Tensor{data: %Expr{op: :parameter, args: [pos]}}, _opts]} ->
            MapSet.put(positions, pos)

          _ ->
            positions
        end

      Enum.reduce(args, {visited, positions}, fn
        %Nx.Tensor{data: %Expr{}} = arg, {v, p} -> scan_gather_indices(arg, v, p)
        list, {v, p} when is_list(list) -> scan_gather_indices_list(list, v, p)
        _, acc -> acc
      end)
    end
  end

  defp scan_gather_indices_list(list, visited, positions) do
    Enum.reduce(list, {visited, positions}, fn
      %Nx.Tensor{data: %Expr{}} = arg, {v, p} -> scan_gather_indices(arg, v, p)
      nested, {v, p} when is_list(nested) -> scan_gather_indices_list(nested, v, p)
      _, acc -> acc
    end)
  end

  defp lower(builder, %Nx.Tensor{data: %Expr{id: id}} = t, param_indices, memo) do
    case memo do
      %{^id => index} ->
        {index, memo}

      %{} ->
        %Expr{op: op, args: args} = t.data
        {index, memo} = do_lower(builder, op, args, t, param_indices, memo)
        {index, Map.put(memo, id, index)}
    end
  end

  defp do_lower(_builder, :parameter, [pos], _t, param_indices, memo) do
    {Enum.fetch!(param_indices, pos), memo}
  end

  # A :constant node just holds a plain Elixir number (args: [number]), not
  # real tensor data, so it's always encoded as f32 here regardless of its
  # own traced dtype -- e.g. Nx.mean's sum/n decomposition introduces a
  # {:s, 32} :constant divisor (n) that must still combine with an f32 sum
  # (ggml requires matching operand types), and re-encoding "3" as f32 3.0
  # is always numerically valid since it's just a number. This is never in
  # tension with gather indices, which are real multi-element tensors and
  # therefore never :constant nodes -- do_lower(:gather, ...) below only
  # accepts indices that are directly a :parameter.
  defp do_lower(builder, :constant, [number], t, _param_indices, memo) do
    shape = Tuple.to_list(t.shape)

    binary =
      %Nx.Tensor{type: {:f, 32}, shape: t.shape, names: t.names}
      |> Nx.BinaryBackend.constant(number, [])
      |> Nx.BinaryBackend.to_binary(Nx.size(t.shape))

    {NxGgml.Nif.nx_ggml_builder_add_constant_f32(builder, shape, binary), memo}
  end

  defp do_lower(builder, op, [a, b], t, param_indices, memo) when op in @binary_ops do
    check_dtype!(t.type)
    {a_index, memo} = lower_broadcast(builder, a, t.shape, param_indices, memo)
    {b_index, memo} = lower_broadcast(builder, b, t.shape, param_indices, memo)
    op_name = Atom.to_string(op)
    {NxGgml.Nif.nx_ggml_builder_add_binary(builder, op_name, a_index, b_index), memo}
  end

  defp do_lower(builder, op, [a], t, param_indices, memo) when op in @unary_ops do
    check_dtype!(t.type)
    {a_index, memo} = lower(builder, a, param_indices, memo)
    op_name = Atom.to_string(op)
    {NxGgml.Nif.nx_ggml_builder_add_unary(builder, op_name, a_index), memo}
  end

  # ggml has no standalone erf op (only the fused gelu_erf activation), so
  # add_erf wraps erff() as a custom ggml_map_custom1 CPU callback --
  # custom map ops aren't portable to the Vulkan backend, so this only
  # lowers on a "cpu" device compile; a "vulkan" compile falls back to
  # Nx.Defn.Evaluator for the whole graph rather than silently miscomputing.
  defp do_lower(builder, :erf, [a], t, param_indices, memo) do
    check_dtype!(t.type)

    unless memo[:__device__] == "cpu" do
      raise NxGgml.UnsupportedOpError, op: :erf
    end

    {a_index, memo} = lower(builder, a, param_indices, memo)
    {NxGgml.Nif.nx_ggml_builder_add_erf(builder, a_index), memo}
  end

  defp do_lower(builder, :reshape, [a], t, param_indices, memo) do
    check_dtype!(t.type)
    {a_index, memo} = lower(builder, a, param_indices, memo)
    shape = Tuple.to_list(t.shape)
    {NxGgml.Nif.nx_ggml_builder_add_reshape(builder, a_index, shape), memo}
  end

  # :squeeze always carries [tensor, axes] (Nx.Defn.Expr.squeeze/3); the
  # resulting output shape is all that matters for lowering, since dropping
  # size-1 axes never reorders the underlying data.
  defp do_lower(builder, :squeeze, [a, _axes], t, param_indices, memo) do
    check_dtype!(t.type)
    {a_index, memo} = lower(builder, a, param_indices, memo)
    shape = Tuple.to_list(t.shape)
    {NxGgml.Nif.nx_ggml_builder_add_reshape(builder, a_index, shape), memo}
  end

  defp do_lower(builder, :transpose, [a, axes], t, param_indices, memo) do
    check_dtype!(t.type)
    {a_index, memo} = lower(builder, a, param_indices, memo)
    {NxGgml.Nif.nx_ggml_builder_add_transpose(builder, a_index, axes), memo}
  end

  defp do_lower(builder, :broadcast, [a, shape, axes], t, param_indices, memo) do
    check_dtype!(t.type)

    # Only the common trailing-aligned case is supported (the input's axes
    # map to a contiguous, rightmost run of the output's axes) — this is
    # the alignment both ggml_repeat and numpy-style broadcasting use
    # natively, and covers ordinary elementwise-broadcast and scalar-to-shape
    # cases. Arbitrary axes (e.g. broadcasting into a *leading* new axis)
    # fall back to Nx.Defn.Evaluator.
    expected_axes = Enum.to_list((tuple_size(shape) - tuple_size(a.shape))..(tuple_size(shape) - 1))

    unless axes == expected_axes do
      raise NxGgml.UnsupportedOpError, op: :broadcast
    end

    {a_index, memo} = lower(builder, a, param_indices, memo)
    {NxGgml.Nif.nx_ggml_builder_add_broadcast(builder, a_index, Tuple.to_list(shape)), memo}
  end

  # Standard (optionally batched) matmul: contract a's last axis with b's
  # second-to-last axis, with any number of matching leading batch axes on
  # both operands (the common case for linear layers/attention/per-batch
  # matmul; the top real-world op per trellis2cpp's usage data). Anything
  # else (mismatched batch axes, non-trailing contraction axes) falls back
  # to Nx.Defn.Evaluator.
  defp do_lower(builder, :dot, [a, c1, b1, b, c2, b2], t, param_indices, memo) do
    check_dtype!(t.type)
    rank_a = tuple_size(a.shape)
    rank_b = tuple_size(b.shape)
    batch_axes = Enum.to_list(0..(rank_a - 3)//1)

    standard_batched_matmul? =
      rank_a == rank_b and rank_a >= 2 and c1 == [rank_a - 1] and c2 == [rank_b - 2] and
        b1 == batch_axes and b2 == batch_axes

    unless standard_batched_matmul? do
      raise NxGgml.UnsupportedOpError, op: :dot
    end

    {a_index, memo} = lower(builder, a, param_indices, memo)
    {b_index, memo} = lower(builder, b, param_indices, memo)
    {NxGgml.Nif.nx_ggml_builder_add_matmul(builder, a_index, b_index), memo}
  end

  # Full-reduction (no :axes, not :keep_axes) -> 0-d scalar, or reduction
  # over just the last axis (either form of :keep_axes, since the sum
  # itself doesn't need to know -- the final add_reshape to t.shape handles
  # keep_axes/squeeze either way). Any other axes combination falls back to
  # Nx.Defn.Evaluator. Last-axis support also unlocks Nx.mean "for free" for
  # the same case, since Nx.mean is itself composed from sum + a constant
  # divide (both independently supported) at the Nx.Defn tracing level, not
  # a raw :mean primitive.
  defp do_lower(builder, :sum, [a, opts], t, param_indices, memo) do
    check_dtype!(t.type)
    rank = tuple_size(a.shape)
    axes = opts[:axes]

    cond do
      t.shape == {} and (axes == nil or axes == []) ->
        {a_index, memo} = lower(builder, a, param_indices, memo)
        {NxGgml.Nif.nx_ggml_builder_add_sum_all(builder, a_index), memo}

      axes == [rank - 1] ->
        {a_index, memo} = lower(builder, a, param_indices, memo)
        summed = NxGgml.Nif.nx_ggml_builder_add_sum_last_axis(builder, a_index)
        {NxGgml.Nif.nx_ggml_builder_add_reshape(builder, summed, Tuple.to_list(t.shape)), memo}

      true ->
        raise NxGgml.UnsupportedOpError, op: :sum
    end
  end

  # Reduction over just the last axis (ggml has no dedicated row-max op,
  # so this goes through add_reduce_max_last_axis's ggml_pool_1d(MAX)
  # trick -- see graph_builder.cpp). No general/full-reduction or
  # multi-axis case is supported (unlike :sum), since a global max-pool
  # only naturally reduces ne0; anything else falls back to
  # Nx.Defn.Evaluator. This still has real payoff: reduce_max, subtract,
  # exp, sum (last axis), and divide are now all independently lowerable,
  # so a user-composed numerically-stable softmax
  # (`x |> subtract(reduce_max) |> exp |> then(&(&1 / sum(&1)))`) lowers
  # entirely through ggml even though Nx has no raw :softmax primitive to
  # intercept directly.
  defp do_lower(builder, :reduce_max, [a, opts], t, param_indices, memo) do
    check_dtype!(t.type)
    rank = tuple_size(a.shape)
    axes = opts[:axes]

    unless axes == [rank - 1] do
      raise NxGgml.UnsupportedOpError, op: :reduce_max
    end

    {a_index, memo} = lower(builder, a, param_indices, memo)
    last_axis_size = elem(a.shape, rank - 1)
    reduced = NxGgml.Nif.nx_ggml_builder_add_reduce_max_last_axis(builder, a_index, last_axis_size)
    {NxGgml.Nif.nx_ggml_builder_add_reshape(builder, reduced, Tuple.to_list(t.shape)), memo}
  end

  # Embedding-lookup-style gather only: a 2-D table (vocab, dim), gathering
  # whole rows (axis 0) by an {:s, 32} index tensor that is *directly* a
  # defn parameter (matching gather_index_positions/1's pre-scan, which is
  # what made it safe to create that parameter as an i32 leaf in the first
  # place) -- maps directly to ggml_get_rows. Nx.gather itself requires
  # indices' last axis to have length(axes) elements (here: 1), so the
  # real-world shape is (n, 1), not (n,) -- reshaped down to (n,) before
  # get_rows, which wants a plain 1-D index list. Nx.gather's fully general
  # nd-index semantics (multiple coordinate axes at once, or indices
  # computed from an expression rather than passed in directly) are not
  # supported; anything outside this specific shape falls back to
  # Nx.Defn.Evaluator.
  defp do_lower(builder, :gather, [tensor, indices, opts], t, param_indices, memo) do
    check_dtype!(t.type)

    n =
      case indices.shape do
        {n} -> n
        {n, 1} -> n
        _ -> nil
      end

    valid? =
      tuple_size(tensor.shape) == 2 and
        (opts[:axes] == [0] or opts[:axes] == nil) and
        n != nil and
        indices.type in @index_dtypes and
        match?(%Nx.Tensor{data: %Expr{op: :parameter}}, indices)

    unless valid? do
      raise NxGgml.UnsupportedOpError, op: :gather
    end

    {tensor_index, memo} = lower(builder, tensor, param_indices, memo)
    {indices_index, memo} = lower(builder, indices, param_indices, memo)
    indices_1d = NxGgml.Nif.nx_ggml_builder_add_reshape(builder, indices_index, [n])
    gathered = NxGgml.Nif.nx_ggml_builder_add_get_rows(builder, tensor_index, indices_1d)
    {NxGgml.Nif.nx_ggml_builder_add_reshape(builder, gathered, Tuple.to_list(t.shape)), memo}
  end

  # Standard 2-D convolution only: rank-4 NCHW input / OIHW kernel (Nx's
  # default axis order, i.e. no custom input/kernel/output permutation),
  # dilation 1, no feature/batch grouping. This is exactly the "patchify"
  # conv used for e.g. a ViT patch embedding (stride == kernel spatial
  # size, no padding) as well as ordinary convs (stride/padding vary).
  # Maps directly to ggml_conv_2d, whose a/b/result layout comments already
  # match this Nx/PyTorch convention exactly (see graph_builder.cpp) --
  # unlike matmul/transpose, no operand-order derivation was needed here.
  defp do_lower(builder, :conv, [input, kernel, opts], t, param_indices, memo) do
    check_dtype!(t.type)

    valid? =
      tuple_size(input.shape) == 4 and
        tuple_size(kernel.shape) == 4 and
        opts[:feature_group_size] == 1 and
        opts[:batch_group_size] == 1 and
        opts[:input_dilation] == [1, 1] and
        identity_permutation?(opts[:input_permutation]) and
        identity_permutation?(opts[:kernel_permutation]) and
        identity_permutation?(opts[:output_permutation]) and
        match?([{0, 0}, {0, 0}], opts[:padding])

    unless valid? do
      raise NxGgml.UnsupportedOpError, op: :conv
    end

    [s0, s1] = opts[:strides]
    [d0, d1] = opts[:kernel_dilation]

    {input_index, memo} = lower(builder, input, param_indices, memo)
    {kernel_index, memo} = lower(builder, kernel, param_indices, memo)

    {NxGgml.Nif.nx_ggml_builder_add_conv2d(builder, kernel_index, input_index, s0, s1, 0, 0, d0, d1),
     memo}
  end

  # Pairwise-folds ggml_concat across an arbitrary-length tensor list (Nx's
  # :concatenate always carries the full list + axis, but ggml_concat only
  # takes two operands at a time).
  defp do_lower(builder, :concatenate, [tensors, axis], t, param_indices, memo) do
    check_dtype!(t.type)
    rank = tuple_size(t.shape)

    [first | rest] = tensors
    {first_index, memo} = lower(builder, first, param_indices, memo)

    Enum.reduce(rest, {first_index, memo}, fn tensor, {acc_index, acc_memo} ->
      {tensor_index, acc_memo} = lower(builder, tensor, param_indices, acc_memo)
      {NxGgml.Nif.nx_ggml_builder_add_concat(builder, acc_index, tensor_index, axis, rank), acc_memo}
    end)
  end

  # Only literal (compile-time-constant) min/max bounds are lowered, via
  # ggml_clamp; tensor-valued bounds fall back to Nx.Defn.Evaluator.
  defp do_lower(builder, :clip, [a, min, max], t, param_indices, memo) do
    check_dtype!(t.type)

    with {:ok, min_value} <- literal_constant(min),
         {:ok, max_value} <- literal_constant(max) do
      {a_index, memo} = lower(builder, a, param_indices, memo)
      {NxGgml.Nif.nx_ggml_builder_add_clamp(builder, a_index, min_value * 1.0, max_value * 1.0), memo}
    else
      :error -> raise NxGgml.UnsupportedOpError, op: :clip
    end
  end

  defp do_lower(_builder, op, _args, _t, _param_indices, _memo) do
    raise NxGgml.UnsupportedOpError, op: op
  end

  defp literal_constant(%Nx.Tensor{data: %Expr{op: :constant, args: [number]}}), do: {:ok, number}
  defp literal_constant(_), do: :error

  defp identity_permutation?(nil), do: true
  defp identity_permutation?(perm), do: perm == Enum.to_list(0..(length(perm) - 1))

  # Lowers `operand` and, if its shape doesn't already match `target_shape`,
  # wraps it in an explicit ggml_repeat broadcast. Applying this uniformly
  # to both operands of every binary op (rather than picking ggml's a/b
  # order based on which side already matches) sidesteps
  # ggml_add/sub/mul/div's `ggml_can_repeat(b, a)` requirement entirely and,
  # unlike an order-swap, stays correct for non-commutative ops too
  # (subtract/divide) where operand order carries meaning.
  defp lower_broadcast(builder, operand, target_shape, param_indices, memo) do
    {index, memo} = lower(builder, operand, param_indices, memo)

    if operand.shape == target_shape do
      {index, memo}
    else
      {NxGgml.Nif.nx_ggml_builder_add_broadcast(builder, index, Tuple.to_list(target_shape)), memo}
    end
  end

  defp check_dtype!(type) when type in @supported_dtypes, do: :ok
  defp check_dtype!(type), do: raise(NxGgml.UnsupportedDTypeError, type: type)
end
