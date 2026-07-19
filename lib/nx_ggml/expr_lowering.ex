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

    param_indices =
      Enum.map(vars, fn var ->
        check_dtype!(var.type)
        shape = Tuple.to_list(var.shape)
        NxGgml.Nif.nx_ggml_builder_add_param(builder, shape)
      end)

    {output_index, _memo} = lower(builder, expr, param_indices, %{})
    compiled = NxGgml.Nif.nx_ggml_builder_finalize(builder, output_index)

    %{compiled: compiled, out_shape: expr.shape, out_type: expr.type}
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

  defp do_lower(builder, :constant, [number], t, _param_indices, memo) do
    check_dtype!(t.type)
    shape = Tuple.to_list(t.shape)

    binary =
      %Nx.Tensor{type: t.type, shape: t.shape, names: t.names}
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
