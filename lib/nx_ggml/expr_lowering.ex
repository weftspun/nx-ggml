defmodule NxGgml.ExprLowering do
  @moduledoc """
  Walks a traced `Nx.Defn.Expr` tree and lowers it into a ggml compute
  graph via `NxGgml.Nif`'s `GraphBuilder`/`CompiledGraph` native resources
  (see `native/nx_ggml/graph_builder.{h,cpp}`).

  Phase 3 scope only: `:parameter`, `:constant`, and `:add`, f32 only.
  Anything else raises `NxGgml.UnsupportedOpError` /
  `NxGgml.UnsupportedDTypeError`, which `NxGgml.Compiler` catches to fall
  back to `Nx.Defn.Evaluator` — broader op/dtype coverage lands in Phases
  4-6, extending `do_lower/6` and the dtype table below, not by touching
  this module's overall shape.
  """

  alias Nx.Defn.Expr

  @supported_dtypes [{:f, 32}]

  @doc """
  Lowers `expr` (the result of tracing a defn function with `vars` as
  inputs) into a compiled ggml graph.

  Returns `%{compiled: compiled_ref, out_shape: shape, out_type: type}`.
  Raises `NxGgml.UnsupportedOpError` or `NxGgml.UnsupportedDTypeError` if
  `expr` uses anything outside Phase 3's supported subset.
  """
  def build(vars, expr) do
    check_dtype!(expr.type)

    builder = NxGgml.Nif.nx_ggml_builder_new()

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

  defp do_lower(builder, :add, [a, b], t, param_indices, memo) do
    check_dtype!(t.type)
    # Nx's expr builder (Nx.Defn.Expr.add/3, via `commute/*`) may reorder
    # operands so a constant comes first regardless of shape — e.g. `x + 1.0`
    # produces args `[constant(scalar), x(shape)]`. ggml_add(ctx, a, b)
    # requires `b` to broadcast *into* `a` (ggml_can_repeat(b, a)), so the
    # operand whose shape matches the node's own output shape must go
    # first, independent of the expr's arg order.
    {larger, smaller} = if a.shape == t.shape, do: {a, b}, else: {b, a}
    {larger_index, memo} = lower(builder, larger, param_indices, memo)
    {smaller_index, memo} = lower(builder, smaller, param_indices, memo)
    {NxGgml.Nif.nx_ggml_builder_add_add(builder, larger_index, smaller_index), memo}
  end

  defp do_lower(_builder, op, _args, _t, _param_indices, _memo) do
    raise NxGgml.UnsupportedOpError, op: op
  end

  defp check_dtype!(type) when type in @supported_dtypes, do: :ok
  defp check_dtype!(type), do: raise(NxGgml.UnsupportedDTypeError, type: type)
end
