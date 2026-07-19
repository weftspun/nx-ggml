defmodule NxGgml.UnsupportedOpError do
  @moduledoc """
  Raised by `NxGgml.ExprLowering` when a traced `Nx.Defn.Expr` node uses an
  op not yet lowered to a ggml graph. `NxGgml.Compiler` catches this and
  falls back to `Nx.Defn.Evaluator` (see its moduledoc for why that's a
  temporary, explicitly-tracked degradation rather than a silent one).
  """
  defexception [:op]

  @impl true
  def message(%{op: op}) do
    "op #{inspect(op)} is not yet lowered to ggml by NxGgml.Compiler"
  end
end

defmodule NxGgml.UnsupportedDTypeError do
  @moduledoc """
  Raised by `NxGgml.ExprLowering` when a tensor's dtype is outside the
  v1-supported subset (see the dtype table in the project plan/README).
  `NxGgml.Compiler` catches this and falls back to `Nx.Defn.Evaluator`.
  """
  defexception [:type]

  @impl true
  def message(%{type: type}) do
    "dtype #{inspect(type)} is not yet supported by NxGgml.Compiler"
  end
end
