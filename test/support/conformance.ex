defmodule NxGgml.Conformance do
  @moduledoc """
  Runs the same `Nx.Defn` function through `NxGgml.Compiler` and
  `Nx.Defn.Evaluator` (Nx's own backend-agnostic reference interpreter) and
  asserts the results match — the conformance-testing pattern used
  throughout the test suite (see the project plan's Phase 6).
  """

  import ExUnit.Assertions

  @doc "Asserts `fun` produces the same result via NxGgml.Compiler and Nx.Defn.Evaluator."
  def assert_matches_evaluator(fun, args, opts \\ []) do
    ggml_result = Nx.Defn.jit_apply(fun, args, compiler: NxGgml.Compiler)
    reference = Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)

    if Keyword.get(opts, :close, false) do
      assert Nx.all_close(ggml_result, reference, atol: 1.0e-5, rtol: 1.0e-5) == Nx.tensor(1, type: :u8)
    else
      assert Nx.to_flat_list(ggml_result) == Nx.to_flat_list(reference)
    end

    ggml_result
  end
end
