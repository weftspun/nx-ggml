defmodule NxGgml.ErfTest do
  @moduledoc """
  `Nx.erf` lowering (`ggml_map_custom1` wrapping `erff()` — ggml has no
  standalone erf op), added to unblock exact (erf-based) GELU for the
  DINOv3 MLP block -- see `scratch_dino_mlp.exs` at the project root for the
  real-weights validation script.
  """

  use ExUnit.Case, async: false

  import Nx.Defn
  import NxGgml.Conformance

  defn erf(x), do: Nx.erf(x)

  defn gelu_exact(x) do
    x * 0.5 * (1.0 + Nx.erf(x / Nx.sqrt(2.0)))
  end

  test "erf matches Nx.Defn.Evaluator" do
    x = Nx.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], type: :f32)
    result = assert_matches_evaluator(&erf/1, [x], close: true)
    assert Nx.shape(result) == {7}
  end

  test "exact (erf-based) GELU, composed, matches Nx.Defn.Evaluator" do
    x = Nx.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], type: :f32)
    result = assert_matches_evaluator(&gelu_exact/1, [x], close: true)
    assert Nx.shape(result) == {5}
  end
end
