defmodule NxGgml.UnaryOpsTest do
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxGgml.Conformance

  defn negate_(x), do: -x
  defn abs_(x), do: Nx.abs(x)
  defn sign_(x), do: Nx.sign(x)
  defn sqrt_(x), do: Nx.sqrt(x)
  defn exp_(x), do: Nx.exp(x)
  defn log_(x), do: Nx.log(x)
  defn sigmoid_(x), do: Nx.sigmoid(x)
  defn tanh_(x), do: Nx.tanh(x)
  defn sin_(x), do: Nx.sin(x)
  defn cos_(x), do: Nx.cos(x)

  @x Nx.tensor([-2.0, -1.0, 0.5, 1.0, 2.0], type: :f32)
  @positive Nx.tensor([0.5, 1.0, 2.0, 4.0], type: :f32)

  test "negate matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&negate_/1, [@x])
  test "abs matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&abs_/1, [@x])
  test "sign matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&sign_/1, [@x])
  test "sqrt matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&sqrt_/1, [@positive], close: true)
  test "exp matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&exp_/1, [@x], close: true)
  test "log matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&log_/1, [@positive], close: true)

  test "sigmoid matches Nx.Defn.Evaluator",
    do: assert_matches_evaluator(&sigmoid_/1, [@x], close: true)

  test "tanh matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&tanh_/1, [@x], close: true)
  test "sin matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&sin_/1, [@x], close: true)
  test "cos matches Nx.Defn.Evaluator", do: assert_matches_evaluator(&cos_/1, [@x], close: true)
end
