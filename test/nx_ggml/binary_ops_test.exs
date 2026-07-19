defmodule NxGgml.BinaryOpsTest do
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxGgml.Conformance

  defn sub(x, y), do: x - y
  defn mul(x, y), do: x * y
  defn div_(x, y), do: x / y
  defn sub_scalar_first(x), do: 10.0 - x
  defn div_scalar_first(x), do: 10.0 / x
  defn mul_broadcast(x, y), do: x * y

  test "subtract matches Nx.Defn.Evaluator" do
    x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    y = Nx.tensor([10.0, 20.0, 30.0], type: :f32)
    assert_matches_evaluator(&sub/2, [x, y])
  end

  test "multiply matches Nx.Defn.Evaluator" do
    x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    y = Nx.tensor([10.0, 20.0, 30.0], type: :f32)
    assert_matches_evaluator(&mul/2, [x, y])
  end

  test "divide matches Nx.Defn.Evaluator" do
    x = Nx.tensor([10.0, 20.0, 30.0], type: :f32)
    y = Nx.tensor([2.0, 4.0, 5.0], type: :f32)
    assert_matches_evaluator(&div_/2, [x, y])
  end

  test "non-commutative op with the constant operand first (10.0 - x)" do
    x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    result = assert_matches_evaluator(&sub_scalar_first/1, [x])
    assert Nx.to_flat_list(result) == [9.0, 8.0, 7.0]
  end

  test "division with the constant operand first (10.0 / x)" do
    x = Nx.tensor([1.0, 2.0, 5.0], type: :f32)
    result = assert_matches_evaluator(&div_scalar_first/1, [x])
    assert Nx.to_flat_list(result) == [10.0, 5.0, 2.0]
  end

  test "broadcasting multiply (row vector times column vector)" do
    x = Nx.tensor([[1.0], [2.0], [3.0]], type: :f32)
    y = Nx.tensor([[10.0, 20.0]], type: :f32)
    assert_matches_evaluator(&mul_broadcast/2, [x, y])
  end
end
