defmodule NxGgml.ShapeOpsTest do
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxGgml.Conformance

  defn reshape_to(x, shape), do: Nx.reshape(x, shape)
  defn transpose_(x), do: Nx.transpose(x)
  defn transpose_axes(x, axes), do: Nx.transpose(x, axes: axes)
  defn squeeze_(x), do: Nx.squeeze(x)
  defn broadcast_(x, shape), do: Nx.broadcast(x, shape)

  test "reshape (2,3) -> (3,2) matches Nx.Defn.Evaluator and preserves row-major order" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
    fun = fn t -> reshape_to(t, {3, 2}) end
    result = assert_matches_evaluator(fun, [x])
    assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert Nx.shape(result) == {3, 2}
  end

  test "reshape (6,) -> (2,3) matches Nx.Defn.Evaluator" do
    x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type: :f32)
    fun = fn t -> reshape_to(t, {2, 3}) end
    assert_matches_evaluator(fun, [x])
  end

  test "2D transpose hand-verified against a known asymmetric matrix" do
    # [[1,2,3],      [[1,4],
    #  [4,5,6]]  ->   [2,5],
    #                 [3,6]]
    x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
    result = assert_matches_evaluator(&transpose_/1, [x])
    assert Nx.shape(result) == {3, 2}
    assert Nx.to_flat_list(result) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
  end

  test "3D transpose with a non-trivial axis permutation matches Nx.Defn.Evaluator" do
    x =
      Enum.to_list(1..24)
      |> Enum.map(&(&1 * 1.0))
      |> Nx.tensor(type: :f32)
      |> Nx.reshape({2, 3, 4})

    fun = fn t -> transpose_axes(t, [2, 0, 1]) end
    result = assert_matches_evaluator(fun, [x])
    assert Nx.shape(result) == {4, 2, 3}
  end

  test "squeeze drops size-1 axes and matches Nx.Defn.Evaluator" do
    x = Nx.tensor([[[1.0, 2.0, 3.0]]], type: :f32)
    result = assert_matches_evaluator(&squeeze_/1, [x])
    assert Nx.shape(result) == {3}
  end

  test "broadcast (trailing-aligned) matches Nx.Defn.Evaluator" do
    x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    fun = fn t -> broadcast_(t, {2, 3}) end
    result = assert_matches_evaluator(fun, [x])
    assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
  end
end
