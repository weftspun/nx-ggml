defmodule NxGgml.LinalgTest do
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxGgml.Conformance

  defn matmul(a, b), do: Nx.dot(a, b)
  defn sum_all(x), do: Nx.sum(x)
  defn clip_(x), do: Nx.clip(x, -1.0, 1.0)

  test "2x2 identity matmul is the identity" do
    identity = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
    result = assert_matches_evaluator(&matmul/2, [identity, x])
    assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0]
  end

  test "hand-verified asymmetric 2x3 @ 3x2 matmul" do
    # A = [[1,2,3],       B = [[ 7, 8],
    #      [4,5,6]]            [ 9,10],
    #                           [11,12]]
    #
    # A @ B = [[1*7+2*9+3*11,  1*8+2*10+3*12],   = [[ 58,  64],
    #          [4*7+5*9+6*11,  4*8+5*10+6*12]]      [139, 154]]
    a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
    b = Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], type: :f32)

    result = assert_matches_evaluator(&matmul/2, [a, b])
    assert Nx.shape(result) == {2, 2}
    assert Nx.to_flat_list(result) == [58.0, 64.0, 139.0, 154.0]
  end

  test "non-square matmul (3x4 @ 4x2) matches Nx.Defn.Evaluator" do
    a =
      Enum.to_list(1..12)
      |> Enum.map(&(&1 * 1.0))
      |> Nx.tensor(type: :f32)
      |> Nx.reshape({3, 4})

    b =
      Enum.to_list(1..8)
      |> Enum.map(&(&1 * 1.0))
      |> Nx.tensor(type: :f32)
      |> Nx.reshape({4, 2})

    result = assert_matches_evaluator(&matmul/2, [a, b])
    assert Nx.shape(result) == {3, 2}
  end

  test "full sum reduction matches Nx.Defn.Evaluator" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
    result = assert_matches_evaluator(&sum_all/1, [x])
    assert Nx.to_number(result) == 21.0
  end

  test "clip with literal bounds matches Nx.Defn.Evaluator" do
    x = Nx.tensor([-5.0, -0.5, 0.0, 0.5, 5.0], type: :f32)
    result = assert_matches_evaluator(&clip_/1, [x])
    assert Nx.to_flat_list(result) == [-1.0, -0.5, 0.0, 0.5, 1.0]
  end
end
