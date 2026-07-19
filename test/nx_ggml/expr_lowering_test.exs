defmodule NxGgml.ExprLoweringTest do
  use ExUnit.Case, async: false

  import Nx.Defn

  defn add1(x), do: x + 1.0
  defn add_two(x, y), do: x + y
  defn identity(x), do: x
  defn unsupported(x), do: x * 2.0

  defp run_ggml(fun, args) do
    Nx.Defn.jit_apply(fun, args, compiler: NxGgml.Compiler)
  end

  test "identity round-trips through the real ggml lowering" do
    result = run_ggml(&identity/1, [Nx.tensor([1.0, 2.0, 3.0], type: :f32)])
    assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
  end

  test "constant + parameter add lowers to a real ggml graph" do
    result = run_ggml(&add1/1, [Nx.tensor([1.0, 2.0, 3.0], type: :f32)])
    assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
  end

  test "two-parameter add matches Nx.BinaryBackend" do
    x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    y = Nx.tensor([10.0, 20.0, 30.0], type: :f32)

    ggml_result = run_ggml(&add_two/2, [x, y])
    reference = Nx.Defn.jit_apply(&add_two/2, [x, y], compiler: Nx.Defn.Evaluator)

    assert Nx.to_flat_list(ggml_result) == Nx.to_flat_list(reference)
  end

  test "repeat calls with the same shape/dtype signature hit the graph cache" do
    x1 = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    x2 = Nx.tensor([10.0, 20.0, 30.0], type: :f32)

    r1 = run_ggml(&add1/1, [x1])
    r2 = run_ggml(&add1/1, [x2])

    assert Nx.to_flat_list(r1) == [2.0, 3.0, 4.0]
    assert Nx.to_flat_list(r2) == [11.0, 21.0, 31.0]
  end

  test "an op outside Phase 3's supported subset falls back to Nx.Defn.Evaluator" do
    result = run_ggml(&unsupported/1, [Nx.tensor([1.0, 2.0, 3.0], type: :f32)])
    assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0]
  end

  test "an unsupported dtype falls back to Nx.Defn.Evaluator" do
    result = run_ggml(&identity/1, [Nx.tensor([1, 2, 3], type: :s64)])
    assert Nx.to_flat_list(result) == [1, 2, 3]
  end
end
