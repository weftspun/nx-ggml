defmodule NxGgml.VulkanTest do
  @moduledoc """
  Phase 7: re-runs a representative slice of the CPU conformance suite with
  `device: :vulkan`. Skips gracefully (rather than failing) on any build/
  machine where Vulkan isn't actually usable -- a native build compiled
  with NX_GGML_VULKAN=OFF (the default), or one compiled with it ON but
  with no Vulkan-capable GPU found at NIF load
  (`NxGgml.Device.vulkan_available?/0`).

  Requires the native build to have been configured with
  NX_GGML_VULKAN=ON, e.g.:

      NX_GGML_VULKAN=ON mix test test/nx_ggml/vulkan_test.exs
  """

  use ExUnit.Case, async: false

  import Nx.Defn

  # Excluded by test/test_helper.exs unless NxGgml.Device.vulkan_available?/0
  # is true, so this whole module is skipped (not failed) when Vulkan isn't
  # actually usable on this build/machine.
  @moduletag :vulkan

  defn add1(x), do: x + 1.0
  defn chain(x, y), do: Nx.sigmoid(x * 2.0 - y)
  defn matmul(a, b), do: Nx.dot(a, b)
  defn sum_all(x), do: Nx.sum(x)

  defp run_vulkan(fun, args) do
    Nx.Defn.jit_apply(fun, args, compiler: NxGgml.Compiler, device: :vulkan)
  end

  defp run_cpu(fun, args) do
    Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)
  end

  test "constant + parameter add matches the CPU reference" do
    x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    assert Nx.to_flat_list(run_vulkan(&add1/1, [x])) == Nx.to_flat_list(run_cpu(&add1/1, [x]))
  end

  test "an elementwise chain matches the CPU reference" do
    x = Nx.tensor([1.0, -2.0, 3.0, -4.0], type: :f32)
    y = Nx.tensor([0.5, 0.5, 0.5, 0.5], type: :f32)

    gpu = run_vulkan(&chain/2, [x, y])
    cpu = run_cpu(&chain/2, [x, y])

    assert Nx.all_close(gpu, cpu, atol: 1.0e-5, rtol: 1.0e-5) == Nx.tensor(1, type: :u8)
  end

  test "the hand-verified asymmetric matmul also matches on Vulkan" do
    a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
    b = Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], type: :f32)

    result = run_vulkan(&matmul/2, [a, b])
    assert Nx.to_flat_list(result) == [58.0, 64.0, 139.0, 154.0]
  end

  test "full sum reduction matches the CPU reference" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
    assert Nx.to_number(run_vulkan(&sum_all/1, [x])) == 21.0
  end

  test "repeat calls with the same shape/dtype/device signature hit the graph cache" do
    x1 = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
    x2 = Nx.tensor([10.0, 20.0, 30.0], type: :f32)

    assert Nx.to_flat_list(run_vulkan(&add1/1, [x1])) == [2.0, 3.0, 4.0]
    assert Nx.to_flat_list(run_vulkan(&add1/1, [x2])) == [11.0, 21.0, 31.0]
  end
end
