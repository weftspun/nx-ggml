# Periodic performance check (not just documentation): times a small
# elementwise chain via NxGgml.Compiler against Nx.BinaryBackend directly,
# at a few tensor sizes. ggml is only worth routing through if it's
# actually faster than a pure-Elixir/BinaryBackend baseline; this should be
# re-run after each phase that changes the lowering or graph-cache path
# (see the "Profiling practice" section of the project plan).
#
# Run with: mix run bench/elementwise_bench.exs

defmodule Bench do
  import Nx.Defn

  defn chain(x) do
    x
    |> Nx.multiply(2.0)
    |> Nx.subtract(1.0)
    |> Nx.sigmoid()
    |> Nx.add(Nx.tanh(x))
  end

  def run(size, iterations) do
    values = for _ <- 1..size, do: :rand.uniform()
    x = Nx.tensor(values, type: :f32)

    # Warm the graph cache (first call traces + lowers; ignore its cost).
    Nx.Defn.jit_apply(&chain/1, [x], compiler: NxGgml.Compiler)

    {ggml_us, _} =
      :timer.tc(fn ->
        for _ <- 1..iterations do
          Nx.Defn.jit_apply(&chain/1, [x], compiler: NxGgml.Compiler)
        end
      end)

    {binary_us, _} =
      :timer.tc(fn ->
        for _ <- 1..iterations do
          Nx.Defn.jit_apply(&chain/1, [x], compiler: Nx.Defn.Evaluator, backend: Nx.BinaryBackend)
        end
      end)

    ggml_per_call = ggml_us / iterations
    binary_per_call = binary_us / iterations

    IO.puts(
      "size=#{size |> to_string() |> String.pad_leading(7)}  " <>
        "ggml=#{Float.round(ggml_per_call, 1)}us  " <>
        "binary=#{Float.round(binary_per_call, 1)}us  " <>
        "speedup=#{Float.round(binary_per_call / ggml_per_call, 2)}x"
    )
  end
end

for size <- [10, 1_000, 100_000] do
  Bench.run(size, 200)
end
