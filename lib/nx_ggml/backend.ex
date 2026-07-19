defmodule NxGgml.Backend do
  @moduledoc """
  `Nx.Backend` implementation for tensor storage under `NxGgml.Compiler`.

  This module intentionally does **not** implement compute callbacks by
  hand: `NxGgml.Compiler` (an `Nx.Defn.Compiler`) is the only path that
  turns Nx tensor operations into ggml compute graphs. Every `Nx.Backend`
  compute callback below is a generated stub that raises, pointing callers
  at `NxGgml.Compiler` instead — this is intentional, not a Phase 3 TODO
  list, since ad hoc (non-`defn`) tensor ops are expected to route through
  the compiler as trivial one-op `defn`s rather than through a hand-written
  eager implementation here (see the design note in `NxGgml.Compiler`).

  Only the tensor lifecycle callbacks are implemented here (creation,
  binary transfer, deallocation) — these land for real in Phase 3
  (currently backed by a plain Elixir binary; Phase 3 replaces this with a
  ggml-backed resource so that `NxGgml.Compiler`-executed graphs don't pay a
  transfer round-trip on every call).
  """

  @behaviour Nx.Backend

  defstruct [:state]

  @impl true
  def init(opts), do: opts

  @impl true
  def constant(%Nx.Tensor{shape: shape} = out, scalar, backend_options) do
    binary = out |> Nx.BinaryBackend.constant(scalar, []) |> Nx.BinaryBackend.to_binary(Nx.size(shape))
    from_binary(out, binary, backend_options)
  end

  @impl true
  def from_binary(out, binary, _backend_options) when is_binary(binary) do
    %{out | data: %__MODULE__{state: binary}}
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{state: state}}, limit) do
    total = byte_size(state)

    if limit == nil or limit * byte_size(state) >= total do
      state
    else
      binary_part(state, 0, limit)
    end
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{}}), do: :ok

  @impl true
  def backend_copy(tensor, __MODULE__, _opts), do: tensor

  def backend_copy(%Nx.Tensor{data: %__MODULE__{state: state}} = tensor, backend, opts) do
    backend.from_binary(tensor, state, opts)
  end

  @impl true
  def backend_transfer(tensor, __MODULE__, _opts), do: tensor

  def backend_transfer(%Nx.Tensor{data: %__MODULE__{state: state}} = tensor, backend, opts) do
    backend.from_binary(tensor, state, opts)
  end

  @impl true
  def inspect(%Nx.Tensor{data: %__MODULE__{state: state}} = tensor, inspect_opts) do
    Nx.Backend.inspect(tensor, state, inspect_opts)
  end

  # All remaining Nx.Backend callbacks (elementwise, reductions, linalg,
  # ...) are intentionally not implemented here; see moduledoc. Generated
  # so we never silently drift from the behaviour's actual callback list.
  for {name, arity} <- Nx.Backend.behaviour_info(:callbacks),
      not Module.defines?(__MODULE__, {name, arity}) do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(name)(unquote_splicing(args)) do
      raise "#{unquote(name)}/#{unquote(arity)} is not implemented by #{inspect(__MODULE__)}. " <>
              "Use NxGgml.Compiler (via Nx.Defn.jit/2 or defn) instead of calling this op eagerly."
    end
  end
end
