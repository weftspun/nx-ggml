defmodule NxGgml.Compiler do
  @moduledoc """
  The `Nx.Defn.Compiler` that lowers a traced `Nx.Defn.Expr` into a ggml
  compute graph (`ggml_cgraph`), cached by input shape/dtype signature, and
  executed on CPU or Vulkan via ggml's backends.

  This is the *only* path from Nx tensor operations to ggml: ad hoc
  (non-`defn`) tensor ops are expected to route through this same compiler
  as trivial one-op `defn`s, rather than through a separate hand-written
  eager backend.

  Differentiability comes for free: `Nx.Defn.Grad` rewrites the traced
  expression graph one level above this compiler, using per-primitive
  gradient rules already in Nx core, so this module never needs to
  implement backward passes.

  `NxGgml.ExprLowering` lowers a real but intentionally partial op/dtype
  subset (see its moduledoc), and only when the traced function returns a
  single tensor (not yet a tuple/map of tensors). Anything outside that
  subset — an unimplemented op, an unsupported dtype, or a composite
  (non-single-tensor) output — raises
  `NxGgml.UnsupportedOpError`/`NxGgml.UnsupportedDTypeError`, which this
  module catches and falls back to `Nx.Defn.Evaluator` (Nx's own
  backend-agnostic tree-walking interpreter). This fallback is explicit and
  intentional, not silent: it's how real ggml coverage broadens
  incrementally without ever leaving a `defn` unable to run.

  ## Device selection

  Pass `device: :cpu` (default) or `device: :vulkan` as a compiler option
  (see `NxGgml.Device`) to choose which ggml backend a graph runs on.
  `:vulkan` requires a build configured with `NX_GGML_VULKAN=ON` and a
  Vulkan-capable GPU found at NIF load (`NxGgml.Device.vulkan_available?/0`).
  """

  @behaviour Nx.Defn.Compiler

  @impl true
  def __partitions_options__(opts), do: [opts]

  @impl true
  def __to_backend__(opts), do: {NxGgml.Backend, opts}

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    device = NxGgml.Device.from_opts(opts)
    cache_key = NxGgml.GraphCache.key(key, vars, device)

    case NxGgml.GraphCache.fetch(cache_key) do
      {:ok, lowered} ->
        run(lowered)

      :error ->
        try do
          expr = fun.(vars)

          unless match?(%Nx.Tensor{}, expr) do
            raise NxGgml.UnsupportedOpError, op: :composite_output
          end

          expr
          |> then(&NxGgml.ExprLowering.build(vars, &1, device))
          |> then(&NxGgml.GraphCache.put(cache_key, &1))
          |> run()
        rescue
          e in [NxGgml.UnsupportedOpError, NxGgml.UnsupportedDTypeError] ->
            _ = e
            Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
        end
    end
  end

  defp run(%{compiled: compiled, out_shape: shape, out_type: type}) do
    fn [params] ->
      # `params` are zero-arity thunks (`(-> Nx.Tensor.t())`), matching
      # Nx.Defn.Evaluator's own convention (see its `Enum.fetch!(...).()`
      # usage) despite the `Nx.Defn.Compiler.__compile__/4` callback typespec
      # reading as plain tensors.
      binaries = Enum.map(params, fn thunk -> thunk.() |> Nx.to_binary() end)
      result_binary = NxGgml.Nif.nx_ggml_compiled_run(compiled, binaries)

      result =
        result_binary
        |> Nx.from_binary(type)
        |> Nx.reshape(shape)

      [result]
    end
  end
end
