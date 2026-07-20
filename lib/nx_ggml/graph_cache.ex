defmodule NxGgml.GraphCache do
  @moduledoc """
  Caches lowered ggml graphs (`NxGgml.ExprLowering.build/3` results) keyed
  by `{defn key, input shapes, input dtypes, device}` (the `key` opaque
  term `Nx.Defn.Compiler` callbacks receive does not itself vary with
  input shapes/dtypes/device, so it must be combined with those here — see
  the `Nx.Defn.Compiler.__compile__/4` docs). Device is part of the key so
  the same traced function compiled for `:cpu` and `:vulkan` gets two
  independent cached graphs, not one clobbering the other.

  Backed by `:persistent_term`: compiled graphs are read on every call and
  written rarely (once per unique shape/dtype/device signature), which is
  exactly the read-mostly access pattern `:persistent_term` is for. This is
  the tinygrad-`TinyJit`/EXLA-executable-cache pattern: trace + lower once
  per signature, then only re-run the cached native graph on repeat calls.
  """

  @doc """
  Builds a cache key from the compiler's opaque `key`, the traced `vars`,
  and the device. `vars` mirrors the original `defn`'s arity, so a
  composite argument (a tuple/map, or a `Nx.Container`-implementing struct
  like `Axon.ModelState`) appears as one nested element, not a flat
  tensor -- flattened via `Nx.Defn.Composite.flatten_list/1` first (the
  same leaf ordering `expr`'s `:parameter` node indices and the runtime
  `params` list use) so every leaf actually has a `.shape`/`.type` to key
  on.
  """
  def key(key, vars, device) do
    flat_vars = Nx.Defn.Composite.flatten_list(vars)
    {key, Enum.map(flat_vars, &{&1.shape, &1.type}), device}
  end

  @doc "Looks up a previously-cached lowered graph, or `:error` on a miss."
  def fetch(cache_key) do
    case :persistent_term.get({__MODULE__, cache_key}, :__not_found__) do
      :__not_found__ -> :error
      value -> {:ok, value}
    end
  end

  @doc "Stores a lowered graph under `cache_key`, returning it for chaining."
  def put(cache_key, value) do
    :persistent_term.put({__MODULE__, cache_key}, value)
    value
  end
end
