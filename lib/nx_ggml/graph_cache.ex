defmodule NxGgml.GraphCache do
  @moduledoc """
  Caches lowered ggml graphs (`NxGgml.ExprLowering.build/2` results) keyed
  by `{defn key, input shapes, input dtypes}` (the `key` opaque term
  `Nx.Defn.Compiler` callbacks receive does not itself vary with input
  shapes/dtypes, so it must be combined with those here — see the
  `Nx.Defn.Compiler.__compile__/4` docs).

  Backed by `:persistent_term`: compiled graphs are read on every call and
  written rarely (once per unique shape/dtype signature), which is exactly
  the read-mostly access pattern `:persistent_term` is for. This is the
  tinygrad-`TinyJit`/EXLA-executable-cache pattern: trace + lower once per
  signature, then only re-run the cached native graph on repeat calls.
  """

  @doc "Builds a cache key from the compiler's opaque `key` and the traced `vars`."
  def key(key, vars) do
    {key, Enum.map(vars, &{&1.shape, &1.type})}
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
