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

  Phase 2 status: the real `Nx.Defn.Expr` -> `ggml_cgraph` lowering lands in
  Phase 3 (`native/nx_ggml/expr_lowering.cpp` + `graph_cache.cpp`). For now
  `__jit__`/`__compile__` delegate to `Nx.Defn.Evaluator` (Nx's own
  backend-agnostic tree-walking interpreter) purely so that the compiler
  behaviour is wired up and callable end-to-end; this delegation is removed
  once Phase 3 lands.
  """

  @behaviour Nx.Defn.Compiler

  @impl true
  def __partitions_options__(opts), do: [opts]

  @impl true
  def __to_backend__(opts), do: {NxGgml.Backend, opts}

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    Nx.Defn.Evaluator.__jit__(key, vars, fun, args_list, opts)
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
  end
end
