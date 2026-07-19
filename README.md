# NxGgml

A JIT-compiled [Nx](https://github.com/elixir-nx/nx) backend that lowers `Nx.Defn` expression
graphs into [ggml](https://github.com/ggml-org/ggml) compute graphs (`ggml_cgraph`), executed on
CPU or Vulkan GPU compute.

## Design

- **JIT-only.** A single `Nx.Defn.Compiler` (`NxGgml.Compiler`) lowers a traced `Nx.Defn.Expr`
  into one `ggml_cgraph`, cached by input shape/dtype signature. There is no separate hand-written
  eager backend — ad hoc (non-`defn`) tensor ops route through the same compiler as trivial
  one-op `defn`s, so there is exactly one code path from Nx ops to ggml graphs.
- **Differentiable for training, with zero backend-side autodiff code.** `Nx.Defn.Grad`
  differentiates the traced expression graph itself, one level above any backend, using
  per-primitive gradient rules already in Nx core. Implementing forward ops correctly here is
  sufficient for `Nx.Defn.grad`/`value_and_grad` to work.
- **Vulkan GPU compute + CPU**, via ggml's existing backends. No Metal.
- ggml is vendored directly from upstream `ggml-org/ggml` via `git subtree`
  (`native/nx_ggml/third_party/ggml`) — untouched, not routed through any downstream fork.

## Status

Early bootstrap. See `native/nx_ggml/` for the C/NIF layer and `lib/nx_ggml/` for the Elixir side.

## Installation

Not yet published to Hex.

```elixir
def deps do
  [
    {:nx_ggml, git: "https://github.com/weftspun/nx-ggml"}
  ]
end
```
