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

CPU path working end-to-end (Vulkan not yet enabled — Phase 7). Any op or dtype not yet lowered
falls back to `Nx.Defn.Evaluator` automatically, so nothing breaks; op coverage grows over time.
See `native/nx_ggml/` for the C/NIF layer and `lib/nx_ggml/` for the Elixir side.

### Dtype support

Only `{:f, 32}` is lowered to ggml today. Every other dtype falls back to `Nx.Defn.Evaluator`.

### Op coverage (lowered to ggml; see `lib/nx_ggml/expr_lowering.ex`)

- **Elementwise binary**: `add`, `subtract`, `multiply`, `divide` (broadcasting supported)
- **Elementwise unary**: `negate`, `abs`, `sign`, `sqrt`, `exp`, `log`, `sigmoid`, `tanh`, `sin`, `cos`
- **Shape**: `reshape`, `squeeze`, `transpose` (any rank ≤ 4, arbitrary permutation), `broadcast`
  (trailing-aligned case only)
- **Linear algebra**: `dot` (standard 2-D matmul only: contract `a`'s last axis with `b`'s first
  axis, no batch dims), `sum` (full reduction to scalar only), `clip` (literal/constant bounds only)

Everything else (comparisons, `pow`, multi-axis reductions, batched/higher-rank `dot`,
`select`/`gather`/`indexed_add`/`indexed_put`/`argmax`/`argmin`/`sort`, `triangular_solve`,
`fft`/`ifft`, the generic `reduce` callback with an arbitrary reducer, non-f32 dtypes) currently
falls back to `Nx.Defn.Evaluator`. Op priority so far has been grounded in real usage data —
tallying `ggml_*` call frequency in [trellis2cpp](https://github.com/weftspun/trellis2cpp) (a
transformer-heavy image-to-3D ggml pipeline) rather than mechanically working through the full
`Nx.Backend` callback list.

## Installation

Not yet published to Hex.

```elixir
def deps do
  [
    {:nx_ggml, git: "https://github.com/weftspun/nx-ggml"}
  ]
end
```
