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

CPU and Vulkan GPU paths both working end-to-end, including real GPU-trained gradient descent
(tested on an NVIDIA RTX 4090). Any op or dtype not yet lowered falls back to `Nx.Defn.Evaluator`
automatically, so nothing breaks; op coverage grows over time. See `native/nx_ggml/` for the C/NIF
layer and `lib/nx_ggml/` for the Elixir side.

### Dtype support

`{:f, 32}` for all compute. `{:s, 32}` is also supported, but *only* for a defn parameter that is
used exclusively as `gather`'s index operand (see below) — never for general arithmetic, and never
for a bare `{:s, 32}` constant (which is always re-encoded as f32 regardless of its traced dtype,
since a `:constant` node is just a number, not real tensor data — this is what makes `Nx.mean` work,
whose `sum/n` decomposition introduces a `{:s, 32}` constant divisor that must still combine with an
f32 sum). Every other dtype, and any non-gather-index use of `{:s, 32}`, falls back to
`Nx.Defn.Evaluator`.

### Op coverage (lowered to ggml; see `lib/nx_ggml/expr_lowering.ex`)

- **Elementwise binary**: `add`, `subtract`, `multiply`, `divide` (broadcasting supported)
- **Elementwise unary**: `negate`, `abs`, `sign`, `sqrt`, `exp`, `log`, `sigmoid`, `tanh`, `sin`, `cos`
- **Shape**: `reshape`, `squeeze`, `transpose` (any rank ≤ 4, arbitrary permutation), `broadcast`
  (trailing-aligned case only), `concatenate` (any number of tensors, any axis)
- **Linear algebra**: `dot` (matmul, optionally batched — contract the last axis of `a` with the
  second-to-last axis of `b`, with any number of matching leading batch axes on both operands),
  `sum` (full reduction to scalar, or reduction over just the last axis — the latter also gives
  `Nx.mean` "for free" since it's composed from `sum` + a constant `divide` at the tracing level,
  not a raw primitive), `reduce_max` (last axis only, via `ggml_pool_1d`'s global max-pool — ggml
  has no dedicated row-max reduction op), `clip` (literal/constant bounds only)
- **Indexed**: `gather` (embedding-lookup shape only: a 2-D `(vocab, dim)` table, gathering whole
  rows by an `{:s, 32}` index tensor that is *directly* a defn parameter — not a computed
  expression, and not Nx's fully general nd-index semantics)

Because `reduce_max`, `subtract`, `exp`, `sum` (last axis), and `divide` are all independently
lowerable, a user-composed **numerically-stable softmax** — `x |> subtract(reduce_max(x)) |> exp()
|> then(&(&1 / sum(&1)))` — lowers entirely through ggml even though Nx has no raw `:softmax`
primitive for this module to intercept directly.

Everything else (comparisons, `pow`, multi-axis (non-last-axis) reductions, `select`/
`indexed_add`/`indexed_put`/`argmax`/`argmin`/`sort`, `triangular_solve`, `fft`/`ifft`, general
nd-gather, the generic `reduce` callback with an arbitrary reducer, non-f32 dtypes) currently falls
back to `Nx.Defn.Evaluator`. Op priority has been grounded in real usage data — tallying `ggml_*`
call frequency in [trellis2cpp](https://github.com/weftspun/trellis2cpp) (a transformer-heavy
image-to-3D ggml pipeline) rather than mechanically working through the full `Nx.Backend` callback
list.

## Installation

Not yet published to Hex.

```elixir
def deps do
  [
    {:nx_ggml, git: "https://github.com/weftspun/nx-ggml"}
  ]
end
```
