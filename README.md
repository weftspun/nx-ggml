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

`{:f, 32}` for all compute. `{:s, 32}` is also supported, but _only_ for a defn parameter that is
used exclusively as `gather`'s index operand (see below) — never for general arithmetic, and never
for a bare `{:s, 32}` constant (which is always re-encoded as f32 regardless of its traced dtype,
since a `:constant` node is just a number, not real tensor data — this is what makes `Nx.mean` work,
whose `sum/n` decomposition introduces a `{:s, 32}` constant divisor that must still combine with an
f32 sum). Every other dtype, and any non-gather-index use of `{:s, 32}`, falls back to
`Nx.Defn.Evaluator`.

### Op coverage (lowered to ggml; see `lib/nx_ggml/expr_lowering.ex`)

- **Elementwise binary**: `add`, `subtract`, `multiply`, `divide` (broadcasting supported)
- **Elementwise unary**: `negate`, `abs`, `sign`, `sqrt`, `exp`, `log`, `sigmoid`, `tanh`, `sin`, `cos`,
  `erf` (CPU only — ggml has no standalone erf op, only the fused `gelu_erf` activation, so this
  wraps the C standard library's `erff()` as a custom `ggml_map_custom1` callback; custom map ops
  aren't portable to the Vulkan backend, so an `erf`-containing graph compiled for `:vulkan` falls
  back to `Nx.Defn.Evaluator` instead of silently miscomputing)
- **Shape**: `reshape`, `squeeze`, `transpose` (any rank ≤ 4, arbitrary permutation), `broadcast`
  (trailing-aligned case only), `concatenate` (any number of tensors, any axis)
- **Linear algebra**: `dot` (matmul, optionally batched — contract the last axis of `a` with the
  second-to-last axis of `b`, with any number of matching leading batch axes on both operands),
  `sum` (full reduction to scalar, or reduction over just the last axis — the latter also gives
  `Nx.mean` "for free" since it's composed from `sum` + a constant `divide` at the tracing level,
  not a raw primitive), `reduce_max` (last axis only, via `ggml_pool_1d`'s global max-pool — ggml
  has no dedicated row-max reduction op), `clip` (literal/constant bounds only)
- **Indexed**: `gather` (embedding-lookup shape only: a 2-D `(vocab, dim)` table, gathering whole
  rows by an `{:s, 32}` index tensor that is _directly_ a defn parameter — not a computed
  expression, and not Nx's fully general nd-index semantics)
- **`conv`** (`ggml_conv_2d`): standard rank-4 NCHW input / OIHW kernel only (Nx's default axis
  order), dilation 1, no feature/batch grouping — covers both ordinary convs and "patchify" convs
  (stride == kernel size, no padding, e.g. a ViT patch embedding)

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

### Validated against a real model, with real trained weights

The `scratch_dino_*.exs`/`scratch_ssflow.exs` validation scripts referenced below have moved to a
separate sibling project, [`trellis2_ex`](../trellis2_ex) (depends on this repo via a `mix` git
dependency) — nx-ggml itself stays a general-purpose `Nx.Defn` → ggml compiler with no model-specific
code. The results described here are unchanged by the move; only the scripts' location did.

`scratch_dino_patch_embed.exs` ports trellis2cpp's DINOv3 ViT-L/16 **patch embedding** stage
(patch conv → bias → CLS/register token concat — the first neural stage of the pipeline) to
Elixir/`Nx.Defn`, loads the real downloaded DINOv3 weights (converted to GGUF the same way
trellis2cpp's own `convert_dino_to_gguf.py` does) via a small `gguf.h`-backed NIF
(`NxGgml.Nif.nx_ggml_gguf_read_f32/2`), and checks the result against a real PyTorch-computed
reference activation (trellis2cpp's own `scripts/dump_dino_reference.py`, using the real
`DINOv3ViTModel` from `transformers`). Result: L2 norms agree to 4 decimal places (955.8098 vs
955.8103), mean absolute error 1.8e-4, and 99.994% of the 1,053,696 output elements pass
trellis2cpp's own exact parity gate (`|got-ref| <= atol + rtol*|ref|`, `atol=rtol=2e-3`) — the
tiny remainder (64 elements) is consistent with ordinary fp32 accumulation-order differences
between ggml's im2col+GEMM conv implementation and PyTorch's native conv kernel, not a bug in this
port. This is real evidence the architecture and op lowering are correct against an actual trained
model, not just synthetic op tests — see the project plan for the full writeup, including how the
weights were obtained (an ungated Hugging Face mirror, since the official DINOv3 checkpoint is
license-gated) and the download/conversion/reference-generation steps.

Two further layer-0 stages have since been validated the same way, composed entirely from
already-supported ops (no new native code beyond `erf` itself):

- **`norm1`** (`scratch_dino_norm1.exs`, standard affine LayerNorm, composed from `mean`/`subtract`/
  `multiply`/`sqrt`/`divide`): max abs diff `3.8e-6`, **0 / 1,053,696** elements fail the gate — a
  clean pass.
- **`mlp`** (`scratch_dino_mlp.exs`, `up_proj` matmul+bias → exact erf-based GELU → `down_proj`
  matmul+bias, checked against the real `l0.mlp` tap): max abs diff `0.0039`, mean abs error `3.5e-7`,
  **0 / 1,053,696** elements fail the gate. This is the first validation to exercise the new `erf` op
  (`ggml_map_custom1` wrapping `erff()`, since ggml has no standalone erf op) — GELU has no other
  lowerable formulation in this backend, so this is real evidence the custom-op escape hatch works
  correctly against real weights, not just a synthetic case.
- **`attention`** (`scratch_dino_attention.exs`, full multi-head self-attention with axial 2D RoPE,
  checked against the real `l0.attention` tap): max abs diff `9.5e-6`, mean abs error `1.8e-7`,
  **0 / 1,053,696** elements fail the gate. Needed **zero new native ops** — RoPE's "rotate half the
  head dim" is normally a slice+concat, which this backend doesn't support, so it's instead expressed
  as a fixed `(head_dim × head_dim)` constant matmul (`rotate_half(x) = x @ R`, where `R` is the
  linear map for `concat(-x2, x1)` given `x = concat(x1, x2)`), and "RoPE only on patch tokens, CLS/
  register prefix passes through" is folded into one all-token elementwise rope by giving the prefix
  rows an identity rotation (`cos=1, sin=0`) instead of slicing a prefix back in. Multi-head attention
  itself is entirely `transpose` + batched `dot` + the already-proven softmax composition.
- **`layer0`** (`scratch_dino_layer0.exs`, the _entire_ layer-0 transformer block — norm1 → attention
  → layer_scale1 → residual → norm2 → MLP → layer_scale2 → residual — run end to end from the real
  `embd` patch-embedding output and checked against the real `l0.out` tap): max abs diff `0.031`, mean
  abs error `1.6e-7`, **0 / 1,053,696** elements fail the gate. This is every stage above composed into
  one compiled graph — real evidence a complete real transformer layer runs correctly through this
  backend, not just its individual sub-blocks in isolation.
- **The full encoder** (`scratch_dino_full.exs`, all 24 transformer layers plus the final affine-free
  LayerNorm, run end to end from the real patch embedding through to the real `cond` reference tap —
  trellis2cpp's own name for the DINOv3 encoder's final output): max abs diff `4.3e-5`, mean abs error
  `1.3e-6`, **0 / 1,053,696** elements fail the gate. No new lowering capability beyond `layer0` — all
  24 layers share one compiled graph (`NxGgml.GraphCache` keys on shape/dtype/device, not tensor
  values), replayed once per layer with that layer's own weights. This is the complete, real DINOv3
  ViT-L/16 vision encoder running end to end through this backend on real trained weights.

### Formal correctness proofs (Lean 4, test-time only)

`lean/` holds Lean 4 proofs backing nontrivial algebraic claims made by the op lowering — a
correctness-verification tool, never a runtime dependency (no NIF or `mix compile`/`mix test` step
touches it). `lean/RopeProof.lean` proves the RoPE "rotate_half via constant matmul" trick used in
the attention/layer scripts above is exactly correct for every `head_dim` and every input vector, not
just the specific vectors exercised by the Elixir test suite — see `lean/README.md`.

## Installation

Not yet published to Hex.

```elixir
def deps do
  [
    {:nx_ggml, git: "https://github.com/weftspun/nx-ggml"}
  ]
end
```
