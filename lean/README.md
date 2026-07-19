# Lean 4 correctness proofs (test-time only)

This directory holds formal proofs backing nontrivial algebraic claims made by nx-ggml's op
lowering — a correctness-verification tool used only while developing/reviewing this code, in the
spirit of `plausible-witness-dag`'s escalating-verification idea (see the project plan). **Lean is
not a runtime dependency**: nothing here is called during actual tensor computation, no NIF touches
it, and `mix compile`/`mix test` never invoke it. It exists purely so a claim like "this constant
matmul is *exactly* equivalent to a slice+negate+concat, for every input, not just the vectors we
happened to test" can be checked algebraically instead of only empirically.

No Mathlib dependency — everything is built from plain Lean 4 core plus the `omega` tactic that
ships with the toolchain, so it builds standalone and fast.

```
lean RopeProof.lean          # or: lake build
```

## `RopeProof.lean`

Proves that the RoPE "rotate_half via constant matmul" trick used in `scratch_dino_attention.exs` /
`scratch_dino_layer0.exs` / `scratch_dino_full.exs` (and mirrored in
`test/nx_ggml/attention_test.exs`) is exactly correct — not just numerically close on the specific
test vectors exercised in Elixir. nx-ggml has no arbitrary-axis slicing primitive, so instead of
`rotate_half(x) = concat(-x2, x1)` (slice the vector in half, negate the second half, swap the two
halves), the Elixir side builds a fixed `(head_dim × head_dim)` constant matrix `M` and computes
`rotate_half(x) = x @ M`. The theorem `Rope.rotate_half_matmul_correct` proves this holds for every
`half` and every input vector `x : Nat → Int` (an infinite family, not a finite test set) by a plain
induction on the dot-product recursion — `#print axioms` confirms it rests only on `propext` and
`Quot.sound` (Lean's standard core axioms), with no `sorry`.

## `GallocrProof.lean`

A different kind of proof: not an algebraic identity in the op lowering, but a safety property of
`ggml_gallocr` (ggml's graph memory allocator, `native/nx_ggml/third_party/ggml/src/ggml-alloc.c`),
written while bisecting a real regression — switching `CompiledGraph`'s allocator from
`ggml_backend_alloc_ctx_tensors` (one permanent buffer slot per tensor) to `ggml_gallocr` (buffer
slots reused across non-overlapping tensors, needed because real-model graphs like TRELLIS.2's
SS-flow DiT have ~800MB attention-score intermediates that exhaust VRAM under the naive allocator)
made both DINOv3's 24-layer encoder and SS-flow's 30-block DiT produce wrong output. The allocator
switch was reverted for correctness; this proof exists to narrow down *why* gallocr broke instead of
guessing blindly at the ~1200-line C allocator.

It models three specific mechanisms in `ggml-alloc.c`, in increasing order of how much is taken on
faith rather than derived:

1. **`Gallocr.no_premature_free`** — the base reference-counting free-list rule at
   `ggml-alloc.c:731-820` (count every `src` reference once per node in one full pass, then free a
   tensor the instant its running per-node decrement hits that total). Proven safe from the counting
   arithmetic alone — no topological-order assumption needed, a fact discovered while writing the
   proof.
2. **`Gallocr.inplace_trigger_is_last_consumer`** — the inplace-reuse heuristic's trigger condition
   at `ggml-alloc.c:656` (`p_hn->n_children == 1`), which decides when it's safe to alias a parent's
   buffer directly onto a node's output. `ggml_op_can_inplace` allow-lists exactly the ops
   `graph_builder.cpp` uses pervasively (every elementwise add/sub/mul/div/sqrt/unary, RoPE,
   RMS_NORM, softmax) — every residual add and every LayerNorm subtraction in every real-model
   script is an inplace candidate. Proven safe by the same counting argument as (1).
3. **`Gallocr.view_trigger_extends_to_base`** — the view-reuse branch at `ggml-alloc.c:657-669`,
   exercised by every `add_transpose` (`ggml_permute` + `ggml_cont`, used for every Q/K/V head
   transpose in every real-model script). This one is proven **conditional on** the C code's own
   `n_views`/`n_children` bookkeeping for the view and its base tensor being accurate — it does not
   independently re-derive that bookkeeping from a lower-level model of `ggml_permute`'s
   view-creation logic (`ggml.c`/`ggml-impl.c`, a different and larger part of the codebase). That's
   flagged explicitly as the theorem's honest boundary.

Since (1) and (2) are both proven safe outright, and both of their real hypotheses (topological node
order from `ggml_build_forward_expand`; an accurate single-pass reference count) genuinely hold for
every graph `graph_builder.cpp` builds, **the regression is not in either of those mechanisms**. That
narrows the C++ investigation to (3)'s unverified premise: either a real bug in how
`ggml_permute`-produced views get their `n_views`/`n_children` counted, or — outside anything a
graph-liveness proof can reach — one of the allow-listed ops' CPU/Vulkan kernel implementations not
actually tolerating the aliased input/output pointers `ggml_op_can_inplace` promises it does.
