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
