/-
Lean 4 correctness proof for the RoPE "rotate_half via constant matmul" trick
used in `scratch_dino_attention.exs` / `scratch_dino_layer0.exs` /
`scratch_dino_full.exs` (see `lib/nx_ggml/expr_lowering.ex` and the project
plan's fifth follow-up for the Elixir side).

nx-ggml has no slicing primitive, so RoPE's usual `rotate_half(x) =
concat(-x2, x1)` (split a length-`2*half` vector in half, negate the second
half, swap the two halves) was instead reformulated as a *linear map*: a
fixed `(2*half) x (2*half)` constant matrix `M` such that `rotate_half(x) =
x @ M` (row-vector-times-matrix), built once in Elixir as a literal
`Nx.tensor` and passed into the traced graph as an ordinary parameter. This
was validated empirically (matches `Nx.Defn.Evaluator` on synthetic cases,
and the real DINOv3 attention output matches the real PyTorch reference to
9.5e-6 max abs diff). This file proves it holds *exactly*, for every `half`
and every input vector `x`, as an algebraic identity -- not just checked on
the specific test vectors used in the Elixir/Nx test suite.

No external dependencies (no Mathlib) -- everything here is built from
Lean 4 core plus the `omega` tactic that ships with the toolchain, so this
compiles standalone with a bare `lean` invocation:

    lean RopeProof.lean
-/

namespace Rope

/-- The fixed rotate-half matrix, matching the construction in
`scratch_dino_attention.exs`/`scratch_dino_layer0.exs`/`scratch_dino_full.exs`:
```
cond do
  j < half and i == half + j -> -1.0
  j >= half and i == j - half -> 1.0
  true -> 0.0
end
```
-/
def M (half : Nat) (i j : Nat) : Int :=
  if j < half then
    (if i = half + j then (-1 : Int) else 0)
  else
    (if i = j - half then (1 : Int) else 0)

/-- The directly-intended operation: `rotate_half(x) = concat(-x2, x1)` for
`x = concat(x1, x2)`, each half of length `half`. -/
def rotateHalf (half : Nat) (x : Nat → Int) (j : Nat) : Int :=
  if j < half then - x (half + j) else x (j - half)

/-- Dot product of `x` and `y` over indices `0 ..< n`, built by plain
recursion on `n` so induction proofs don't need any list/Finset machinery. -/
def dotUpTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => dotUpTo f n + f n

/-- If `f` vanishes at every index below `n` except possibly `k`, then the
partial sum collapses to `f k` (or `0`, if `k` is out of range). This is the
one nontrivial lemma the whole proof rests on; everything else is
definitional unfolding + case splits. -/
theorem dotUpTo_single (f : Nat → Int) (k : Nat) :
    ∀ n, (∀ i, i < n → i ≠ k → f i = 0) → dotUpTo f n = if k < n then f k else 0 := by
  intro n
  induction n with
  | zero => intro _; simp [dotUpTo]
  | succ n ih =>
    intro hz
    have hz' : ∀ i, i < n → i ≠ k → f i = 0 := fun i hi hne => hz i (by omega) hne
    have hprev := ih hz'
    unfold dotUpTo
    by_cases hkn : k < n
    · have hfn : f n = 0 := hz n (by omega) (by omega)
      simp [hprev, hkn, hfn]
      omega
    · by_cases hkeq : k = n
      · subst hkeq
        simp [hprev, hkn]
      · have hfn : f n = 0 := hz n (by omega) (by omega)
        have hkn1 : ¬ (k < n + 1) := by omega
        simp [hprev, hkn, hfn, hkn1]

/-- The main theorem: for every `half`, every `n = half + half`, every input
vector `x`, and every output index `j < n`, the constant-matmul formulation
computes exactly `rotate_half`. This holds for *every* `x : Nat → Int`, not
just the finitely many test vectors exercised by the Elixir property tests
-- the algebraic argument covers the whole (infinite) input space at once. -/
theorem rotate_half_matmul_correct (half : Nat) (x : Nat → Int) (j : Nat)
    (hj : j < half + half) :
    dotUpTo (fun i => x i * M half i j) (half + half) = rotateHalf half x j := by
  by_cases hlt : j < half
  · -- k = half + j
    have hkn : half + j < half + half := by omega
    have hvanish : ∀ i, i < half + half → i ≠ half + j → x i * M half i j = 0 := by
      intro i _ hne
      have hm : M half i j = 0 := by unfold M; simp [hlt, hne]
      simp [hm]
    have hkey := dotUpTo_single (fun i => x i * M half i j) (half + j) (half + half) hvanish
    rw [hkey]
    simp only [hkn, if_true]
    unfold M rotateHalf
    simp [hlt]
  · -- k = j - half
    have hkn : j - half < half + half := by omega
    have hvanish : ∀ i, i < half + half → i ≠ j - half → x i * M half i j = 0 := by
      intro i _ hne
      have hm : M half i j = 0 := by unfold M; simp [hlt, hne]
      simp [hm]
    have hkey := dotUpTo_single (fun i => x i * M half i j) (j - half) (half + half) hvanish
    rw [hkey]
    simp only [hkn, if_true]
    unfold M rotateHalf
    simp [hlt]

end Rope

-- Sanity check: confirm the abstract theorem's statement actually applies
-- to the concrete `head_dim = 64` (`half = 32`) used in the real DINOv3
-- ViT-L/16 port, and that the axiom check is clean (no `sorry`).
#print axioms Rope.rotate_half_matmul_correct

example (x : Nat → Int) (j : Nat) (hj : j < 64) :
    Rope.dotUpTo (fun i => x i * Rope.M 32 i j) 64 = Rope.rotateHalf 32 x j :=
  Rope.rotate_half_matmul_correct 32 x j hj
