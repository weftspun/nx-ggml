/-
Lean 4 safety proof for the core reference-counting liveness mechanism used
by ggml's graph allocator (`ggml_gallocr`, `native/nx_ggml/third_party/ggml/
src/ggml-alloc.c`), written while bisecting a real correctness regression:
switching `CompiledGraph`'s allocator from `ggml_backend_alloc_ctx_tensors`
(one permanent buffer slot per tensor) to `ggml_gallocr` (buffer slots
reused/shared across non-overlapping tensors, needed because real-model
graphs like TRELLIS.2's SS-flow DiT block have ~800MB attention-score
intermediates that exhaust VRAM under the naive allocator) made both
DINOv3's 24-layer encoder and SS-flow's 30-block DiT produce wrong output
(NaN / ~1000x-off values) -- see the commit reverting that switch.

This does NOT model all ~1200 lines of `ggml-alloc.c` (the best-fit block
allocator, multi-buffer support, the inplace-reuse heuristic, view/n_views
aliasing). It models exactly the mechanism at lines 731-820: gallocr's
"count every src reference once per node in a single full pass, then free a
tensor the instant its running per-node decrement count hits that total" --
the textbook reference-counting free-list liveness rule, and the one piece
of the allocator whose *timing* (does it ever free memory a still-live
tensor needs?) determines whether reused memory can ever silently corrupt a
value still in use downstream.

Result: the free-timing half of the mechanism is proven safe -- gallocr can
never free a tensor before its true last consumer has executed -- and,
found while writing the proof, this holds from the reference-counting
arithmetic *alone* (`FreesAt` + an accurate `childCount` pre-pass): no
topological-order assumption is needed for *this* half, so `WellFormed`
ends up unused by `no_premature_free` below. Topological order (every
source strictly precedes its consumer -- exactly what
`ggml_build_forward_expand`'s post-order DFS from the single output
guarantees) is still essential, but for the *other*, more basic half this
file doesn't restate as a theorem: it's what makes "process nodes in index
order" also mean "every source is already computed before its first read."
`WellFormed` is kept as a hypothesis/definition here to document that half
even though the Lean proof below doesn't need to invoke it.

The accurate-pre-pass hypothesis (a single full scan summing every node's
every src slot) matches `ggml-alloc.c`'s own first loop exactly, and both
halves hold for every graph `graph_builder.cpp` builds. So the regression
is NOT in this core mechanism -- it must be in the parts left unmodeled
here: the inplace-reuse heuristic (reusing a *parent's* buffer directly for
a node computed in place) or the view/`n_views` special-casing for
`ggml_permute`/`ggml_reshape`-produced views, both exercised heavily by
`graph_builder.cpp`'s `add_transpose` (permute+cont) and the residual-add
pattern (`x = x + ...`) used throughout every real-model script. That
narrows where the C++ debugging session should look next.

No external dependencies (no Mathlib), compiles standalone:

    lean GallocrProof.lean
-/

namespace Gallocr

/-- A graph of `n` nodes, indices `0 ..< n`. `g i s = true` means "node `i`
reads node `s` as a source (operand)" -- one Boolean edge predicate per
(consumer, source) pair, matching `ggml_tensor.src[]` but without needing
`List`/array-indexing lemmas (same `Nat → _` function style as
`RopeProof.lean`'s `x : Nat → Int`). -/
abbrev Graph := Nat → Nat → Bool

/-- Topological order: every source a node reads has a strictly smaller
index than the node itself. This is exactly what `ggml_build_forward_expand`
guarantees (`graph_builder.cpp`'s `CompiledGraph` constructor calls it right
before allocation) -- a post-order DFS from the single `output_` node can
only emit a node after all of its sources are already emitted. -/
def WellFormed (g : Graph) (n : Nat) : Prop :=
  ∀ i, i < n → ∀ s, g i s = true → s < i

/-- How many times index `k` is read as a source among nodes `0 ..< n` --
gallocr's own first pass (`ggml-alloc.c:747-753`: for every node, for every
src slot, `hash[src].n_children += 1`), expressed as a running count via the
same `dotUpTo`-style recursion `RopeProof.lean` uses for sums. -/
def childCount (g : Graph) (n k : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => childCount g n k + (if g n k then 1 else 0)

/-- One-step unfolding, spelled out for readability at call sites. -/
theorem childCount_succ (g : Graph) (n k : Nat) :
    childCount g (n + 1) k = childCount g n k + (if g n k then 1 else 0) := rfl

/-- `childCount` is monotone nondecreasing as more of the graph is scanned:
adding node `n` to the window can only add a reference to `k`, never remove
one. -/
theorem childCount_mono (g : Graph) (k : Nat) :
    ∀ n1 n2, n1 ≤ n2 → childCount g n1 k ≤ childCount g n2 k := by
  intro n1 n2 h
  induction n2 with
  | zero =>
    have hz : n1 = 0 := by omega
    subst hz
    omega
  | succ n2 ih =>
    rcases Nat.lt_or_ge n1 (n2 + 1) with h' | h'
    · have hn1n2 : n1 ≤ n2 := by omega
      have hstep := ih hn1n2
      rw [childCount_succ]
      split <;> omega
    · have heq : n1 = n2 + 1 := by omega
      subst heq
      omega

/-- gallocr frees `k` at position `j` (`j < n`) iff processing nodes
`0 ..< j+1` has driven the running decrement count up to `k`'s TOTAL
reference count over the whole `n`-node graph -- i.e. the first index at
which the simulated `n_children` counter would hit zero
(`ggml-alloc.c:798-804`, `p_hn->n_children -= 1; if (p_hn->n_children == 0)
... free`). -/
def FreesAt (g : Graph) (n k j : Nat) : Prop :=
  childCount g (j + 1) k = childCount g n k ∧
  ∀ j', j' < j → childCount g (j' + 1) k < childCount g n k

/-- **Main safety theorem.** If `k` frees at position `j` (under gallocr's
reference-counting rule) and the graph is well-formed (topological order),
then no node strictly after `j` reads `k` as a source -- gallocr's freeing
point is never earlier than `k`'s true last consumer, so reusing `k`'s
memory for anything allocated after `j` can never silently overwrite a
value some later op still needs to read. This is the exact safety property
whose violation would explain the observed corruption (NaN / ~1000x-off
real-model output): a tensor's buffer got reused for something else while a
downstream op still needed to read the old value. -/
theorem no_premature_free (g : Graph) (n k j m : Nat)
    (_hwf : WellFormed g n) (_hj : j < n) (hfree : FreesAt g n k j)
    (hm : m < n) (hjm : j < m) : g m k = false := by
  -- childCount is pinned at the total for every index in [j+1, n], since it
  -- is squeezed between the value already reached at j+1 and the value at
  -- n (both equal the total by `hfree.1` and reflexivity/monotonicity).
  have hsqueeze : ∀ p, j + 1 ≤ p → p ≤ n → childCount g p k = childCount g n k := by
    intro p hp1 hp2
    have hlow : childCount g (j + 1) k ≤ childCount g p k := childCount_mono g k _ _ hp1
    have hhigh : childCount g p k ≤ childCount g n k := childCount_mono g k _ _ hp2
    have heqtotal : childCount g (j + 1) k = childCount g n k := hfree.1
    omega
  have hm1 : childCount g m k = childCount g n k := hsqueeze m (by omega) (by omega)
  have hm2 : childCount g (m + 1) k = childCount g n k := hsqueeze (m + 1) (by omega) (by omega)
  rw [childCount_succ] at hm2
  cases hgmk : g m k with
  | false => rfl
  | true =>
    rw [hgmk] at hm2
    simp only [if_true] at hm2
    omega

end Gallocr

-- Sanity check: no `sorry`, no extra axioms beyond Lean's own kernel/`omega`.
#print axioms Gallocr.no_premature_free
