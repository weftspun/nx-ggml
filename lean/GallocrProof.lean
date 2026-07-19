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
allocator, multi-buffer support). It models three specific mechanisms, in
increasing order of how much they're taken on faith rather than derived:

1. `no_premature_free` -- the base reference-counting free-list rule at
   `ggml-alloc.c:731-820` (count every `src` reference once per node in a
   single full pass, then free a tensor the instant its running per-node
   decrement count hits that total).
2. `inplace_trigger_is_last_consumer` -- the inplace-reuse heuristic's
   trigger condition at `ggml-alloc.c:656` (`p_hn->n_children == 1`), which
   decides when it's safe to alias a *parent's* buffer directly onto a
   node's output instead of allocating fresh memory. `ggml_op_can_inplace`
   (line 22) allow-lists exactly the ops `graph_builder.cpp` uses
   pervasively for elementwise arithmetic (ADD, SUB, MUL, DIV, SQRT, LOG,
   UNARY (covers sigmoid/tanh/exp/etc.), ROPE, RMS_NORM, SOFT_MAX) --
   every residual add, every LayerNorm subtraction, every RoPE combine in
   every real-model script is an inplace candidate. Matmul (`GGML_OP_MUL_MAT`)
   is not on the list, so `add_matmul`'s outputs are never inplace targets.
3. `view_trigger_extends_to_base` -- the view-reuse branch at
   `ggml-alloc.c:657-669`, which lets a consumer take over a *view's*
   underlying base tensor's memory directly (bypassing the normal
   free-then-realloc path), exercised by every `add_transpose`
   (`ggml_permute` + `ggml_cont`) call -- used for every Q/K/V head
   transpose and every attention-weight transpose in every real-model
   script. This one is NOT independently derived the way (1) and (2) are:
   it's proven correct **conditional on** the C code's own view-bookkeeping
   guard (`n_views`/`n_children` on both the view and its base) being
   accurate, which requires modeling `ggml_permute`'s view-creation
   bookkeeping in `ggml.c`/`ggml-impl.c` -- a different, larger piece of
   the codebase this file does not model. That unverified premise is
   flagged explicitly at the theorem.

Result: (1) and (2) are both proven safe from the reference-counting
arithmetic *alone* -- no topological-order assumption is needed for
either, a fact discovered while writing the proof (`WellFormed` ends up
unused below; it's kept only to document the separate, more basic
"computed before read" property that topological order actually provides,
which this file doesn't restate as a theorem). Ruling out both the base
mechanism and the inplace-trigger timing narrows the regression to (3)'s
unverified premise: either ggml's own `n_views` bookkeeping has a real bug
under this exact usage pattern, or -- outside anything a graph-liveness
proof can reach -- one of the allow-listed ops' CPU/Vulkan kernel
implementations doesn't actually tolerate the aliased input/output pointers
`ggml_op_can_inplace` promises it does. Both are now the concrete next
things to check in a C++ repro, not further Lean modeling.

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

/-- **Core combinatorial lemma.** Once the running consumer count for `k`
reaches its final total at some position `j+1`, it stays pinned at that
total for every later position up to `n` -- so no node strictly after `j`
can be a fresh consumer of `k` (a fresh consumer would have to strictly
increase the count past a value that's already at its ceiling). Both
`no_premature_free` and `inplace_trigger_is_last_consumer` below are
one-line corollaries of this: they differ only in *how* they establish the
"count already reached its total at j+1" hypothesis. -/
theorem no_consumer_after_saturation (g : Graph) (n k j : Nat)
    (_hj : j < n) (hsat : childCount g (j + 1) k = childCount g n k) :
    ∀ m, j < m → m < n → g m k = false := by
  intro m hjm hmn
  have hsqueeze : ∀ p, j + 1 ≤ p → p ≤ n → childCount g p k = childCount g n k := by
    intro p hp1 hp2
    have hlow : childCount g (j + 1) k ≤ childCount g p k := childCount_mono g k _ _ hp1
    have hhigh : childCount g p k ≤ childCount g n k := childCount_mono g k _ _ hp2
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

/-- gallocr frees `k` at position `j` (`j < n`) iff processing nodes
`0 ..< j+1` has driven the running decrement count up to `k`'s TOTAL
reference count over the whole `n`-node graph -- i.e. the first index at
which the simulated `n_children` counter would hit zero
(`ggml-alloc.c:798-804`, `p_hn->n_children -= 1; if (p_hn->n_children == 0)
... free`). -/
def FreesAt (g : Graph) (n k j : Nat) : Prop :=
  childCount g (j + 1) k = childCount g n k ∧
  ∀ j', j' < j → childCount g (j' + 1) k < childCount g n k

/-- **Safety theorem 1: the base free-list rule.** If `k` frees at position
`j` (under gallocr's reference-counting rule) and the graph is well-formed
(topological order), then no node strictly after `j` reads `k` as a source
-- gallocr's freeing point is never earlier than `k`'s true last consumer,
so reusing `k`'s memory for anything allocated after `j` can never silently
overwrite a value some later op still needs to read. -/
theorem no_premature_free (g : Graph) (n k j m : Nat)
    (_hwf : WellFormed g n) (hj : j < n) (hfree : FreesAt g n k j)
    (hm : m < n) (hjm : j < m) : g m k = false :=
  no_consumer_after_saturation g n k j hj hfree.1 m hjm hm

/-- gallocr's live `n_children` value for `k` at the moment node `m` is
about to be allocated -- BEFORE `m`'s own consumption of `k` (if any) gets
decremented, matching `ggml_gallocr_allocate_node`'s inplace check
(`ggml-alloc.c:622-680`) running *before* the "update parents" decrement
step that comes later in the same pass (`ggml-alloc.c:792-799`). -/
def liveChildrenAt (g : Graph) (n k m : Nat) : Nat :=
  childCount g n k - childCount g m k

/-- **Safety theorem 2: the inplace-reuse trigger.** If, at the moment node
`m` is allocated, `k`'s live remaining-consumer count is exactly `1` (the
`p_hn->n_children == 1` check gating the inplace-reuse branch,
`ggml-alloc.c:656`), and `m` itself reads `k` as a source, then `m` is `k`'s
unique remaining consumer: no other node at or after `m` also reads `k`.
This is exactly the property inplace reuse needs -- when the C code decides
to alias `k`'s buffer directly onto `m`'s output, `k` genuinely has no
other future reader, so overwriting it in place cannot corrupt a value
anything else still needs. Combined with `no_premature_free`, this rules
out BOTH the base free-list timing and the inplace-trigger timing as the
regression's source: every op `graph_builder.cpp` builds that's eligible
for inplace reuse (every elementwise add/sub/mul/div/sqrt/unary, RoPE,
RMS_NORM, softmax -- see `ggml_op_can_inplace`) only ever gets aliased onto
a parent that has truly reached the end of its lifetime. -/
theorem inplace_trigger_is_last_consumer (g : Graph) (n k m : Nat)
    (hm : m < n) (hmk : g m k = true) (htrigger : liveChildrenAt g n k m = 1) :
    ∀ m', m < m' → m' < n → g m' k = false := by
  have hsat : childCount g (m + 1) k = childCount g n k := by
    have hle : childCount g m k ≤ childCount g n k := childCount_mono g k m n (by omega)
    unfold liveChildrenAt at htrigger
    rw [childCount_succ, hmk]
    simp only [if_true]
    omega
  exact no_consumer_after_saturation g n k m hm hsat

/-! ### The view-reuse branch (`ggml-alloc.c:657-669`), and its unverified premise

`add_transpose` (`graph_builder.cpp`) lowers to `ggml_permute` (a VIEW of
its input -- no new memory, just reinterpreted strides) followed by
`ggml_cont` (a fresh, contiguous copy). When some downstream node `m` reads
that `ggml_cont` output `p` and triggers inplace reuse (theorem 2 above,
applied to `p`), the C code has an extra branch (`ggml-alloc.c:657-669`)
for when `p` is *itself* a view of some base tensor `v` (`view_src`): if
`v`'s own bookkeeping shows `n_views == 1` (only `p` views it) and
`n_children == 0` (no direct, non-view readers of `v` remain), the code
lets `m` take over `v`'s memory address directly, marking both `p` and `v`
as `allocated = false` without going through the normal free step.

This section models that as: `m` effectively becomes a consumer of `v`
itself, one level removed through the view `p`. The theorem below shows
that transfer is safe *conditional on* the same two counts the C code
itself checks being accurate -- it does NOT re-derive `n_views`/`n_children`
for `v` and `p` from a lower-level model of `ggml_permute`'s view-creation
bookkeeping (that lives in `ggml.c`/`ggml-impl.c`, a different and larger
part of the codebase). That gap is the honest boundary of this file: if
there's a real bug in how `ggml_permute`-produced views get their
`n_views`/`n_children` counted, it's invisible here by construction. -/

/-- The C code's exact view-transfer precondition
(`ggml-alloc.c:656,660`): `p` (the view, e.g. a `ggml_cont` output whose
*input* was a `ggml_permute` view -- but the transfer logic only cares that
`p.view_src = some v`) has exactly one remaining child (`m`) and no other
views; `v`, the ultimate base tensor, has exactly one view (`p`) and no
direct children of its own. -/
structure ViewTransferGuard (g : Graph) (n k_p k_v m : Nat) : Prop where
  p_solely_consumed_by_m : liveChildrenAt g n k_p m = 1
  p_reads_v : g m k_p = true
  v_no_direct_children : childCount g n k_v = 0

/-- **Safety theorem 3 (conditional): the view-reuse trigger.** Given the
exact guard the C code checks (`ViewTransferGuard`), `m` taking over `v`'s
memory in place of going through `p` is safe by the same
last-consumer argument as theorem 2, *provided* `v`'s reported
`childCount` genuinely reflects zero real readers -- which is
`v_no_direct_children`, taken as a hypothesis here rather than derived.
Since `v` has no direct children at all (not just "one remaining"), no
node anywhere in the graph reads `v` directly, so -- trivially, and without
needing `no_consumer_after_saturation` -- no node after `m` (or anywhere
else) can read `v` as a direct source either. The real question this
theorem can't answer is whether `v_no_direct_children` (i.e. `v`'s
`n_children`, as bookkept elsewhere in `ggml-alloc.c`/`ggml-impl.c` for
view-producing ops) is actually accurate for `graph_builder.cpp`'s
`ggml_permute`+`ggml_cont` pattern -- that's exactly the unverified premise
flagged in the section comment above. -/
theorem view_trigger_extends_to_base (g : Graph) (n k_p k_v m : Nat)
    (hguard : ViewTransferGuard g n k_p k_v m) :
    ∀ m', m' < n → g m' k_v = false := by
  intro m' hm'
  have htotal0 : childCount g n k_v = 0 := hguard.v_no_direct_children
  have hle : childCount g (m' + 1) k_v ≤ childCount g n k_v :=
    childCount_mono g k_v (m' + 1) n (by omega)
  rw [childCount_succ] at hle
  cases hgm'kv : g m' k_v with
  | false => rfl
  | true =>
    rw [hgm'kv] at hle
    simp only [if_true] at hle
    omega

end Gallocr

-- Sanity check: no `sorry`, no extra axioms beyond Lean's own kernel/`omega`.
#print axioms Gallocr.no_premature_free
#print axioms Gallocr.inplace_trigger_is_last_consumer
#print axioms Gallocr.view_trigger_extends_to_base
