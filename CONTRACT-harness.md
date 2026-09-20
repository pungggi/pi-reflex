# CONTRACT-harness — the pi-continual-harness consumer contract

**Status:** pinned 2026-09-20 by the first real consumer ([pi-continual-harness](https://github.com/pungggi/pi-continual-harness) ≥ 0.10.0).
**Purpose:** pi-continual-harness is pi-reflex's (formerly pi-jev) first integration target. This
document pins the decision shapes, abstain semantics, latency budgets, and data
contract **before** A3 builds the extension surfaces — so integration is glue,
not redesign. Changes to this contract go through a PR touching BOTH repos.

The harness side of the seam already shipped (unreleased 0.10.0,
`feat/similarity-seam`): `DedupeOptions.similarity` accepts
`(a, b) => number | { score, abstain }`.

## 1. Decision call-sites

| # | Harness call-site | pi-reflex primitive | Inputs | On abstain |
|---|---|---|---|---|
| D1 | **Dedupe**: are two items the same durable fact? | `noul("same durable fact?")` | item contents (+ kind, evidence context) | **keep both**, flag pair for agent review — never auto-merge/delete |
| D2 | **Create-gate**: is a proposed delta durable & reusable? | `noul("durable and reusable?")` | delta content + evidence | **escalate to agent** (existing steering path) — never silently accept/reject |
| D3 | **Importance re-score**: how fit is this stored item? | `score` (ordinal rubric) + `ScoreConformal.interval()` | item content + trajectory evidence | skip the item this pass; importance unchanged |
| D4 | **Injection relevance**: does this item matter THIS turn? | `score` | item content vs current task/user message | item falls back to importance-order ranking |

Rule zero (applies to every call-site): **abstain is conservative.** The
harness never destroys or mutates user state on an uncertain decision.
Abstention is derived from the conformal layer: singleton prediction set →
act; anything larger → abstain. That is exactly the mapping from
`NoulConformal.set()` → `{ score, abstain }` on the similarity seam.

## 2. The similarity seam (shipped, must stay stable)

```ts
// pi-continual-harness — src/proposer.ts (re-exported from the package entry)
export interface SimilarityResult {
  score: number;    // similarity in [0,1], compared against dedupe.threshold
  abstain?: boolean // true = keep both; never merge or delete
}
type Similarity = (a: string, b: string) => number | SimilarityResult;
```

A pi-reflex companion package registers itself as the dedupe `similarity` (and/or
a `jev` proposer via `registerProposer`). Plain numeric returns must keep
working — the companion is a drop-in upgrade over token Jaccard, and the
harness falls back to Jaccard whenever pi-reflex is absent, offline, or over
budget (**soft-fail composition**, the pi-mem pattern).

## 3. Latency budgets (per call-site)

| Call-site | Fires | Budget | Notes |
|---|---|---|---|
| D4 injection relevance | `session_prompt` (turn start) | **≤ 100 ms total** else skip to importance-order | runs per selected item per turn — the hot path |
| D1 dedupe | `/refine`, turn_end auto-refine | ≤ 250 ms per pair-batch | can cache per (item-id, item-id) — contents are immutable between mutations |
| D2 create-gate | `harness_mutate` intercept | ≤ 250 ms per delta batch | |
| D3 importance re-score | offline / manual command | unbounded | batch job, no turn in flight |

> Measured 2026-09-20 (int8 CPU, Zen3, ORT 1.30): see [BENCHMARKS.md](BENCHMARKS.md).
> D4 viable on CPU only with the multilingual checkpoint and ≤3 items per turn;
> english 421M never fits D4 — use it for D1–D3 with caching, or GPU.

Honesty check vs pi-reflex's own numbers (ARCHITECTURE.md): CPU int8 is
~40–120 ms (322 M multilingual) / ~80–250 ms (421 M English) per call. That
makes D2/D1 viable on CPU **only with result caching**, D4 marginal (batch the
selected items into ONE engine call, not N), D3 free. If the budget is missed:
skip, don't block the turn.

## 4. Data contract — the calibration corpus

The harness exports labeled decision data pi-reflex uses to fit/verify its
conformal layers (split: calibration vs eval). Format: JSONL, one record per
decision, schema versioned with this contract.

```jsonc
// dedupe-pairs.jsonl (from planDedupe runs)
{ "v": 1, "kind": "dedupe_pair", "a": "<content>", "b": "<content>",
  "label": "dup" | "not_dup",            // merged = dup; same-key-field pairs in [0.3, 0.6) overlap the planner did NOT merge = not_dup
  "similarity": 0.71, "source": "tokenOverlap", "needs_review": true,  // set on not_dup only
  "meta": { "kinds": ["prompt"], "owners": ["anthropic/claude-..."] } }

// lifecycle.jsonl (from outcome-loop + prune/keep/drop events)
{ "v": 1, "kind": "lifecycle", "item_kind": "memory", "content_hash": "sha256:...",
  "event": "created" | "cited" | "pruned" | "dropped" | "kept" | "deleted", "importance": 0.42 }
```

Shipped in pi-continual-harness 0.11.0 as `/harness export-corpus [path]`
(pure `buildCorpus()` core, re-exported from the package entry for offline use).
One record per unique pair across runs; lifecycle contents are hashed, pair
contents are carried (training needs them).

Current size (honest): ~23 sessions with harness entries — enough for schema +
smoke evals, **not** headline numbers. The exporter doubles as ongoing
collection: corpus grows as the harness runs. Target use: A4 `BENCHMARKS.md`
conformal coverage plots on real agent-memory decisions — the differentiator
neither laya nor von can copy.

## 5. Integration invariants (non-negotiable)

1. **Audit trail** — every pi-reflex-gated mutation flows through the harness's
   `applyDeltas` + `harness-state` session entries. `/tree` rollback must cover
   engine-triggered deletes. No side-channel writes.
2. **Soft-fail** — pi-reflex absent/slow/erroring ⇒ harness behavior degrades to
   token Jaccard + the plain steering proposer. Never broken.
3. **Per-model isolation** — harness stores are strictly per-model
   (`ownerModel`). The A3 model-router preset must know: routing a turn to a
   different tier changes which items inject. Feature, not bug.
4. **Determinism** — same input ⇒ same prediction set. Golden tests required
   (fixed question set, fixed checkpoints, snapshot the sets) so conformal
   guarantees aren't undermined by runtime drift.
5. **No PII exfiltration** — corpus export is local-only files; nothing leaves
   the machine unless the user pushes it.

## 6. Versioning

This contract is semver-ish: additive changes bump the minor (new call-sites,
new optional fields); anything the harness-side seam depends on
(`SimilarityResult` shape, abstain semantics, budgets getting *tighter*)
requires a coordinated change PR in both repos. The `v` field in corpus
records follows the same major.minor.
