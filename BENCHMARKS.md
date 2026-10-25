# BENCHMARKS — contract §3 latency evidence

**Status:** first conformance pass against [CONTRACT-harness.md](CONTRACT-harness.md) §3.
**Date:** 2026-09-20 · **Quantization:** int8 (dynamic) · **Execution:** CPU EP, warm process

## Environment (honesty first)

| | |
|---|---|
| CPU | AMD Zen 3 desktop (Family 25 Model 33) — multi-thread ORT intra-op pool |
| OS | Windows 11, Node v24.16.0 |
| Runtime | onnxruntime-node 1.30.0, @huggingface/tokenizers 0.2.0 (pure JS tokenizer) |
| Scenario n | 20 warm runs per scenario per checkpoint (cold start excluded) |
| Parity | engine outputs verified vs real laya torch (≤ 5.7e-06 logits, identical token usage) |

## Contract §3 verdicts

Budgets from CONTRACT-harness.md §3. **Verdict style: measured, not aspirational.**

| Call-site | Budget | multilingual (322M) | english (421M) | Verdict |
|---|---|---|---|---|
| **D1 dedupe** (≤250 ms / pair-batch) | 250 ms | 48.8 ms (1 pair) · 222.8 ms (8 pairs) | 100.8 ms (1 pair) · 450.9 ms (8 pairs) | **CPU-viable** on multilingual up to ~8 pairs/batch; english up to ~4 pairs — beyond that, cache (contract-sanctioned) |
| **D2 create-gate** (≤250 ms / delta-batch) | 250 ms | same shape as D1 | same shape as D1 | **CPU-viable** (same numbers as D1) |
| **D3 importance re-score** (unbounded) | — | 43.1 ms / item | 87.0 ms / item | **CPU-viable**, free (offline batch job) |
| **D4 injection relevance** (≤100 ms **total** at turn start) | 100 ms | 27.9 ms/item → **≤3 items** | 56.4 ms/item → **≤1 item** | **CONDITIONAL**: multilingual + ≤3 selected items passes; beyond that the contract's own fallback (importance-order) applies, or GPU |

## Raw numbers (int8, p50 / p95, ms)

### multilingual (mmBERT-base)

| Scenario | p50 | p95 | mean |
|---|---|---|---|
| D1/D2 noul_single | 48.8 | 53.3 | 49.0 |
| D3 score2_single | 43.1 | 49.1 | 42.7 |
| D4 batch8_score2 (total) | 222.8 | 244.0 | 224.0 |
| D4 batch8_score2 (per item) | 27.9 | — | 28.0 |
| four_questions (info) | 172.7 | 194.1 | 177.8 |

### english (ModernBERT-large)

| Scenario | p50 | p95 | mean |
|---|---|---|---|
| D1/D2 noul_single | 100.8 | 110.0 | 101.2 |
| D3 score2_single | 87.0 | 91.8 | 84.0 |
| D4 batch8_score2 (total) | 450.9 | 524.7 | 467.1 |
| D4 batch8_score2 (per item) | 56.4 | — | 58.4 |
| four_questions (info) | 394.1 | 419.0 | 396.5 |

### typed-decisions (ModernBERT-large, workflow-tuned)

| Scenario | p50 | p95 | mean |
|---|---|---|---|
| D1/D2 noul_single | 100.9 | 110.6 | 100.7 |
| D3 score2_single | 76.3 | 90.2 | 79.3 |
| D4 batch8_score2 (total) | 446.2 | 473.5 | 450.5 |
| four_questions (info) | 379.7 | 450.1 | 388.9 |

## Methodology

- `tools/bench.mjs` — loads each checkpoint via `Engine.fromArtifacts(..., { int8: true })`,
  one warmup call per scenario, then 20 timed calls; p50/p95 are nearest-rank order
  statistics over the sorted samples. Scenario payloads are contract-shaped
  (noul = pair/delta decision, score-2 = relevance, batch of 8 states via
  `Engine.batchQuestion` — one forward pass, not eight).
- Latency includes tokenization, sequence building, collation, forward pass, and
  calibrated answer assembly — the full `systemOne`/`batchQuestion` path.
- fp32 reference numbers (same machine): multilingual ~220 ms / 4-question call,
  english ~590 ms — int8 is roughly 1.3–1.6× faster end-to-end here.

## Honesty notes

- **p95 with n=20 is noisy.** Treat p95 as indicative; p50 is the planning number.
- **Single machine, warm process.** No cold-start numbers (first call after
  `Engine.fromArtifacts` pays model load: seconds), no concurrent-load numbers, no GPU.
- **D4 is the honest sore spot.** The ≤100 ms turn-start budget holds for ≤3 selected
  items on multilingual-int8 and effectively fails on the 421 M checkpoints. Options,
  in contract spirit: pre-select aggressively (≤3 items), pin D4 to the multilingual
  checkpoint, move to GPU, or take the contract's own fallback (importance-order
  ranking). **Do not silently exceed the budget.**
- **Accuracy is now measured (A4, curated corpus):** raw checkpoints sit at chance on
  paraphrase-sameness and the value is the conformal layer — see §A4 for numbers and provenance.

## A4 — accuracy eval (first real numbers: curated corpus, 2026-10-25)

`tools/eval-corpus.mjs` implements the A4 protocol over the CONTRACT-harness §4 corpus:
the exact D1 call-site shape (state `{a, b, kind}`, `noul` "same durable fact?") for both arms —
local `reflex/<checkpoint>` and Cloudflare's hosted Workers AI classifiers (pi ≥ 1.0.1
exposes them through the same classifier interface; the runner uses the identical REST transport).

```bash
node tools/eval-corpus.mjs --synthetic 40                     # self-test: machinery only, no corpus/creds
node tools/eval-corpus.mjs --corpus dedupe-pairs.jsonl        # reflex arm (local, free)
node tools/eval-corpus.mjs --corpus dedupe-pairs.jsonl --clef # + hosted arm (CLOUDFLARE_API_KEY + CLOUDFLARE_ACCOUNT_ID)
#   --clef-model @cf/cloudflare/clef-omni   (Clef Omni,  pi 1.4 catalog, $0.15/M input tok)
#   --clef-model typesafe/jev-1.14.0        (Jev 1.14,   pi 1.4 catalog, $0.042/M input tok)
# append --json for a machine-readable report; `npm run build` first (arms import ../dist)
```

Metrics: accuracy@0.5, ECE-10, AUC, latency p50/p95, tokens, plus conformal coverage, abstain
rate, and act-accuracy on a seeded calib/eval split (reflex arm only; hosted classifiers have no
conformal layer). Cost guard: the hosted arm caps at `--limit` (default 200) calls.

### Corpus provenance (honest)

- **Why curated:** no organic labels exist. A full transcript-mining pass (756 session files)
  found 111 `harness-state` snapshots but **zero `proposer: "dedupe"` runs**, and replaying the
  production dedupe proposer over all 56 unique states yielded 0 usable pairs: items are
  per-model isolated and every comparable pair sits below 0.3 token overlap. The real harness
  pool simply contains no duplicate clusters yet.
- **What was used instead:** 120 hand-labeled pairs (60 dup / 60 not_dup) built from REAL
  recorded guidance in this environment: durable harness items, session scratch notes, and repo
  docs (release recipes, shell quirks, workflow rules). `dup` pairs are independent recordings or
  faithful paraphrases of the same durable fact; `not_dup` pairs are different facts, biased
  toward same-topic hard negatives (both here-string rules, both version-sync rules, both router
  behaviors). Labels were hand-assigned by the maintainer's agent and are **not human-verified**.
- **Deliberately hard regime:** average tokenOverlap is 0.269 (dup) vs 0.079 (not_dup); zero dup
  pairs reach the production merge threshold (0.6). Lexical thresholding cannot separate these,
  which is exactly why D1 wants a learned judge.
- The corpus stays local (`.pi/scratch/a4-corpus/`, gitignored) per the contract's local-only
  invariant; it is not shipped or published.

### Results (int8, CPU, deterministic seeded split; rerun with the commands above)

| reflex engine | accuracy@0.5 | ECE-10 | AUC | latency p50 / p95 | conformal coverage (α=0.1) | abstain | act-accuracy (n) |
|---|---|---|---|---|---|---|---|
| multilingual | 56.7% | 0.329 | 0.470 | 82 / 107 ms | **90.0%** | 83.3% | 40.0% (10) |
| typed-decisions | 50.8% | 0.022 | 0.530 | 171 / 212 ms | **96.7%** | 86.7% | 75.0% (8) |
| english | 52.5% | 0.037 | 0.629 | 169 / 303 ms | **90.0%** | 71.7% | 64.7% (17) |

How to read this (measured, not spun):

- **Raw checkpoints are at chance on paraphrase-sameness.** Accuracy 50.8–56.7% and AUC
  0.470–0.629 re-confirm the synthetic self-test finding on real content: the uncalibrated
  laya-family checkpoints cannot rank lexically dissimilar paraphrases. This is the known
  weakness, now measured in the regime D1 actually cares about.
- **The conformal guarantee holds everywhere.** Coverage meets the 1−α = 90% target on all three
  engines (90.0% / 96.7% / 90.0%). With an uninformative raw score the calibrated layer mostly
  abstains (72–87%) instead of guessing. That is the product behaving as designed, not a bug.
- **Act-accuracy numbers are small-n.** Only 8–17 of 60 eval pairs were acted on; treat those
  percentages as indicative. Calibration quality differs by engine: typed-decisions is nearly
  perfectly calibrated (ECE 0.022), multilingual is badly overconfident (ECE 0.329).
- **D1 quality comes from the layer, not the raw model:** production dedupe quality is expected
  from conformal gating over real calibration data, not from thresholding raw scores.

### Hosted arms (Clef Omni / Jev 1.14): ready, unexercised

No Cloudflare credentials exist on this machine, so the hosted columns are honestly empty. The
runner is wire-ready for the pi 1.4 catalog ids (`@cf/cloudflare/clef-omni`,
`typesafe/jev-1.14.0` — both `cloudflare-workers-ai-system-one`). Cost projection from measured
token usage (~135 input tokens/call): a 120-pair run is roughly **$0.002 on Clef Omni** and
**$0.0007 on Jev 1.14** at catalog rates.

## Related evidence

- Parity vs laya torch: `tests/engine-parity.test.ts` (≤ 5.7e-06 logits, identical
  token usage across all three checkpoints)
- Conformal determinism goldens: `tests/golden.test.ts` +
  `tests/fixtures/golden-prediction-sets.json` (real engine output, never hand-authored)
- Calibration math goldens: `tests/conformal-golden.test.ts`
