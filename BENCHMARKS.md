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
- **Accuracy is now measured (A4, curated corpus):** raw checkpoints are at/near chance as
  thresholded judges (accuracy@0.5 50.8–56.7%, AUC 0.47–0.63) and the value is the conformal
  layer — see §A4 for numbers and provenance.

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

Local arms (all three checkpoints, curated corpus):

| reflex engine | accuracy@0.5 | ECE-10 | AUC | latency p50 / p95 | conformal coverage (α=0.1) | abstain | act-accuracy (n) |
|---|---|---|---|---|---|---|---|
| multilingual | 56.7% | 0.329 | 0.470 | 82 / 107 ms | **90.0%** | 83.3% | 40.0% (10) |
| typed-decisions | 50.8% | 0.022 | 0.530 | 171 / 212 ms | **96.7%** | 86.7% | 75.0% (8) |
| english | 52.5% | 0.037 | 0.629 | 169 / 303 ms | **90.0%** | 71.7% | 64.7% (17) |

Hosted arms (Workers AI, same 120 pairs, run 2026-10-26; latency includes network RTT;
the temporary token used was rolled after the runs):

| hosted model | accuracy@0.5 | ECE-10 | AUC | latency p50 / p95 | billed input tokens |
|---|---|---|---|---|---|
| @cf/cloudflare/clef-omni | **95.0%** | 0.066 | 0.996 | 243 / 565 ms | 28,037 |
| @cf/cloudflare/clef | 94.2% | 0.048 | 0.989 | 341 / 570 ms | 28,396 |
| @cf/cloudflare/clef-flash | 92.5% | 0.113 | **0.997** | 495 / 1222 ms | 28,396 |

How to read this (measured, not spun):

- **Raw local checkpoints are at or near chance as thresholded judges.** Accuracy@0.5 is
  50.8–56.7% on every engine: the default 0.5 threshold extracts nothing usable from the raw
  scores. Ranking signal is weak and uneven: AUC 0.470 for multilingual (at/below chance),
  0.530 for typed-decisions (marginal), 0.629 for english (measurably above chance). So the
  engines are not uniformly "unable to rank" — english carries real but weak signal — but all
  three sit far from the hosted 0.989–0.997 and none is trustworthy as a raw paraphrase judge.
  This re-confirms the synthetic self-test finding on real content, in the regime D1 cares about.
- **The conformal guarantee holds on every local engine.** Coverage meets the 1−α = 90% target
  (90.0% / 96.7% / 90.0%). With an uninformative raw score the calibrated layer mostly abstains
  (72–87%) instead of guessing. That is the product behaving as designed, not a bug. Hosted
  classifiers have no such layer: they answer every pair at their measured accuracy.
- **Hosted models dominate accuracy on this hard corpus, at 3–6× the latency plus cost and
  egress.** Clef Omni scores 95.0% / AUC 0.996 vs multilingual's 56.7% / 0.470. That is the
  expected trade: 30B-MoE served weights vs a 322M local checkpoint. pi-reflex's case is not
  raw accuracy parity; it is local, free, private, offline decisions with a guarantee layer.
- **Latency ladder:** local multilingual p50 82 ms; clef-omni 243 ms; clef 341 ms; clef-flash
  495 ms (p95 1222 ms). The flash variant is both the least accurate and the slowest here.
- **Act-accuracy numbers are small-n.** Only 8–17 of 60 eval pairs were acted on; treat those
  percentages as indicative. Calibration quality differs by engine: typed-decisions is nearly
  perfectly calibrated (ECE 0.022), multilingual is badly overconfident (ECE 0.329).
- **D1 quality comes from the layer, not the raw model:** production dedupe quality is expected
  from conformal gating over real calibration data, not from thresholding raw scores.

### Hosted arms: exercised (Clef family) + Jev 1.14 pending enablement

- **Run 2026-10-26** on a free-plan account: `@cf/cloudflare/clef-omni`, `@cf/cloudflare/clef`,
  `@cf/cloudflare/clef-flash` all served 120/120 calls (table above). Measured cost from billed
  tokens: roughly **$0.004** for all three runs combined at catalog input rates.
- **Jev 1.14 is in the catalog but not provisioned.** The dashboard lists
  `typesafe/jev-1.14.0` (pi 1.4's id matches), yet every REST form
  (`typesafe/jev`, `typesafe/jev-1.14`, `typesafe/jev-1.14.0`) answers 400 "No route for that URI"
  while sibling models serve fine. Per Cloudflare's Oct 2026 changelog, TypeSafe models roll out
  account-by-account; enabling is a dashboard action on the model page, not a REST call. The arm
  stays wired for the day it is enabled.
- **Transport quirks discovered (now handled in the runner):** the model id must be sent in the
  path with a RAW `@` — URL-encoding it (`%40cf/...`) makes Cloudflare's router return 400 "No
  route for that URI". Classifier inputs go unwrapped in the body (`{state, questions}`), and
  answers arrive at `result.answers` with `usage.input_tokens` as the billed quantity.

## Related evidence

- Parity vs laya torch: `tests/engine-parity.test.ts` (≤ 5.7e-06 logits, identical
  token usage across all three checkpoints)
- Conformal determinism goldens: `tests/golden.test.ts` +
  `tests/fixtures/golden-prediction-sets.json` (real engine output, never hand-authored)
- Calibration math goldens: `tests/conformal-golden.test.ts`
