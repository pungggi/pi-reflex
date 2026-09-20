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
- Accuracy is NOT benchmarked yet (A4): no headline claims until the corpus-driven
  conformal coverage evals run on real harness exports.

## Related evidence

- Parity vs laya torch: `tests/engine-parity.test.ts` (≤ 5.7e-06 logits, identical
  token usage across all three checkpoints)
- Conformal determinism goldens: `tests/golden.test.ts` +
  `tests/fixtures/golden-prediction-sets.json` (real engine output, never hand-authored)
- Calibration math goldens: `tests/conformal-golden.test.ts`
