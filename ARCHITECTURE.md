# pi-reflex Architecture

TypeScript-native System 1 decision engine — a best-of-both layer over
[laya](https://github.com/NandhaKishorM/laya)'s open Apache-2.0 checkpoints, with
guarantee layers neither laya nor [von](https://github.com/wfzyx/von) ship, targeting
**AI harnesses (pi first)** as model-router + guardrail brain.

## Answers: "what would be needed to remove Python completely?"

**Python can be removed 100% from the runtime.** It survives only in two dev-only
spots that never ship to users:

| Where Python appears | Who runs it | Ships to users? |
|---|---|---|
| One-time ONNX export of laya checkpoints (`tools/export_onnx.py`) | us, on a dev box | ❌ users get `.onnx` artifacts |
| Future training / fine-tuning (Phase B) | us, on GPU boxes | ❌ users get new checkpoints |

### The swap, layer by layer

| laya (Python) | pi-reflex (TS) | Status |
|---|---|---|
| `transformers` + `torch` inference | **onnxruntime-node** (native, prebuilt binaries; CPU EP everywhere, CUDA EP optional) running a **fused ONNX graph** of DecisionModel (encoder + type_emb + 2-layer head + scorer + act_head, dynamic axes for seq/kmax/batch) | ✅ done — 3/3 checkpoints exported, parity ≤ 5.7e-06 logits |
| `AutoTokenizer` (Rust via HF) | **@huggingface/tokenizers** (tokenizers.js — the transformers.js v3 engine, pure JS, ~8 kB) | ✅ done — identical token counts vs Python |
| `common.py` sequence builder | **exact port** `src/core/serialize.ts` incl. head-budget squeeze, marker filtering, the `st[-room:]` room=0 quirk | ✅ done, 55 tests |
| temperature bucketing + entropy confidence | **exact port** `src/core/calibration.ts` (`temperature_by_options` buckets `choice:3-5` etc.) | ✅ done |
| answer assembly (`system_one` post-proc) | **exact port** `src/core/answers.ts` (choice/score/noul shapes, `act_probability`, round-4) | ✅ done |
| `lang.py` script/LID detection | **exact port** `src/lang/analyze.ts` (24 script blocks, weighted stopwords, diacritic margin) — zero deps | ✅ done |
| `router.py` route decision | **exact port** `src/router/route.ts` (precedence: model > task > workflow > lang > detection > default) | ✅ done |
| heuristic confidence gating | **our conformal layer** `src/core/conformal.ts`: split-conformal sets (choice/noul), intervals (score), risk-controlled escalation thresholds — finite-sample guarantees | ✅ done (ours) |
| PyPI package + torch dependency (GBs) | npm package, installs in seconds | ✅ this repo |

### Why this is the right bet

1. **Distribution edge nobody has.** laya and von are both Python+PyTorch — unusable
   from a TS agent harness without a heavyweight sidecar. pi (and Claude Code, Cursor,
   any MCP client) can `npm install pi-reflex` and get a local System 1 brain.
2. **The heavy lifting is open.** laya's checkpoints are Apache 2.0; the architecture
   is ModernBERT/mmBERT + small heads (421M/322M params) — we don't retrain to reach
   parity, we port the runtime and add guarantees.
3. **Same weights, same numbers.** Our ports are line-exact (55 tests mirror Python
   semantics incl. edge quirks), and the export script parity-checks ONNX vs torch.

## Component map

```
pi-reflex/
├── src/core/          # pure logic — exact laya ports + our conformal layer
│   ├── serialize.ts   #   build_sequence, budgets, collate
│   ├── calibration.ts #   temp buckets, entropy confidence, softmax
│   ├── answers.ts     #   system_one answer assembly
│   └── conformal.ts   #   OUR EDGE: coverage guarantees + risk control
├── src/lang/analyze.ts  # script detection + Latin LID (dependency-free)
├── src/router/route.ts  # checkpoint routing decision (pure)
├── tools/export_onnx.py # DEV-ONLY: checkpoint → fused ONNX (+int8), parity check
├── reference/laya/      # vendored laya sources (Apache 2.0) used by the export
└── tests/               # 55 tests mirroring python semantics
```

## Roadmap

- **A1 (done):** core ports + router + conformal, 55 tests, typecheck green.
- **A2 (done):** all 3 checkpoints exported to fused ONNX (parity ≤ 5.7e-06 logits vs torch;
  int8 variants), onnxruntime-node + tokenizers.js glue, `Engine.systemOne()` — **parity
  tests vs real laya torch on every checkpoint with identical token usage**. CPU fp32 latency
  (4 questions, one pass): multilingual ~220 ms/call (~55 ms/q), english ~590 ms/call
  (~147 ms/q). Zero Python in the runtime — proven.
- **A2.5 (done): deep-review hardening** — H1 python-spaced JSON for non-string instructions;
  H3 integer-like-label warning (JS owns their order — ECMAScript normalizes literals AND
  JSON.parse); M1 conformal input validation (throw, never silent-NaN); M7 collate guard;
  M9 noUncheckedIndexedAccess on; M5 `Engine.batchQuestion` (contract D4 batching);
  M3 `dispose()`; M2 tokenizer vocab fallback; M4 exports subpaths (`./core` is
  native-dep-free); H4 conformal golden tests + empty-set⇒ABSTAIN documented; M6 lazy
  parity loading; H2 LICENSE + THIRD_PARTY_NOTICES + README (Apache-2.0 compliance for
  vendored laya code/weights); dev toolchain on vitest 5 — **npm audit: 0 vulnerabilities,
  84/84 tests, typecheck + build clean**.
- **A3:** Surfaces — pi extension (tools: `decide`/`judge`/`rate`), MCP server,
  `/v1/systemone` HTTP shim; harness presets: model-router (cheap vs frontier),
  prompt-injection guard, triage. **Build these against the pinned first-consumer
  contract: [CONTRACT-harness.md](CONTRACT-harness.md)** (decision shapes, abstain
  semantics, latency budgets, calibration-corpus schema for pi-continual-harness).
- **A4:** Benchmarks (`BENCHMARKS.md`): reproduce laya's numbers through the TS
  stack; add conformal coverage plots; latency on CPU int8 vs GPU.
- **B:** our own checkpoints (RESEARCH.md P0–P4) — CORAL ordinal heads,
  retrieve-rerank for K>20, RLCD training, distilled router model.

## Honesty notes

- CPU int8 latency for the 421M English checkpoint will land ~80–250 ms depending on
  threads; the 322M multilingual ~40–120 ms. GPU (CUDA EP) restores ~10–30 ms.
  Apple Silicon: CPU EP only in onnxruntime-node — fine for the base checkpoint.
- laya's `round()` is banker's rounding; ours is half-away-from-zero. Drift ≤ 1e-4
  on reported probabilities — documented, acceptable.
- Reason strings in `route()` are ours (same semantics, phrased for our output).
