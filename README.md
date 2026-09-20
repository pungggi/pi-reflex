# pi-reflex

**TypeScript-native System 1 decision engine.** Typed decisions (`choice` / `score` / `noul`)
with calibrated probabilities over any state — text, ticket, or JSON document — in a single
non-autoregressive forward pass. Zero Python at runtime.

> Formerly `pi-jev-jev`. A Jev-style, laya-compatible open alternative — not affiliated
> with TypeSafe or their "Jev" product.

It is a faithful, pure-TypeScript runtime for the open
[Laya](https://github.com/NandhaKishorM/laya) checkpoints (Apache-2.0), plus a
**conformal guarantee layer** neither laya nor [von](https://github.com/wfzyx/von)
ship: prediction sets and escalation thresholds with finite-sample coverage
guarantees instead of heuristic confidence gating. See
[ARCHITECTURE.md](ARCHITECTURE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- **Parity**: ≤ 5.7e-06 logits vs the real torch runtime; byte-identical tokenization
  (verified against Python on all three checkpoints).
- **Latency** (CPU fp32, 4 questions in one pass): multilingual ~220 ms/call (~55 ms/q),
  English ~590 ms/call (~147 ms/q). int8 graphs included.
- **No Python, no PyTorch**: `onnxruntime-node` (native, prebuilt) + `tokenizers.js` (pure JS).

## Install

```bash
npm install pi-reflex
```

The npm package is code-only. Model artifacts (~400 MB int8 / ~1.6 GB fp32 per checkpoint)
are generated or downloaded separately — see [Artifacts](#artifacts).

Subpath imports keep the native runtime optional:

```ts
import { ChoiceConformal, route, detectScript } from "pi-reflex/core";   // pure logic, no native deps
import { Engine } from "pi-reflex/engine";                               // needs onnxruntime-node
```

## Quickstart

```ts
import { Engine } from "pi-reflex/engine";
import { NoulConformal } from "pi-reflex/core";

const engine = await Engine.fromArtifacts("./artifacts/multilingual");

const result = await engine.systemOne(
  { from: "user@acme.com", subject: "Duplicate charge", body: "We were billed twice..." },
  {
    department: {
      type: "choice",
      instructions: "Which department should handle this?",
      criteria: { billing: "invoices, refunds", technical: "bugs, outages" },
    },
    urgent: { type: "noul", instructions: "Does this need immediate action?" },
    severity: { type: "score", instructions: "Rate severity.", criteria: ["low", "high", "critical"] },
  },
);

result.answers.department.choice;      // "billing"
result.answers.urgent.noul;            // calibrated P(true)
result.answers.severity.score;         // expected level, e.g. 0.9
```

Contract-style abstention (act only on a singleton prediction set):

```ts
const noul = new NoulConformal().fit(calibrationP, calibrationLabels, 0.1);
const [falseIn, trueIn] = noul.set(pTrue);
const verdict = falseIn !== trueIn ? (trueIn ? "true" : "false") : "abstain";
```

Batching M states against one question in a single forward pass (hot paths like
per-message guardrails):

```ts
const answers = await engine.batchQuestion(states, {
  type: "score",
  instructions: "How relevant is this item to the current task?",
  criteria: ["irrelevant", "relevant"],
});
```

## Use as a pi extension

pi-reflex ships a pi-package extension (tools for pi coding-agent sessions):

```bash
pi install /absolute/path/to/pi-reflex   # local; npm:pi-reflex when published
```

Tools (engine loads lazily on first use; `multilingual` int8 by default):

| Tool | What it does |
|---|---|
| `reflex_decide` | calibrated single-choice decision (routing, triage) |
| `reflex_judge` | calibrated P(true) for a yes/no question |
| `reflex_rate` | ordinal rubric rating (expected level + distribution) |
| `reflex_route` | **model tier + guardrails for an incoming message in one ~200 ms pass** |

`/reflex` shows engine status. Env: `PI_REFLEX_ENGINE` (english|multilingual|typed-decisions),
`PI_REFLEX_QUANT` (int8|fp32), `PI_REFLEX_ARTIFACTS` (local artifacts dir).

## Use as the pi-continual-harness companion

`pi-reflex/harness` implements the pinned [CONTRACT-harness.md](CONTRACT-harness.md)
call-sites D1–D4: `createDedupeSimilarity` (cached seam similarity + conformal abstain),
`createCreateGate`, `createImportanceRescorer`, `createInjectionRelevance` (≤3-item
policy per the measured §3 budget). Uncalibrated ⇒ plain scores; over budget ⇒ the
contract's own fallback. **Honesty note:** raw checkpoints judge paraphrase-sameness
(D1) poorly uncalibrated (~0.05 for near-duplicates) — thresholds and calibration
land with the real corpus in A4.

## Artifacts

Engines resolve: `$PI_REFLEX_ARTIFACTS` → cache (`~/.pi-reflex/engines`) → HF download
(`pungggi/pi-reflex-artifacts`, lazy, on first use). Generate locally instead:

```bash
python tools/export_onnx.py --checkpoint convaiinnovations/laya --out artifacts/english --int8
python tools/export_onnx.py --checkpoint convaiinnovations/laya --subfolder multilingual --out artifacts/multilingual --int8
python tools/export_onnx.py --checkpoint convaiinnovations/laya --subfolder typed-decisions --out artifacts/typed-decisions --int8
python tools/parity_reference.py   # ground-truth fixtures from the real laya runtime
npm test                           # includes ONNX-vs-torch parity tests
```

## API surface

| Import | Contents |
|---|---|
| `pi-reflex` | everything (loads `onnxruntime-node`) |
| `pi-reflex/core` | primitives, serialization, calibration, **conformal layer** — zero native deps |
| `pi-reflex/router` | checkpoint routing decision (script/LID detection, precedence rules) |
| `pi-reflex/lang` | script + language detection |
| `pi-reflex/engine` | `Engine` (ONNX runtime), tokenizer adapter, session |

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — design, ports table, measured latencies
- [CONTRACT-harness.md](CONTRACT-harness.md) — pinned consumer contract (pi-continual-harness)
- [RESEARCH.md](RESEARCH.md) — the 53-paper research stack behind the design
- [BENCHMARKS.md](BENCHMARKS.md) — planned (A4)

## Constraints worth knowing

- **States and criteria must be JSON-clean**: `NaN`/`Infinity` serialize as `null`
  (Python prints `NaN`), `-0` as `0` (Python prints `-0.0`).
- **Integer-like choice labels** (`"1"`, `"10"`, `"2"`) are reordered by JavaScript
  (ascending, first) unlike Python dicts — keep them ascending or use non-numeric
  labels; pi-reflex warns at runtime when it detects a diverging order.
- Reported probabilities are rounded to 4 decimals half-away-from-zero; laya uses
  banker's rounding (drift ≤ 1e-4).

## License

Apache-2.0 — see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Not affiliated with TypeSafe or their "Jev" product.
