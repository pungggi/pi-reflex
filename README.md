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

**As a pi extension** (pi ≥ 0.99, tested through pi 1.0.1) — the primary way to run pi-reflex:

```bash
pi install npm:pi-reflex
```

Registers the `reflex` tools, the local `reflex/*` classifier models, the `reflex/auto`
tier router, and the opt-in injection guard — see
[Use as a pi extension](#use-as-a-pi-extension). Update later with `pi update npm:pi-reflex`.

**As a library** — for embedding the decision engine in your own code:

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

pi-reflex ships a pi-package extension (tools for pi coding-agent sessions, **pi ≥ 0.99, tested through pi 1.0.1**):

```bash
pi install npm:pi-reflex                  # from npm
pi install /absolute/path/to/pi-reflex    # or a local checkout
```

Tools (engine loads lazily on first use; `multilingual` int8 by default). Every tool returns
text **and** a typed `structuredContent` payload (pi outputSchema), is read-only annotated,
and grouped under the `reflex` namespace:

| Tool | What it does |
|---|---|
| `reflex_decide` | calibrated single-choice decision (routing, triage) |
| `reflex_judge` | calibrated P(true) for a yes/no question |
| `reflex_rate` | ordinal rubric rating (expected level + distribution) |
| `reflex_route` | **model tier + guardrails for an incoming message in one ~50–200 ms pass** |

On pi ≥ 1.0.1 the extension also registers compact tool renderers (`pi.registerToolRenderer`):
tool calls draw as one line — `P(true)=0.42 · conf 70% · 12 tok` — colored by confidence
(success ≥ 0.5, warning = abstain, red = error), with the full probability distribution on
ctrl+e expansion. The resolver matches by name, so the same rendering covers the extension's
tools **and** their MCP-served twins (`mcp__reflex__reflex_*`), including reflex calls in
resumed sessions and HTML exports drawn before the server connected.

### Codemode & tool exposure

pi ≥ 1.0's leaner codemode lists each tool as one line (its `description`) and keeps the
namespace `instructions` out of the prompt — codemode scripts read them with
`describeNamespace("reflex")`. Probe availability with `"reflex_judge" in tools` (`typeof`
probes no longer work in codemode). Every tool declares an `outputSchema`, so scripts receive
the typed `structuredContent` payloads instead of text; engine failures resolve to
`{ type: "error", error, recovery }` rather than rejecting, so scripts can degrade:

```js
if ("reflex_judge" in tools) {
  const r = await tools.reflex_judge({ state: diff, question: "Does this change delete user data?" });
  if (r.type === "bool" && r.probability > 0.8) return "destructive — ask the user first";
}
```

Keep the tools out of the model's tool list with `PI_REFLEX_EXPOSURE`:

- `codemode` — listed one line each in the `codemode` tool, callable from scripts.
- `deferred` — not listed anywhere; `tool_search` finds and activates them on demand
  (pi ≥ 1.0 keeps deferred tools across resume/`/reload`). **Deferred requires
  `tool_search`**: on `session_start` the extension checks for it and warns when it is
  missing; if neither `tool_search` nor `codemode` is active the tools would be
  unreachable, so it activates `tool_search` itself (when the host registered it) and
  says so. `/reflex` shows the effective exposure and `tool_search` state.

### Classifier models (`reflex/*`)

The extension registers a `reflex` provider with three **local classifier models** —
`reflex/multilingual`, `reflex/english`, `reflex/typed-decisions` — next to TypeSafe's hosted
Jev classifiers, but offline, private, and free (no API key). Codemode scripts reach them
through the uniform classifier interface:

```js
const reflex = await models.getModelOfType("classifier", "reflex", "multilingual");
const r = await models.classify(reflex, {
  state: { message: "The change works, thanks." },
  questions: { approved: { type: "bool", instructions: "Does the user approve?" } },
});
return r.answers; // { approved: { type: "bool", probability: 0.97 } }
```

Extensions can do the same via `ctx.modelRegistry.classify()`. `bool`/`choice`/`score`
map 1:1 onto our `noul`/`choice`/`score` primitives. Hosted alternatives — TypeSafe's Jev,
Cloudflare's Clef and Clef Flash (pi ≥ 1.0.1) — work through that same interface when you
have their API keys; `reflex/*` stays local, private, and free, with no key and no egress
(`tools/eval-corpus.mjs` runs both arms over the same corpus for an apples-to-apples read).

### Virtual model `reflex/auto` (tier routing)

Select `reflex/auto` in `/model` and each user turn is classified locally
(`MODEL_ROUTER` + `INJECTION_GUARD`, one forward pass) before being dispatched to a
cheap, mid, or frontier model. Continuations/retries stay sticky on the turn's model
(prompt caches survive). Map tiers via env:

```bash
export PI_REFLEX_TIER_SMALL=anthropic/claude-haiku-4-5
export PI_REFLEX_TIER_MID=anthropic/claude-sonnet-4-5
export PI_REFLEX_TIER_FRONTIER=anthropic/claude-opus-4-5
```

Virtual thinking level `high` bumps one tier; unmapped tiers fall back to the previous
physical model; engine failure degrades to mid tier instead of blocking the turn.

### Prompt-injection guard (opt-in)

`PI_REFLEX_GUARD=1` enables a `context_with_system` guard: new user messages are
classified once (cached), and flagged ones (P ≥ `PI_REFLEX_GUARD_THRESHOLD`, default 0.75)
are annotated as untrusted data before each provider request — never removed, never
reordered, budget-capped (≤ 3 new messages/request), with a circuit breaker on engine
failure.

### MCP server

```bash
pi mcp add reflex -- node <pkg>/bin/pi-reflex-mcp.js   # or: PI_REFLEX_MCP=1 (extension registers it)
```

A zero-dependency stdio MCP server exposing the same four tools to any MCP client
(pi, Claude Code, Cursor). JSON-RPC per line; `initialize` / `tools/list` / `tools/call`.

#### Per-project overrides (pi ≥ 1.0.1)

A **user-level** `reflex` server (defined by `pi mcp add reflex -- …` in
`~/.pi/agent/mcp.json`) can be flipped per project with a `.pi/mcp.json` entry
that sets only `enabled`, `exposure`, or `toolExposure` — no command needed,
and `env`/`auth` carry over. Turn it off in one repo:

```json
{ "mcpServers": { "reflex": { "enabled": false } } }
```

…or declare the tools to the model in one repo (they are `codemode`-only by default):

```json
{ "mcpServers": { "reflex": { "exposure": "direct" } } }
```

> **Scope caveat:** overrides resolve against user-level `mcp.json` servers only.
> The extension's `PI_REFLEX_MCP=1` registration is session-level and **not**
> overridable this way — pi rejects the project entry with *"needs a global server
> to override"*. If you installed via `PI_REFLEX_MCP=1` and want per-project
> control, register the server user-level instead: `pi mcp add reflex -- node
> <pkg>/dist/mcp/server.js` (then unset `PI_REFLEX_MCP`).

`/mcp` toggles the same per-project state interactively for user-level servers, and
a `.pi/mcp.json` entry with a `command`/`url` fully replaces the user-level server.
Run `/reload` after editing the file outside the session.

`/reflex` shows engine, classifier, router, guard, and MCP status.

Env: `PI_REFLEX_ENGINE` (english|multilingual|typed-decisions), `PI_REFLEX_QUANT` (int8|fp32),
`PI_REFLEX_ARTIFACTS` (local artifacts dir), `PI_REFLEX_TIER_{SMALL,MID,FRONTIER}` (`provider/model-id`),
`PI_REFLEX_GUARD` (1|0), `PI_REFLEX_GUARD_THRESHOLD`, `PI_REFLEX_MCP` (1|0),
`PI_REFLEX_EXPOSURE` (`codemode` lists the tools one line each in the codemode tool; `deferred`
leaves discovery to `tool_search` — both keep them out of the model's tool list; `direct` is the
explicit default; anything else warns and falls back to `direct`),
`PI_REFLEX_QUIET` (default `0` — startup banner on; `1` silences it — `/reflex` always shows status).

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
(`ngSoftware/pi-reflex-artifacts`, lazy, on first use). Generate locally instead:

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
- [BENCHMARKS.md](BENCHMARKS.md) — §3 latency evidence + A4 accuracy eval runner (reflex vs hosted clef)

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
