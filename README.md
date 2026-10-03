# pi-reflex

**Local System 1 decisions for [pi](https://pi.dev).** Calibrated choice / yes-no / score
decisions over any JSON-able state in a single non-autoregressive forward pass — as pi
decision tools, as local classifier models for codemode, and as a tier-routing virtual
model. A faithful pure-TypeScript runtime for the open Laya checkpoints (≤ 5.7e-06 logits
parity, byte-identical tokenization; ~55 ms/question on CPU int8), plus a **conformal
guarantee layer** — prediction sets and abstain thresholds with finite-sample coverage —
that neither laya nor [von](https://github.com/wfzyx/von) ship. Zero Python at runtime.

> Formerly `pi-jev-jev`. A Jev-style, laya-compatible open alternative — not affiliated
> with TypeSafe or their "Jev" product.

## Install

```bash
pi install npm:pi-reflex        # pi ≥ 0.99, tested through 1.0.1
pi update npm:pi-reflex         # update later
```

Registers the `reflex` decision tools, three local classifier models (`reflex/*`), the
`reflex/auto` tier router, and an opt-in injection guard. It is also a library —
`pi-reflex/core` (pure logic, no native deps), `/engine` (ONNX runtime), `/harness`
(pinned [pi-continual-harness](CONTRACT-harness.md) contract D1–D4) — code-only on npm;
model artifacts download lazily on first use and are shared by the extension and the
MCP server.

## Decision tools

Engine loads lazily on first use. Every tool returns text **and** typed
`structuredContent` (pi outputSchema), is read-only, and reports failures as
`isError` + a recovery payload instead of throwing, so codemode scripts can degrade.
On pi ≥ 1.0.1 each call draws as one line — `P(true)=0.42 · conf 70% · 12 tok` —
colored by confidence (success ≥ 0.5, warning = abstain, red = error), full
distribution on ctrl+e; the resolver matches by name, so it also covers the
MCP-served twins and reflex calls in resumed sessions / HTML exports.

| Tool | What it does |
|---|---|
| `reflex_decide` | calibrated single-choice decision (routing, triage) |
| `reflex_judge` | calibrated P(true) for a yes/no question |
| `reflex_rate` | ordinal rubric rating (expected level + distribution) |
| `reflex_route` | model tier + guardrail flags for an incoming message in one ~50–200 ms pass |

## Codemode & tool exposure

pi ≥ 1.0's leaner codemode lists each tool as one line and keeps namespace
`instructions` out of the prompt — scripts read them with
`describeNamespace("reflex")`. Probe availability with `"reflex_judge" in tools`
(`typeof` probes no longer work). Every tool declares an `outputSchema`, so scripts
receive the typed payloads; engine failures resolve to `{ type: "error", error,
recovery }` rather than rejecting:

```js
if ("reflex_judge" in tools) {
  const r = await tools.reflex_judge({ state: diff, question: "Does this change delete user data?" });
  if (r.type === "bool" && r.probability > 0.8) return "destructive — ask the user first";
}
```

Keep the tools out of the model's tool list with `PI_REFLEX_EXPOSURE`:

- `codemode` — listed one line each in the `codemode` tool, callable from scripts.
- `deferred` — listed nowhere; `tool_search` finds and activates them on demand
  (pi ≥ 1.0 keeps deferred tools across resume/`/reload`). Requires `tool_search`:
  on `session_start` the extension checks, warns when it is missing, and activates
  `tool_search` itself if nothing else could reach the tools.
- `direct` (default) — declared to the model normally; also callable from scripts.

`/reflex` shows the effective exposure and `tool_search` state.

## Classifier models (`reflex/*`)

The extension registers a `reflex` provider with three **local classifier models** —
`reflex/multilingual`, `reflex/english`, `reflex/typed-decisions` — next to TypeSafe's
hosted Jev and Cloudflare's Clef (pi ≥ 1.0.1), but offline, private, and free (no API
key, no egress). Codemode scripts reach them through the uniform classifier interface:

```js
const reflex = await models.getModelOfType("classifier", "reflex", "multilingual");
const r = await models.classify(reflex, {
  state: { message: "The change works, thanks." },
  questions: { approved: { type: "bool", instructions: "Does the user approve?" } },
});
return r.answers; // { approved: { type: "bool", probability: 0.97 } }
```

Other extensions can do the same via `ctx.modelRegistry.classify()`.

## Virtual model `reflex/auto` (tier routing)

Select `reflex/auto` in `/model` and each user turn is classified locally
(`MODEL_ROUTER` + `INJECTION_GUARD`, one forward pass) before being dispatched to a
cheap, mid, or frontier model. Map tiers via env — `PI_REFLEX_TIER_SMALL`,
`PI_REFLEX_TIER_MID`, `PI_REFLEX_TIER_FRONTIER` = `provider/model-id` (e.g.
`PI_REFLEX_TIER_SMALL=anthropic/claude-haiku-4-5`).
Continuations/retries stay sticky on the turn's model (prompt caches survive); virtual
thinking level `high` bumps one tier; unmapped tiers fall back to the previous
physical model; engine failure degrades to mid tier instead of blocking the turn.

### Quantization (int8 vs fp32)

- **`int8` (default)** — ~400 MB/checkpoint, ~55 ms/question: the right pick for
  routing, triage, and guardrails.
- **`fp32`** — ~1.6 GB/checkpoint, full precision: set `PI_REFLEX_QUANT=fp32` in the
  environment **before launching pi** (read once at activation; restart to switch).

Both quants cache side by side in `~/.pi-reflex/engines/`, so switching never
re-downloads the other. The MCP server reads the same variable.

## Prompt-injection guard (opt-in)

`PI_REFLEX_GUARD=1` enables a `context_with_system` guard: new user messages are
classified once (cached), and flagged ones (P ≥ `PI_REFLEX_GUARD_THRESHOLD`, default
0.75) are annotated as untrusted data before each provider request — never removed,
never reordered, budget-capped (≤ 3 new messages/request), circuit breaker on engine
failure.

## MCP server

```bash
pi mcp add reflex -- node <pkg>/bin/pi-reflex-mcp.js   # user-level; or: PI_REFLEX_MCP=1 (extension registers it)
```

A zero-dependency stdio MCP server exposing the same four tools to any MCP client.
With a **user-level** server, pi ≥ 1.0.1 per-project overrides work: a `.pi/mcp.json`
entry sets only `enabled` or `exposure` (`{ "mcpServers": { "reflex": { "enabled": false } } }`),
and `/mcp` toggles the same state. A `command`/`url` entry fully replaces the server;
`/reload` after external edits. (The session-level `PI_REFLEX_MCP=1` registration is
not overridable this way — register user-level for per-project control.)

## Status

`/reflex` shows engine, classifier, router, guard, and MCP status.

## Environment variables

| Variable | Values (default) | Purpose |
|---|---|---|
| `PI_REFLEX_ENGINE` | `english` \| `multilingual` \| `typed-decisions` (`multilingual`) | default engine for tools, router, guard |
| `PI_REFLEX_QUANT` | `int8` \| `fp32` (`int8`) | precision + artifact set; set before launch, restart to switch |
| `PI_REFLEX_ARTIFACTS` | directory | local artifacts override (skips cache + download) |
| `PI_REFLEX_TIER_SMALL/MID/FRONTIER` | `provider/model-id` | `reflex/auto` tier mapping |
| `PI_REFLEX_GUARD` | `1` \| `0` (`0`) | prompt-injection guard |
| `PI_REFLEX_GUARD_THRESHOLD` | 0–1 (`0.75`) | guard flag threshold |
| `PI_REFLEX_EXPOSURE` | `direct` \| `codemode` \| `deferred` (`direct`) | tool visibility to the model |
| `PI_REFLEX_MCP` | `1` \| `0` (`0`) | register the MCP server for the session |
| `PI_REFLEX_QUIET` | `1` \| `0` (`0`) | silence the startup banner (`/reflex` always shows status) |

## Artifacts

Engines resolve: `$PI_REFLEX_ARTIFACTS` → cache (`~/.pi-reflex/engines`) → HF download
(`ngSoftware/pi-reflex-artifacts`, lazy, on first use). Interrupted downloads leave a
`.part` file and **resume from that byte offset** on the next attempt (transient errors
retry within the call; progress shows in the pi footer) — a failed first use just needs
a retry, not a cleanup. Generate locally instead with
`python tools/export_onnx.py --checkpoint convaiinnovations/laya --subfolder <name> --out <dir> --int8`.

## Constraints worth knowing

- **States and criteria must be JSON-clean**: `NaN`/`Infinity` serialize as `null`
  (Python prints `NaN`), `-0` as `0` (Python prints `-0.0`).
- **Integer-like choice labels** (`"1"`, `"10"`, `"2"`) are reordered by JavaScript
  (ascending, first) unlike Python dicts — keep them ascending; pi-reflex warns at
  runtime when it detects a diverging order.
- Reported probabilities are rounded to 4 decimals half-away-from-zero; laya uses
  banker's rounding (drift ≤ 1e-4).

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — design, ports table, measured latencies
- [CONTRACT-harness.md](CONTRACT-harness.md) — pinned consumer contract (pi-continual-harness)
- [BENCHMARKS.md](BENCHMARKS.md) — latency evidence + A4 accuracy eval runner (reflex vs hosted clef)
- [RESEARCH.md](RESEARCH.md) — the 53-paper research stack behind the design

## License

Apache-2.0 — see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Not affiliated with TypeSafe or their "Jev" product.
