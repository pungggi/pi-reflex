# pi-reflex

**The decision layer for [pi](https://pi.dev).** Calibrated choice / yes-no / score
decisions over any JSON-able state in a single non-autoregressive forward pass: as pi
decision tools, as local classifier models for codemode, and as a tier-routing virtual
model, all backed by a **conformal guarantee layer** and a zero-server local runtime.

The runtime is a faithful pure-TypeScript port of the open Laya checkpoints (≤ 5.7e-06
logits parity, byte-identical tokenization; ~55 ms/question on CPU int8). The layer on
top is what sets it apart: a mathematical safety net that either guarantees the correct
answer is within a small set of options, or safely says "I don't know" instead of
guessing. Neither laya nor [von](https://github.com/wfzyx/von) ship this kind of layer.

> Formerly `pi-jev-jev`. A Jev-style, laya-compatible open alternative, not affiliated
> with TypeSafe or their "Jev" product.

## Install

```bash
pi install npm:pi-reflex        # pi ≥ 1.0.0
```

Installing `pi-reflex` registers the following components:
- The `reflex` decision tools
- Three local classifier models (`reflex/*`)
- The `reflex/auto` tier router
- An opt-in prompt-injection guard

It also serves as a modular library, split into:
- `pi-reflex/core`: Pure logic with zero native dependencies
- `pi-reflex/engine`: The ONNX runtime wrapper
- `pi-reflex/harness`: The pinned [pi-continual-harness](CONTRACT-harness.md) contract (D1–D4)

**Note:** The npm packages are code-only. Model artifacts download lazily on first use and are automatically shared between the extension and the MCP server.

## Why pi-reflex in the pi 1.1 era

Since pi 1.1, inference itself is becoming a commodity: Laya-family checkpoints served
by llama.cpp ≥ 0.6.0 are listed natively as pi classifiers, and hosted decision models
(TypeSafe's Jev, Cloudflare's Clef, OpenAI's GPT-6 Luna) are one API key away. `classify()` calls are no longer the differentiator: the layer around them is.

- **Conformal guarantees**: calibrated abstention with mathematical backing. Either the
  correct answer is certified to be within a returned option set, or the decision
  abstains instead of guessing. No other pi decision surface (native, hosted, or
  llama.cpp-served) ships this.
- **Agent-facing tools, not just script APIs**: the `reflex_*` tools put calibrated
  decisions directly in the agent's tool list with one-line rendering, typed
  `structuredContent` payloads, error-recovery contracts, and an explicit abstain rule
  (confidence < 0.5 renders as a warning; fall back to the model). Catalog classifiers
  alone give you none of that.
- **Zero-server operation**: `pi install npm:pi-reflex` is the whole setup; artifacts
  download lazily and are shared across surfaces. The llama.cpp path requires obtaining
  and running llama-server; hosted paths require keys and send data off-box.
- **Behavior, not just answers**: `reflex/auto` tier routing (with stickiness and
  graceful degradation) and the opt-in prompt-injection guard act on every turn;
  classifiers alone decide nothing on their own.
- **Free, private, offline**: no API keys, no network egress at decision time, no
  per-token cost.

**When to use what:** already operating llama.cpp? Its native classifiers are a great
`classify()` backend. Pair them with pi-reflex for the guarantee layer, the agent tools,
the router, and the guard; the runtime then serves as your zero-dependency fallback.
Hosted models fit when you want someone else's weights and accept egress; pi-reflex fits
when you want decisions you can trust locally.

## Decision tools

The engine loads lazily on first use.

All tools are read-only and return both text and a typed `structuredContent` (pi `outputSchema`). To help codemode scripts fail gracefully, tools return failures as an `isError` flag with a recovery payload instead of throwing exceptions.

In pi ≥ 1.0.1, the UI renders each call as a single, compact line (e.g., `P(true)=0.42 · conf 70% · 12 tok · 55ms`). The trailing latency is pi 1.1's measured `durationMs` for the call. This line is color-coded by confidence (success ≥ 0.5, warning = abstain, red = error), and you can press `ctrl+e` to see the full distribution. Because the UI matches tools by name, this neat formatting also applies to MCP-served twins and resumed sessions; the latency suffix requires pi ≥ 1.1 and does not appear in HTML exports (pi's HTML renderer does not pass durations), and it is omitted on results stored before 1.1.

| Tool            | What it does                                                                |
| --------------- | --------------------------------------------------------------------------- |
| `reflex_decide` | calibrated single-choice decision (routing, triage)                         |
| `reflex_judge`  | calibrated P(true) for a yes/no question                                    |
| `reflex_rate`   | ordinal rubric rating (expected level + distribution)                       |
| `reflex_route`  | model tier + guardrail flags for an incoming message in one ~50–200 ms pass |

## Codemode & tool exposure

pi ≥ 1.0's leaner codemode lists each tool as one line and keeps namespace
`instructions` out of the prompt. Scripts read them with
`describeNamespace("reflex")`. Probe availability with `"reflex_judge" in tools`
(`typeof` probes no longer work). Every tool declares an `outputSchema`, so scripts
receive the typed payloads; engine failures resolve to `{ type: "error", error,
recovery }` rather than rejecting:

```js
if ("reflex_judge" in tools) {
  const r = await tools.reflex_judge({
    state: diff,
    question: "Does this change delete user data?",
  });
  if (r.type === "bool" && r.probability > 0.8)
    return "destructive: ask the user first";
}
```

Keep the tools out of the model's tool list with `PI_REFLEX_EXPOSURE`:

- `codemode`: listed one line each in the `codemode` tool, callable from scripts.
- `deferred`: listed nowhere; `tool_search` finds and activates them on demand
  (pi ≥ 1.0 keeps deferred tools across resume/`/reload`). Requires `tool_search`:
  on `session_start` the extension checks, warns when it is missing, and activates
  `tool_search` itself if nothing else could reach the tools.
- `direct` (default): declared to the model normally; also callable from scripts.

`/reflex` shows the effective exposure, `tool_search` state, and per-tool latency telemetry (pi ≥ 1.1: call count and p50/p95 from `tool_execution_end.durationMs`).

## Classifier models (`reflex/*`)

The extension registers a `reflex` provider featuring three **local classifier models**:

- `reflex/multilingual`
- `reflex/english`
- `reflex/typed-decisions`

These models sit alongside hosted options (TypeSafe's Jev, Cloudflare's Clef; pi ≥ 1.0.1) and pi 1.1's native llama.cpp-served Laya-family classifiers. Unlike all of those, the `reflex` models run completely offline: they are private, free to use, and require no API keys, no llama-server process, and no network egress.

Your codemode scripts can access them directly through the uniform classifier interface:

```js
const reflex = await models.getModelOfType(
  "classifier",
  "reflex",
  "multilingual",
);
const r = await models.classify(reflex, {
  state: { message: "The change works, thanks." },
  questions: {
    approved: { type: "bool", instructions: "Does the user approve?" },
  },
});
return r.answers; // { approved: { type: "bool", probability: 0.97 } }
```

Other extensions can do the same via `ctx.modelRegistry.classify()`.

## Virtual model `reflex/auto` (tier routing)

If you select `reflex/auto` in `/model`, each user turn is classified locally in a single forward pass (`MODEL_ROUTER` + `INJECTION_GUARD`). Based on this classification, the request is automatically dispatched to a cheap, mid, or frontier model.

You can map these tiers to specific models using environment variables (format: `provider/model-id`):

- `PI_REFLEX_TIER_SMALL` (e.g., `anthropic/claude-haiku-4-5`)
- `PI_REFLEX_TIER_MID`
- `PI_REFLEX_TIER_FRONTIER`

**Router Behavior:**

- **Stickiness**: Continuations and retries stay on the same model used for the initial turn (so prompt caches survive).
- **Thinking Level**: Setting the virtual thinking level to `high` automatically bumps the request up one tier.
- **Fallbacks**: Any unmapped tiers will fall back to the previous physical model.
- **Resilience**: If the local engine fails, the router degrades gracefully to the mid tier instead of blocking your turn.

### Native llama.cpp interop (pi ≥ 1.1)

pi 1.1 lists llama.cpp decision models (Laya, OpenJev, Julia-1, Kev, lev) served by llama-server ≥ 0.6.0 as native classifiers (provider `llama.cpp`, `typesafe-system-one` API). When one is in the catalog, `reflex/auto` classifies each turn through the running llama-server (GPU speed, your quantization of choice) and the local ONNX engine stays as the zero-dependency fallback: native missing or failing falls back to local, local failing degrades to mid.

- **`PI_REFLEX_ROUTER_BACKEND`** (`auto` \| `native` \| `local`, default `auto`): `local` never leaves the box.
- **`PI_REFLEX_NATIVE_CLASSIFIER`**: pin a specific decision-model id (e.g. `Laya`). The pin is strict: if the pinned id is not among the listed decision models, the router stays on the local engine (visible in `/reflex`) instead of silently switching models. Without a pin, the first decision model in the catalog is used.
- Chat models served by llama.cpp also appear as classifiers (`llama-cpp-classify` API) but are **not** used: only native decision models produce the calibrated System-1 probabilities the tier thresholds were tuned on.
- `/reflex` shows the configured backend and what the last turn used, e.g. `router backend: auto · last turn: native (llama.cpp/Laya)`.

### Quantization (int8 vs fp32)

- **`int8` (default)**: ~400 MB/checkpoint, ~55 ms/question. The right pick for routing, triage, and guardrails where speed is critical.
- **`fp32`**: ~1.6 GB/checkpoint, full precision (roughly 1.3×–1.6× slower than `int8`). Set `PI_REFLEX_QUANT=fp32` in the environment **before launching pi** (read once at activation; restart to switch).
  - *When to use `fp32`:* Stick to `int8` for most workflows. Only switch to `fp32` for offline batch jobs, rigorous analytics, or edge cases where absolute confidence precision is more important than memory and latency.

Both quants cache side by side in `~/.pi-reflex/engines/`, so switching never re-downloads the other. The MCP server reads the same variable.

## Prompt-injection guard (opt-in)

Setting `PI_REFLEX_GUARD=1` enables a `context_with_system` guard. New user messages are classified once (and cached).

If a message is flagged (Probability ≥ `PI_REFLEX_GUARD_THRESHOLD`, default `0.75`), it is annotated as untrusted data before reaching the provider. 

**Guard Behavior:**
- **Non-destructive:** Flagged messages are never removed or reordered.
- **Budget-capped:** Limits interventions to ≤ 3 new messages per request.
- **Resilient:** Includes a circuit breaker if the engine fails.

## MCP server

```bash
pi mcp add reflex -- node <pkg>/bin/pi-reflex-mcp.js   # user-level; or: PI_REFLEX_MCP=1 (extension registers it)
```

Provides a zero-dependency stdio MCP server that exposes the same four tools to any MCP client.

**Configuration & Overrides:**
- **User-level (Recommended):** Allows per-project overrides (pi ≥ 1.0.1). You can toggle `enabled` or `exposure` via `.pi/mcp.json` or the `/mcp` UI command. If you provide a `command`/`url` entry, it fully replaces the server (run `/reload` after editing).
- **Session-level:** You can also register it globally via the `PI_REFLEX_MCP=1` environment variable. Note that this method is **not overridable** on a per-project basis.

## Status

`/reflex` shows engine, classifier, router, guard, and MCP status.

## Environment variables

| Variable                            | Values (default)                                                  | Purpose                                                        |
| ----------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| `PI_REFLEX_ENGINE`                  | `english` \| `multilingual` \| `typed-decisions` (`multilingual`) | default engine for tools, router, guard                        |
| `PI_REFLEX_QUANT`                   | `int8` \| `fp32` (`int8`)                                         | precision + artifact set; set before launch, restart to switch |
| `PI_REFLEX_ARTIFACTS`               | directory                                                         | local artifacts override (skips cache + download)              |
| `PI_REFLEX_TIER_SMALL/MID/FRONTIER` | `provider/model-id`                                               | `reflex/auto` tier mapping                                     |
| `PI_REFLEX_ROUTER_BACKEND`          | `auto` \| `native` \| `local` (`auto`)                            | router classification backend; `local` never uses llama.cpp    |
| `PI_REFLEX_NATIVE_CLASSIFIER`       | decision-model id (e.g. `Laya`)                                  | pin a specific llama.cpp decision model (pi ≥ 1.1)             |
| `PI_REFLEX_GUARD`                   | `1` \| `0` (`0`)                                                  | prompt-injection guard                                         |
| `PI_REFLEX_GUARD_THRESHOLD`         | 0–1 (`0.75`)                                                      | guard flag threshold                                           |
| `PI_REFLEX_EXPOSURE`                | `direct` \| `codemode` \| `deferred` (`direct`)                   | tool visibility to the model                                   |
| `PI_REFLEX_MCP`                     | `1` \| `0` (`0`)                                                  | register the MCP server for the session                        |
| `PI_REFLEX_QUIET`                   | `1` \| `0` (`0`)                                                  | silence the startup banner (`/reflex` always shows status)     |

## Artifacts

The engine resolves artifacts in the following order:

1. Local override (`$PI_REFLEX_ARTIFACTS`)
2. Local cache (`~/.pi-reflex/engines`)
3. Hugging Face download (`ngSoftware/pi-reflex-artifacts`), downloaded lazily on first use.

**Download Behavior:**

- **Resilience**: Interrupted downloads leave a `.part` file and will **resume from that byte offset** on the next attempt. Transient errors retry automatically within the call.
- **Visibility**: Download progress is shown in the pi footer.
- **Failures**: If a download fails on first use, you just need to retry; no manual cleanup is required.

**Generate Locally:**
You can also generate the artifacts locally instead of downloading them:

```bash
python tools/export_onnx.py --checkpoint convaiinnovations/laya [--subfolder <name>] --out <dir> --int8
```

_(Note: Omit `--subfolder` for the English engine, which lives at the repository root.)_

## Constraints worth knowing

- **States and criteria must be JSON-clean**: `NaN`/`Infinity` serialize as `null`
  (Python prints `NaN`), `-0` as `0` (Python prints `-0.0`).
- **Integer-like choice labels** (`"1"`, `"10"`, `"2"`) are reordered by JavaScript
  (ascending, first) unlike Python dicts. Keep them ascending; pi-reflex warns at
  runtime when it detects a diverging order.
- Reported probabilities are rounded to 4 decimals half-away-from-zero; laya uses
  banker's rounding (drift ≤ 1e-4).

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md): design, ports table, measured latencies
- [CONTRACT-harness.md](CONTRACT-harness.md): pinned consumer contract (`pi-continual-harness`) and blueprint for integrating `pi-reflex` into your own packages
- [BENCHMARKS.md](BENCHMARKS.md): latency evidence + A4 accuracy eval runner (reflex vs hosted clef)
- [RESEARCH.md](RESEARCH.md): the 53-paper research stack behind the design

## License

Apache-2.0: see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Not affiliated with TypeSafe or their "Jev" product.
