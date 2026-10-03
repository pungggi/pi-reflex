# pi-reflex

**Local System 1 decisions for [pi](https://pi.dev).** Calibrated choice / yes-no / score
decisions over any JSON-able state in a single non-autoregressive forward pass: as pi
decision tools, as local classifier models for codemode, and as a tier-routing virtual
model.

A faithful pure-TypeScript runtime for the open Laya checkpoints (≤ 5.7e-06 logits
parity, byte-identical tokenization; ~55 ms/question on CPU int8), plus a **conformal
guarantee layer**: a mathematical safety net that either guarantees the correct answer is
within a small set of options, or safely says "I don't know" instead of guessing. Neither laya nor [von](https://github.com/wfzyx/von) ship this kind of layer.

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

## Decision tools

The engine loads lazily on first use.

All tools are read-only and return both text and a typed `structuredContent` (pi `outputSchema`). To help codemode scripts fail gracefully, tools return failures as an `isError` flag with a recovery payload instead of throwing exceptions.

In pi ≥ 1.0.1, the UI renders each call as a single, compact line (e.g., `P(true)=0.42 · conf 70% · 12 tok`). This line is color-coded by confidence (success ≥ 0.5, warning = abstain, red = error), and you can press `ctrl+e` to see the full distribution. Because the UI matches tools by name, this neat formatting also applies to MCP-served twins, resumed sessions, and HTML exports.

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

`/reflex` shows the effective exposure and `tool_search` state.

## Classifier models (`reflex/*`)

The extension registers a `reflex` provider featuring three **local classifier models**:

- `reflex/multilingual`
- `reflex/english`
- `reflex/typed-decisions`

These models sit alongside hosted options like TypeSafe's Jev and Cloudflare's Clef (pi ≥ 1.0.1). However, unlike the hosted models, the `reflex` models run completely offline. They are private, free to use, and require no API keys or network egress.

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

### Quantization (int8 vs fp32)

- **`int8` (default)**: ~400 MB/checkpoint, ~55 ms/question. The right pick for
  routing, triage, and guardrails.
- **`fp32`**: ~1.6 GB/checkpoint, full precision. Set `PI_REFLEX_QUANT=fp32` in the
  environment **before launching pi** (read once at activation; restart to switch).

Both quants cache side by side in `~/.pi-reflex/engines/`, so switching never
re-downloads the other. The MCP server reads the same variable.

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
