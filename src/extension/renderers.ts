/**
 * Tool renderers (pi ≥ 1.0.1) — compact one-line display for reflex tool calls.
 *
 * Registered through `pi.registerToolRenderer()`, which resolves by NAME, so the
 * same renderers cover three cases pi could not cover before 1.0.1:
 *  - the extension's own registered tools (`reflex_decide` …),
 *  - the same tools served over MCP (`mcp__reflex__reflex_decide`), including
 *    resumed sessions and HTML exports where the call renders before the server
 *    connected (or if it never does),
 *  - reflex tools from any other source (another server name, a fork) — the
 *    matcher accepts `mcp__<server>__reflex_<verb>` for any sanitized server.
 *
 * Collapsed results are one line (`P(true)=0.42 · conf 70% · 12 tok`); ctrl+e
 * expansion adds the full probability distribution and guard detail. Confidence
 * colors follow the documented abstain rule: < 0.5 renders as a warning, ≥ 0.5
 * as success, engine errors as errors. Old session entries without a structured
 * payload fall back to the model-facing text.
 *
 * The summary helpers (`summarizeCall`/`summarizeResult`) are pure and pi-free
 * so tests and other surfaces can reuse them; only the thin `render*` wrappers
 * touch the TUI.
 */
import type { AgentToolResult, Theme, ToolRenderers } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";

export const REFLEX_TOOL_NAMES = ["reflex_decide", "reflex_judge", "reflex_rate", "reflex_route"] as const;
export type ReflexToolName = (typeof REFLEX_TOOL_NAMES)[number];

const REFLEX_TOOL_SET: ReadonlySet<string> = new Set(REFLEX_TOOL_NAMES);

/** Long user/model-supplied strings are clipped to this many characters before styling. */
const SNIPPET_MAX = 120;

/**
 * The reflex tool a pi tool name refers to, or undefined.
 * Accepts the bare names (registered extension tools) and pi's MCP form
 * `mcp__<server>__<tool>` for any sanitized server name.
 */
export function reflexToolFromName(toolName: string): ReflexToolName | undefined {
  if (REFLEX_TOOL_SET.has(toolName)) return toolName as ReflexToolName;
  const mcp = /^mcp__[A-Za-z0-9_-]+__(reflex_(?:decide|judge|rate|route))$/.exec(toolName);
  return mcp ? (mcp[1] as ReflexToolName) : undefined;
}

function snippet(value: unknown, max = SNIPPET_MAX): string {
  const s = typeof value === "string" ? value : value === undefined || value === null ? "" : String(value);
  const flat = s.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

// ── call summaries (pure) ────────────────────────────────────────────────────

export interface CallSummary {
  /** Main line: the question/instructions/message the decision is about. */
  title: string;
  /** Secondary detail (option/level counts), rendered dim. */
  detail?: string;
}

export function summarizeCall(tool: ReflexToolName, args: Record<string, unknown> | undefined): CallSummary {
  const a = args ?? {};
  switch (tool) {
    case "reflex_decide": {
      const options = a.options && typeof a.options === "object" ? Object.keys(a.options as Record<string, unknown>) : [];
      const detail = options.length ? `${options.length} option${options.length === 1 ? "" : "s"}: ${snippet(options.join(", "), 60)}` : undefined;
      return { title: snippet(a.instructions || a.state), detail };
    }
    case "reflex_judge":
      return { title: snippet(a.question) };
    case "reflex_rate":
      return { title: snippet(a.instructions), detail: `${Array.isArray(a.levels) ? a.levels.length : "?"} levels` };
    case "reflex_route":
      return { title: snippet(a.message) };
  }
}

// ── result summaries (pure) ──────────────────────────────────────────────────

export interface ResultSummary {
  /** Severity driving the theme color: success (conf ≥ 0.5), warning (abstain), or error. */
  severity: "success" | "warning" | "error";
  /** Collapsed one-line summary. */
  line: string;
  /** Extra lines shown when the result view is expanded. */
  detailLines: string[];
}

/** Tool payloads are small; probabilities beyond this many keys render as counts. */
const MAX_DIST_KEYS = 8;

function distribution(probabilities: unknown): string {
  if (!probabilities || typeof probabilities !== "object") return "";
  const entries = Object.entries(probabilities as Record<string, unknown>)
    .filter(([, v]) => typeof v === "number")
    .sort((x, y) => (y[1] as number) - (x[1] as number));
  if (!entries.length) return "";
  if (entries.length > MAX_DIST_KEYS) return `${entries.length} labels (expand for full distribution)`;
  return entries.map(([k, v]) => `P(${k})=${(v as number).toFixed(2)}`).join(" ");
}

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

/** Unwrap the structured payload: direct for extension tools, nested for MCP results (CallToolResult wrapper). */
export function unwrapStructured(result: AgentToolResult<unknown>): Record<string, unknown> | undefined {
  const sc = result.structuredContent;
  if (!sc || typeof sc !== "object" || Array.isArray(sc)) return undefined;
  const inner = (sc as Record<string, unknown>).structuredContent;
  return inner && typeof inner === "object" && !Array.isArray(inner) ? (inner as Record<string, unknown>) : (sc as Record<string, unknown>);
}

function textFallback(result: AgentToolResult<unknown>): string[] {
  const lines = (result.content ?? [])
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .flatMap((c) => c.text.split("\n"))
    .map((l) => l.trim())
    .filter(Boolean);
  return lines;
}

export function summarizeResult(tool: ReflexToolName, result: AgentToolResult<unknown>, expanded: boolean): ResultSummary {
  const data = unwrapStructured(result);

  if (result.isError || data?.type === "error") {
    const error = typeof data?.error === "string" ? data.error : textFallback(result)[0] ?? "failed";
    const detailLines = expanded ? [typeof data?.recovery === "string" ? data.recovery : ""].filter(Boolean) : [];
    return { severity: "error", line: snippet(error), detailLines };
  }

  if (!data) {
    // Old entries (or foreign reflex tools) without structured output: show the text payload.
    const lines = textFallback(result);
    return { severity: "success", line: snippet(lines[0] ?? "done"), detailLines: expanded ? lines.slice(1, 16) : [] };
  }

  const confidence = typeof data.confidence === "number" ? data.confidence : 1;
  const tokens = typeof data.inputTokens === "number" ? ` · ${data.inputTokens} tok` : "";
  const severity = confidence < 0.5 ? "warning" : "success";

  switch (tool) {
    case "reflex_judge": {
      const line = `P(true)=${(typeof data.probability === "number" ? data.probability : 0).toFixed(2)} · conf ${pct(confidence)}${tokens}`;
      return { severity, line, detailLines: expanded && confidence < 0.5 ? ["confidence < 0.5 — treat as abstain, fall back to the model"] : [] };
    }
    case "reflex_decide": {
      const line = `${String(data.choice ?? "?")} · ${pct((data.probabilities as Record<string, number> | undefined)?.[String(data.choice ?? "")] ?? 0)} · conf ${pct(confidence)}${tokens}`;
      const dist = distribution(data.probabilities);
      return { severity, line, detailLines: expanded && dist ? [dist] : [] };
    }
    case "reflex_rate": {
      const n = typeof data.levelCount === "number" ? data.levelCount - 1 : "?";
      const line = `score ${typeof data.score === "number" ? data.score.toFixed(1) : "?"}/${n} · conf ${pct(confidence)}${tokens}`;
      const dist = distribution(data.probabilities);
      return { severity, line, detailLines: expanded && dist ? [dist] : [] };
    }
    case "reflex_route": {
      const guards = (data.guards && typeof data.guards === "object" ? data.guards : {}) as Record<string, unknown>;
      const inj = typeof guards.injection === "number" ? guards.injection : 0;
      const line = `tier: ${String(data.tier ?? "?")} — ${snippet(data.reason, 80)}${inj >= 0.5 ? ` · ⚠ injection ${inj.toFixed(2)}` : ""}${tokens}`;
      const detail = [
        `injection=${inj.toFixed(2)} harmful=${(typeof guards.harmful === "number" ? guards.harmful : 0).toFixed(2)}`,
        typeof data.complexity === "string" ? `complexity=${data.complexity}` : "",
        typeof data.needsCode === "number" ? `needs_code=${data.needsCode.toFixed(2)}` : "",
        typeof data.longContext === "number" ? `long_context=${data.longContext.toFixed(2)}` : "",
      ].filter(Boolean);
      return { severity: inj >= 0.5 ? "warning" : severity, line, detailLines: expanded ? detail : [] };
    }
  }
}

// ── TUI wrappers ─────────────────────────────────────────────────────────────

function renderCallComponent(tool: ReflexToolName, args: Record<string, unknown> | undefined, theme: Theme): Text {
  const { title, detail } = summarizeCall(tool, args);
  let text = theme.fg("text", title || "…");
  if (detail) text += theme.fg("dim", ` — ${detail}`);
  return new Text(text, 0, 0);
}

function renderResultComponent(
  tool: ReflexToolName,
  result: AgentToolResult<unknown>,
  { expanded, isPartial }: { expanded: boolean; isPartial: boolean },
  theme: Theme,
): Text {
  if (isPartial) return new Text(theme.fg("warning", "deciding…"), 0, 0);
  const { severity, line, detailLines } = summarizeResult(tool, result, expanded);
  const color = severity === "error" ? "error" : severity === "warning" ? "warning" : "success";
  let text = theme.fg(color, line);
  for (const detail of detailLines) text += `\n${theme.fg("dim", detail)}`;
  return new Text(text, 0, 0);
}

/** Renderers for one reflex tool, for `registerTool()` or a `registerToolRenderer` resolver. */
export function reflexToolRenderers(tool: ReflexToolName): ToolRenderers {
  return {
    renderCall: (args, theme) => renderCallComponent(tool, args as Record<string, unknown>, theme),
    renderResult: (result, options, theme) => renderResultComponent(tool, result as AgentToolResult<unknown>, options, theme),
  };
}

/**
 * A `registerToolRenderer` resolver: authoritative for reflex tool names (bare
 * and `mcp__<server>__reflex_*`), passes everything else through untouched.
 */
export function reflexToolRendererResolver(toolName: string, next: () => ToolRenderers | undefined): ToolRenderers | undefined {
  const tool = reflexToolFromName(toolName);
  return tool ? reflexToolRenderers(tool) : next();
}
