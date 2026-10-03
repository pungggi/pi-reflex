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
 * Payload sources, in priority order (pi never passes `structuredContent` to
 * `renderResult` — verified against pi 1.0.1: ToolExecutionComponent passes
 * `{ content, details }`, HTML export builds `{ content, details, isError }`):
 *  1. `result.structuredContent` when present and a recognized reflex payload —
 *     future-proof (a pi that passes it) plus direct programmatic callers/tests;
 *     MCP results wrap it in a `CallToolResult`, which is unwrapped. A wrapper
 *     without an inner payload (text-only MCP result) is NOT reflex data.
 *  2. `result.content[0].text` — the deterministic `cores.ts` formats shared by
 *     the extension tools and the MCP server (`P(true)=0.42 (conf 70.0%) [12 tok]`).
 *     This is what real TUI and HTML-export rendering sees.
 *  3. raw text fallback — old entries and foreign reflex tools.
 *
 * The error flag arrives on the render context (`ToolRenderContext.isError`),
 * not on the result object (TUI passes only `{content, details}`); both are
 * honored. Collapsed results are one line; ctrl+e expansion always reveals the
 * full probability distribution (never a dead "expand for more" hint). Colors
 * follow the documented abstain rule: confidence < 0.5 renders as a warning.
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

/** Payload `type` values our tools emit. Anything else is not reflex data. */
const PAYLOAD_TYPES = new Set(["choice", "bool", "score", "route", "error"]);

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

/** Collapsed views show no distribution; expanded views always show the FULL list
 *  (review #3: the old >8-label clamp printed an "expand" hint even when already expanded). */
function distribution(probabilities: unknown): string {
  if (!probabilities || typeof probabilities !== "object") return "";
  const entries = Object.entries(probabilities as Record<string, unknown>)
    .filter(([, v]) => typeof v === "number")
    .sort((x, y) => (y[1] as number) - (x[1] as number));
  if (!entries.length) return "";
  return entries.map(([k, v]) => `P(${k})=${(v as number).toFixed(2)}`).join(" ");
}

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

/**
 * Unwrap the structured payload: direct for extension tools, nested for MCP results
 * (CallToolResult wrapper). Returns undefined unless the payload carries a
 * recognized reflex `type` — a text-only MCP wrapper (`{content: [...]}` with no
 * inner payload) is not reflex data and must not be rendered as such.
 */
export function unwrapStructured(result: AgentToolResult<unknown>): Record<string, unknown> | undefined {
  const sc = result.structuredContent as Record<string, unknown> | undefined;
  if (!sc || typeof sc !== "object" || Array.isArray(sc)) return undefined;
  const inner = sc.structuredContent;
  const candidate =
    inner && typeof inner === "object" && !Array.isArray(inner) ? (inner as Record<string, unknown>) : sc;
  return typeof candidate.type === "string" && PAYLOAD_TYPES.has(candidate.type) ? candidate : undefined;
}

function textLines(result: AgentToolResult<unknown>): string[] {
  return (result.content ?? [])
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .flatMap((c) => c.text.split("\n"))
    .map((l) => l.trim())
    .filter(Boolean);
}

/** The extension's errorResult emits `pi-reflex error: <message>`; the MCP server
 *  emits raw messages without the prefix — its errors render via context.isError /
 *  result.isError instead. The prefix is the fallback signal for context-less callers. */
const ERROR_PREFIX = "pi-reflex error: ";

function num(s: string | undefined): number | undefined {
  if (s === undefined) return undefined;
  const v = Number(s);
  return Number.isFinite(v) ? v : undefined;
}

/**
 * Parse the deterministic cores.ts text formats — the only payload real TUI and
 * HTML-export rendering sees (pi passes just `{content, details}` to renderers).
 * Both the extension tools and the MCP server emit identical text.
 */
export function parseResultText(tool: ReflexToolName, text: string): Partial<Record<string, unknown>> {
  const t = text.trim();
  if (t.startsWith(ERROR_PREFIX)) return { type: "error", error: t.slice(ERROR_PREFIX.length) };

  const usage = / \[(\d+) tok\]$/.exec(t);
  const tokens = usage ? Number(usage[1]) : undefined;
  const body = usage ? t.slice(0, t.length - usage[0].length) : t;

  switch (tool) {
    case "reflex_judge": {
      // Probability may render in scientific notation at extremes (1e-7) — the
      // exponent group keeps those parsing instead of falling back to raw text.
      const m = /^P\(true\)=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) \(conf (\d+(?:\.\d+)?)%\)$/.exec(body);
      if (!m) return {};
      return { type: "bool", probability: num(m[1]), confidence: (num(m[2]) ?? 0) / 100, inputTokens: tokens };
    }
    case "reflex_decide": {
      // `<choice> (conf X%) — {json probabilities}`
      const m = /^([\s\S]+) \(conf (\d+(?:\.\d+)?)%\) — (\{.*\})$/.exec(body);
      if (!m || m[1] === undefined || m[2] === undefined || m[3] === undefined) return {};
      let probabilities: Record<string, number> | undefined;
      try {
        const parsed = JSON.parse(m[3]) as Record<string, unknown>;
        if (parsed && typeof parsed === "object") {
          probabilities = Object.fromEntries(Object.entries(parsed).filter(([, v]) => typeof v === "number")) as Record<string, number>;
        }
      } catch {
        return {};
      }
      return { type: "choice", choice: m[1].trim(), probabilities, confidence: (num(m[2]) ?? 0) / 100, inputTokens: tokens };
    }
    case "reflex_rate": {
      const m = /^score (\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\/([\d?]+) \(conf (\d+(?:\.\d+)?)%\)$/.exec(body);
      if (!m) return {};
      const max = num(m[2]);
      return { type: "score", score: num(m[1]), levelCount: max === undefined ? undefined : max + 1, confidence: (num(m[3]) ?? 0) / 100, inputTokens: tokens };
    }
    case "reflex_route": {
      // `tier: <tier> — <reason> (key=value detail)`
      const m = /^tier: (small|mid|frontier) — (.*) \(([^()]*)\)$/.exec(body);
      if (!m || m[2] === undefined || m[3] === undefined) return {};
      const detail = Object.fromEntries(
        m[3]
          .split(" ")
          .map((kv) => kv.split("=", 2))
          .filter((kv) => kv.length === 2)
          .map(([k, v]) => [k, num(v) ?? v]),
      );
      return {
        type: "route",
        tier: m[1] as string,
        reason: m[2],
        complexity: typeof detail.complexity === "string" ? detail.complexity : undefined,
        needsCode: typeof detail.needs_code === "number" ? detail.needs_code : undefined,
        longContext: typeof detail.long_context === "number" ? detail.long_context : undefined,
        guards: { injection: typeof detail.injection === "number" ? detail.injection : 0, harmful: typeof detail.harmful === "number" ? detail.harmful : 0 },
        inputTokens: tokens,
      };
    }
  }
}

function dataToSummary(tool: ReflexToolName, data: Record<string, unknown>, expanded: boolean): ResultSummary | undefined {
  const confidence = typeof data.confidence === "number" ? data.confidence : 1;
  const tokens = typeof data.inputTokens === "number" ? ` · ${data.inputTokens} tok` : "";
  const severity = confidence < 0.5 ? "warning" : "success";

  switch (tool) {
    case "reflex_judge": {
      if (data.type !== "bool" || typeof data.probability !== "number") return undefined;
      const line = `P(true)=${data.probability.toFixed(2)} · conf ${pct(confidence)}${tokens}`;
      return { severity, line, detailLines: expanded && confidence < 0.5 ? ["confidence < 0.5 — treat as abstain, fall back to the model"] : [] };
    }
    case "reflex_decide": {
      if (data.type !== "choice" || typeof data.choice !== "string") return undefined;
      const top = (data.probabilities as Record<string, number> | undefined)?.[data.choice] ?? 0;
      const line = `${data.choice} · ${pct(top)} · conf ${pct(confidence)}${tokens}`;
      const dist = expanded ? distribution(data.probabilities) : "";
      return { severity, line, detailLines: dist ? [dist] : [] };
    }
    case "reflex_rate": {
      if (data.type !== "score" || typeof data.score !== "number") return undefined;
      const n = typeof data.levelCount === "number" ? data.levelCount - 1 : "?";
      const line = `score ${data.score.toFixed(1)}/${n} · conf ${pct(confidence)}${tokens}`;
      const dist = expanded ? distribution(data.probabilities) : "";
      return { severity, line, detailLines: dist ? [dist] : [] };
    }
    case "reflex_route": {
      if (data.type !== "route" || typeof data.tier !== "string") return undefined;
      const guards = (data.guards && typeof data.guards === "object" ? data.guards : {}) as Record<string, unknown>;
      const inj = typeof guards.injection === "number" ? guards.injection : 0;
      const line = `tier: ${data.tier} — ${snippet(data.reason, 80)}${inj >= 0.5 ? ` · ⚠ injection ${inj.toFixed(2)}` : ""}${tokens}`;
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

/**
 * `isError` comes from the render context (`ToolRenderContext.isError`) — the
 * TUI result object carries no flag; HTML export sets both.
 */
export function summarizeResult(
  tool: ReflexToolName,
  result: AgentToolResult<unknown>,
  expanded: boolean,
  isError = false,
): ResultSummary {
  const lines = textLines(result);
  const structured = unwrapStructured(result);
  const parsed = lines[0] !== undefined ? parseResultText(tool, lines[0]) : {};

  if (isError || result.isError || structured?.type === "error" || parsed.type === "error") {
    const error =
      typeof structured?.error === "string"
        ? structured.error
        : typeof parsed.error === "string"
          ? parsed.error
          : (lines[0] ?? "failed");
    const recovery = typeof structured?.recovery === "string" ? structured.recovery : "engine errors usually mean missing artifacts — see /reflex";
    return { severity: "error", line: snippet(error), detailLines: expanded ? [recovery] : [] };
  }

  // 1) structured payload when pi provides one; 2) the deterministic text format;
  // 3) raw text fallback (old entries, foreign reflex tools).
  const fromStructured = structured ? dataToSummary(tool, structured, expanded) : undefined;
  if (fromStructured) return fromStructured;
  if (lines[0] !== undefined) {
    const fromText = Object.keys(parsed).length ? dataToSummary(tool, parsed as Record<string, unknown>, expanded) : undefined;
    if (fromText) return fromText;
    return { severity: "success", line: snippet(lines[0]), detailLines: expanded ? lines.slice(1, 16) : [] };
  }
  return { severity: "success", line: "done", detailLines: [] };
}

// ── TUI wrappers ─────────────────────────────────────────────────────────────

/** The subset of ToolRenderContext the renderers need (the error flag lives here, not on the result). */
interface RenderContext {
  isError?: boolean;
}

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
  context?: RenderContext,
): Text {
  if (isPartial) return new Text(theme.fg("warning", "deciding…"), 0, 0);
  const { severity, line, detailLines } = summarizeResult(tool, result, expanded, context?.isError ?? false);
  const color = severity === "error" ? "error" : severity === "warning" ? "warning" : "success";
  let text = theme.fg(color, line);
  for (const detail of detailLines) text += `\n${theme.fg("dim", detail)}`;
  return new Text(text, 0, 0);
}

/** Renderers for one reflex tool, for `registerTool()` or a `registerToolRenderer` resolver. */
export function reflexToolRenderers(tool: ReflexToolName): ToolRenderers {
  return {
    renderCall: (args, theme) => renderCallComponent(tool, args as Record<string, unknown>, theme),
    renderResult: (result, options, theme, context) =>
      renderResultComponent(tool, result as AgentToolResult<unknown>, options, theme, context as RenderContext | undefined),
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
