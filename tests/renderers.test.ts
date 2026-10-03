/**
 * Tool renderers (pi ≥ 1.0.1 registerToolRenderer):
 * - name matching covers bare names and pi's MCP form mcp__<server>__reflex_<verb>
 * - the resolver is authoritative for reflex names and passes others through
 * - pi passes only {content, details} to renderResult (structuredContent never
 *   arrives) — the deterministic cores.ts text format is parsed as the primary
 *   payload for real TUI / HTML-export rendering
 * - the error flag comes from ToolRenderContext.isError, not the result object
 * - MCP results (structuredContent = CallToolResult wrapper) unwrap correctly;
 *   a text-only wrapper is not reflex data
 * - collapsed results are one compact line; expansion ALWAYS shows the full
 *   distribution (never a dead "expand" hint)
 * - confidence < 0.5 (abstain) renders as warning, engine errors as error
 * - entries without any recognizable payload fall back to the text payload
 */
import { describe, expect, it } from "vitest";
import type { AgentToolResult, Theme, ToolRenderers } from "@earendil-works/pi-coding-agent";
import activate from "../src/extension/index.js";
import { reflexToolFromName, reflexToolRendererResolver, summarizeResult, unwrapStructured } from "../src/extension/renderers.js";

/** Theme that applies no ANSI — summary tests assert on plain text. */
const plainTheme = { fg: (_c: string, s: string) => s, bold: (s: string) => s } as unknown as Theme;

function resultOf(partial: Partial<AgentToolResult<unknown>>): AgentToolResult<unknown> {
  return { content: [], details: {}, ...partial } as AgentToolResult<unknown>;
}

function render(toolName: string, result: AgentToolResult<unknown>, expanded = false, context?: { isError?: boolean }): string[] {
  const renderers = reflexToolRendererResolver(toolName, () => undefined)!;
  const comp = renderers.renderResult!(result, { expanded, isPartial: false }, plainTheme, (context ?? {}) as never);
  // Text pads every line to the terminal width — trim for assertions.
  return (comp as { render: (w: number) => string[] }).render(200).map((l) => l.trimEnd());
}

describe("reflexToolFromName", () => {
  it("matches the four bare tool names", () => {
    expect(reflexToolFromName("reflex_decide")).toBe("reflex_decide");
    expect(reflexToolFromName("reflex_judge")).toBe("reflex_judge");
    expect(reflexToolFromName("reflex_rate")).toBe("reflex_rate");
    expect(reflexToolFromName("reflex_route")).toBe("reflex_route");
  });

  it("matches pi's MCP tool names mcp__<server>__<tool>", () => {
    expect(reflexToolFromName("mcp__reflex__reflex_decide")).toBe("reflex_decide");
    expect(reflexToolFromName("mcp__reflex__reflex_judge")).toBe("reflex_judge");
    expect(reflexToolFromName("mcp__other-server__reflex_route")).toBe("reflex_route"); // any server name
    expect(reflexToolFromName("mcp__my_repo_2__reflex_rate")).toBe("reflex_rate"); // sanitized names
  });

  it("rejects non-reflex and malformed names", () => {
    expect(reflexToolFromName("read")).toBeUndefined();
    expect(reflexToolFromName("bash")).toBeUndefined();
    expect(reflexToolFromName("mcp__reflex__other_tool")).toBeUndefined();
    expect(reflexToolFromName("mcp__reflex__reflex_decide_extra")).toBeUndefined();
    expect(reflexToolFromName("reflex_decide_extra")).toBeUndefined();
  });
});

describe("resolver behavior", () => {
  it("is authoritative for reflex names (ignores next())", () => {
    const sentinel = { renderCall: () => null, renderResult: () => null } as unknown as ToolRenderers;
    expect(reflexToolRendererResolver("reflex_judge", () => sentinel)).not.toBe(sentinel);
    expect(reflexToolRendererResolver("mcp__reflex__reflex_judge", () => sentinel)).not.toBe(sentinel);
  });

  it("passes non-reflex names through to the remaining resolvers", () => {
    const sentinel = { renderCall: () => null, renderResult: () => null } as unknown as ToolRenderers;
    expect(reflexToolRendererResolver("read", () => sentinel)).toBe(sentinel);
    expect(reflexToolRendererResolver("read", () => undefined)).toBeUndefined();
  });

  it("extension registers exactly one resolver via pi.registerToolRenderer", () => {
    const resolvers: unknown[] = [];
    const pi = {
      registerProvider: () => {},
      registerVirtualModel: () => {},
      registerTool: () => {},
      registerCommand: () => {},
      registerMcpServer: () => {},
      registerToolRenderer: (r: unknown) => resolvers.push(r),
      on: () => {},
    } as never;
    activate(pi, { env: {} as Record<string, string | undefined> });
    expect(resolvers).toHaveLength(1);
    expect(resolvers[0]).toBe(reflexToolRendererResolver);
  });
});

describe("unwrapStructured (extension vs MCP results)", () => {
  it("returns the payload directly for extension tools", () => {
    const r = resultOf({ structuredContent: { type: "bool", probability: 0.42 } });
    expect(unwrapStructured(r)).toEqual({ type: "bool", probability: 0.42 });
  });

  it("unwraps the CallToolResult wrapper MCP tools produce", () => {
    const r = resultOf({
      structuredContent: { content: [{ type: "text", text: "P(true)=0.42" }], structuredContent: { type: "bool", probability: 0.42 }, isError: false },
    });
    expect(unwrapStructured(r)).toEqual({ type: "bool", probability: 0.42 });
  });

  it("returns undefined without a structured payload", () => {
    expect(unwrapStructured(resultOf({}))).toBeUndefined();
  });
});

describe("collapsed result lines", () => {
  it("judge: P(true), confidence, tokens on one line", () => {
    const r = resultOf({ structuredContent: { type: "bool", probability: 0.42, confidence: 0.7, inputTokens: 12 } });
    expect(render("reflex_judge", r)).toEqual(["P(true)=0.42 · conf 70% · 12 tok"]);
  });

  it("decide: choice, top probability, confidence", () => {
    const r = resultOf({
      structuredContent: { type: "choice", choice: "frontend", probabilities: { frontend: 0.91, backend: 0.07 }, confidence: 0.84, inputTokens: 20 },
    });
    expect(render("reflex_decide", r)).toEqual(["frontend · 91% · conf 84% · 20 tok"]);
  });

  it("rate: score over N with confidence", () => {
    const r = resultOf({
      structuredContent: { type: "score", score: 3.4, levelCount: 5, probabilities: { "0": 0.05, "1": 0.1, "2": 0.2, "3": 0.4, "4": 0.25 }, confidence: 0.8, inputTokens: 30 },
    });
    expect(render("reflex_rate", r)).toEqual(["score 3.4/4 · conf 80% · 30 tok"]);
  });

  it("route: tier and reason, injection warning only when flagged", () => {
    const base = { type: "route", tier: "mid", reason: "balanced complexity", guards: { injection: 0.02, harmful: 0.01 }, inputTokens: 8 };
    expect(render("reflex_route", resultOf({ structuredContent: base }))).toEqual(["tier: mid — balanced complexity · 8 tok"]);
    const flagged = { ...base, guards: { injection: 0.81, harmful: 0.01 } };
    expect(render("reflex_route", resultOf({ structuredContent: flagged }))[0]).toMatch(/⚠ injection 0\.81/);
  });

  it("expanded decide adds the probability distribution", () => {
    const r = resultOf({
      structuredContent: { type: "choice", choice: "frontend", probabilities: { frontend: 0.91, backend: 0.07 }, confidence: 0.84, inputTokens: 20 },
    });
    const lines = render("reflex_decide", r, true);
    expect(lines[1]).toBe("P(frontend)=0.91 P(backend)=0.07");
  });
});

describe("severity (abstain and error)", () => {
  it("confidence < 0.5 is flagged as abstain with guidance when expanded", () => {
    const summary = summarizeResult("reflex_judge", resultOf({ structuredContent: { type: "bool", probability: 0.5, confidence: 0.3, inputTokens: 5 } }), true);
    expect(summary.severity).toBe("warning");
    expect(summary.detailLines[0]).toMatch(/abstain/);
  });

  it("isError results render the error message and recovery hint", () => {
    const r = resultOf({
      isError: true,
      structuredContent: { type: "error", error: "engine unavailable", recovery: "set PI_REFLEX_ARTIFACTS" },
    });
    const lines = render("reflex_judge", r, true);
    expect(lines[0]).toBe("engine unavailable");
    expect(lines[1]).toBe("set PI_REFLEX_ARTIFACTS");
  });

  it("falls back to text content without structuredContent (old session entries)", () => {
    const r = resultOf({ content: [{ type: "text", text: "frontend (conf 91.0%) [20 tok]" }] });
    expect(render("reflex_decide", r)).toEqual(["frontend (conf 91.0%) [20 tok]"]);
  });
});

describe("renderCall", () => {
  it("shows the question for judge and dims extra detail for decide", () => {
    const renderers = reflexToolRendererResolver("reflex_judge", () => undefined)!;
    const call = (renderers.renderCall as (a: unknown, t: Theme) => { render: (w: number) => string[] })({ question: "Is prod down?" }, plainTheme);
    expect(call.render(200).map((l) => l.trimEnd())).toEqual(["Is prod down?"]);

    const decide = reflexToolRendererResolver("reflex_decide", () => undefined)!;
    const call2 = (decide.renderCall as (a: unknown, t: Theme) => { render: (w: number) => string[] })(
      { state: "s", instructions: "Fix approach?", options: { a: "first", b: "second" } },
      plainTheme,
    );
    expect(call2.render(200).map((l) => l.trimEnd())).toEqual(["Fix approach? — 2 options: a, b"]);
  });
});

describe("review fix: pi's real renderResult contract (PR #15)", () => {
  // pi 1.0.1 ToolExecutionComponent passes only {content, details}; structuredContent
  // never arrives (HTML export builds {content, details, isError}). The deterministic
  // cores.ts text format is therefore the payload real rendering sees.
  it("#1 renders compact lines from the cores text format with no structuredContent", () => {
    const judge = resultOf({ content: [{ type: "text", text: "P(true)=0.42 (conf 70.0%) [12 tok]" }] });
    expect(render("reflex_judge", judge)).toEqual(["P(true)=0.42 · conf 70% · 12 tok"]);

    const decide = resultOf({
      content: [{ type: "text", text: 'frontend (conf 84.0%) — {"frontend":0.91,"backend":0.07} [20 tok]' }],
    });
    expect(render("reflex_decide", decide)).toEqual(["frontend · 91% · conf 84% · 20 tok"]);
    const decideExpanded = render("reflex_decide", decide, true);
    expect(decideExpanded[1]).toBe("P(frontend)=0.91 P(backend)=0.07");

    const rate = resultOf({ content: [{ type: "text", text: "score 3.4/4 (conf 80.0%) [30 tok]" }] });
    expect(render("reflex_rate", rate)).toEqual(["score 3.4/4 · conf 80% · 30 tok"]);

    const route = resultOf({
      content: [{ type: "text", text: "tier: mid — balanced complexity (complexity=balanced needs_code=0.80 long_context=0.10 injection=0.02 harmful=0.01)" }],
    });
    expect(render("reflex_route", route)).toEqual(["tier: mid — balanced complexity"]);
    const routeExpanded = render("reflex_route", route, true);
    expect(routeExpanded[1]).toBe("injection=0.02 harmful=0.01");
    expect(routeExpanded[2]).toBe("complexity=balanced");
    expect(routeExpanded[3]).toBe("needs_code=0.80");
    expect(routeExpanded[4]).toBe("long_context=0.10");
  });

  it("#2 error flag arrives on the render context, not the result (TUI shape)", () => {
    // Exactly what pi passes in the interactive TUI: no isError on the result object.
    const r = resultOf({ content: [{ type: "text", text: "pi-reflex error: engine unavailable: no artifacts" }] });
    const lines = render("reflex_judge", r, true, { isError: true });
    expect(lines[0]).toBe("engine unavailable: no artifacts");
    expect(lines[1]).toMatch(/artifacts/);
    // Without the context flag the same text still parses as an error via the cores prefix.
    expect(render("reflex_judge", r)[0]).toBe("engine unavailable: no artifacts");
  });

  it("#3 expansion always reveals the full distribution, even beyond 8 labels", () => {
    const probabilities = Object.fromEntries(Array.from({ length: 9 }, (_, i) => [`opt${i}`, 0.11]));
    const r = resultOf({
      structuredContent: { type: "choice", choice: "opt4", probabilities, confidence: 0.9, inputTokens: 44 },
    });
    const collapsed = render("reflex_decide", r);
    expect(collapsed).toHaveLength(1); // collapsed shows no distribution hint at all
    const expanded = render("reflex_decide", r, true);
    expect(expanded[1]).toMatch(/^P\(opt\d\)=0\.11( P\(opt\d\)=0\.11){8}$/);
    expect(expanded[1]).not.toContain("expand");
  });

  it("#4 a text-only MCP CallToolResult wrapper is not reflex data", () => {
    // MCP results store the whole CallToolResult in structuredContent; a text-only
    // result has no inner payload — rendering it as reflex data fabricated numbers.
    const text = 'frontend (conf 91.0%) — {"frontend":0.91} [20 tok]';
    const r = resultOf({
      content: [{ type: "text", text }],
      structuredContent: { content: [{ type: "text", text }], isError: false },
    });
    expect(unwrapStructured(r)).toBeUndefined();
    const lines = render("reflex_decide", r);
    expect(lines[0]).toBe("frontend · 91% · conf 91% · 20 tok");
    expect(lines[0]).not.toContain("P(true)=0.00");
  });
});

describe("follow-up fixes: route text format + scientific notation", () => {
  it("route text now carries [N tok]; the token count renders in real TUI rendering", () => {
    const r = resultOf({
      content: [{ type: "text", text: "tier: mid — balanced complexity (complexity=balanced needs_code=0.80 long_context=0.10 injection=0.02 harmful=0.01) [26 tok]" }],
    });
    expect(render("reflex_route", r)).toEqual(["tier: mid — balanced complexity · 26 tok"]);
  });

  it("route detail omits undefined signals instead of printing complexity=undefined", () => {
    const r = resultOf({
      content: [{ type: "text", text: "tier: small — trivial request (needs_code=0.12 injection=0.61 harmful=0.03) [12 tok]" }],
    });
    const expanded = render("reflex_route", r, true);
    expect(expanded[0]).toBe("tier: small — trivial request · ⚠ injection 0.61 · 12 tok");
    expect(expanded.join("\n")).not.toContain("complexity=");
    expect(expanded[1]).toBe("injection=0.61 harmful=0.03");
    expect(expanded[2]).toBe("needs_code=0.12");
    expect(expanded).toHaveLength(3); // no complexity / long_context lines — they were undefined
  });

  it("scientific-notation probabilities still parse (P(true)=1e-7)", () => {
    const r = resultOf({ content: [{ type: "text", text: "P(true)=1e-7 (conf 99.9%) [5 tok]" }] });
    expect(render("reflex_judge", r)).toEqual(["P(true)=0.00 · conf 100% · 5 tok"]);
    const score = resultOf({ content: [{ type: "text", text: "score 1.5e-8/3 (conf 40.0%) [9 tok]" }] });
    expect(render("reflex_rate", score)).toEqual(["score 0.0/3 · conf 40% · 9 tok"]);
  });
});
