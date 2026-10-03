/**
 * Tool renderers (pi ≥ 1.0.1 registerToolRenderer):
 * - name matching covers bare names and pi's MCP form mcp__<server>__reflex_<verb>
 * - the resolver is authoritative for reflex names and passes others through
 * - collapsed results are one compact line; expansion adds distributions
 * - MCP results (structuredContent = CallToolResult wrapper) unwrap correctly
 * - confidence < 0.5 (abstain) renders as warning, engine errors as error
 * - entries without structuredContent fall back to the text payload
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

function render(toolName: string, result: AgentToolResult<unknown>, expanded = false): string[] {
  const renderers = reflexToolRendererResolver(toolName, () => undefined)!;
  const comp = renderers.renderResult!(result, { expanded, isPartial: false }, plainTheme, {} as never);
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
