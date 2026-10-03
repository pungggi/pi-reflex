/**
 * pi 1.0 alignment (feat/pi-1.0-alignment):
 * - PI_REFLEX_EXPOSURE accepts `deferred` next to `codemode` (unknown → default direct)
 * - startup banner is on by default; PI_REFLEX_QUIET=1 silences it
 * - `deferred` exposure enforces its own contract on session_start: without an
 *   active tool_search it warns — and when nothing could reach the tools at all
 *   (no codemode either), it activates tool_search if the host registered it
 * - tool failures return isError results with structured recovery payloads
 *   (codemode scripts read structuredContent) instead of throwing
 */
import { describe, expect, it, vi } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { SystemOneResult } from "../src/core/types.js";
import type { EngineName } from "../src/engine/download.js";
import activate, { type ExtensionDeps } from "../src/extension/index.js";

function fakeEngine(): Engine {
  return {
    name: "fake",
    async systemOne(_state: unknown, questions: Parameters<Engine["systemOne"]>[1]): Promise<SystemOneResult> {
      const answers: SystemOneResult["answers"] = {};
      for (const qid of Object.keys(questions)) {
        (answers as Record<string, unknown>)[qid] = { type: "noul", noul: 0.5, confidence: 0.8, action: { act_probability: 0.5 } };
      }
      return { model: "fake", answers, usage: { input_tokens: 1, output_tokens: 0 } };
    },
    async batchQuestion() {
      return [];
    },
  } as unknown as Engine;
}

interface CapturedTool {
  name: string;
  exposure?: string;
  execute: (id: string, params: Record<string, unknown>) => Promise<{
    content: { type: string; text: string }[];
    structuredContent?: Record<string, unknown>;
    isError?: boolean;
  }>;
}

interface Notification {
  message: string;
  level: string;
}

interface ActivateOpts {
  /** Tools declared to the model (pi.getActiveTools). */
  activeTools?: string[];
  /** All registered tools (pi.getAllTools); defaults to just tool_search. */
  allToolNames?: string[] | null;
  /** Simulate a pi ≥ 0.99 host without the tool introspection APIs. */
  noIntrospection?: boolean;
}

function activateCapturing(env: Record<string, string | undefined>, loadEngine?: ExtensionDeps["loadEngine"], opts: ActivateOpts = {}) {
  const tools: CapturedTool[] = [];
  const handlers = new Map<string, (event: unknown, ctx: unknown) => Promise<unknown>>();
  const activeTools = opts.activeTools ?? [];
  const allToolNames = opts.allToolNames === null ? [] : (opts.allToolNames ?? ["tool_search"]);
  const setActiveCalls: string[][] = [];
  const pi: Record<string, unknown> = {
    registerProvider: () => {},
    registerVirtualModel: () => {},
    registerTool: (def: CapturedTool) => tools.push(def),
    registerCommand: () => {},
    registerMcpServer: () => {},
    registerToolRenderer: () => {},
    on: (event: string, handler: (event: unknown, ctx: unknown) => Promise<unknown>) => handlers.set(event, handler),
  };
  if (!opts.noIntrospection) {
    pi.getActiveTools = () => [...activeTools];
    pi.getAllTools = () => allToolNames.map((name) => ({ name }));
    pi.setActiveTools = (names: string[]) => setActiveCalls.push([...names]);
  }
  activate(pi as never, { env, loadEngine });
  return { tools, handlers, setActiveCalls };
}

describe("PI_REFLEX_EXPOSURE (pi ≥ 1.0 deferred exposure)", () => {
  it.each(["codemode", "deferred"] as const)("accepts %s for all four tools", (value) => {
    const { tools } = activateCapturing({ PI_REFLEX_EXPOSURE: value });
    expect(tools.map((t) => t.name)).toEqual(["reflex_decide", "reflex_judge", "reflex_rate", "reflex_route"]);
    expect(tools.map((t) => t.exposure)).toEqual([value, value, value, value]);
  });

  it("defaults to direct (undefined) and rejects unknown values", () => {
    expect(activateCapturing({}).tools.every((t) => t.exposure === undefined)).toBe(true);
    expect(activateCapturing({ PI_REFLEX_EXPOSURE: "hidden" }).tools.every((t) => t.exposure === undefined)).toBe(true);
  });

  it("warns on unrecognized values (stderr); direct is the silent explicit no-op", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      activateCapturing({ PI_REFLEX_EXPOSURE: "hidden" });
      expect(err).toHaveBeenCalledTimes(1);
      expect(String(err.mock.calls[0][0])).toMatch(/PI_REFLEX_EXPOSURE='hidden'.*codemode \| deferred \| direct/);
      err.mockClear();
      activateCapturing({ PI_REFLEX_EXPOSURE: "model-only" });
      expect(err).toHaveBeenCalledTimes(1); // any unsupported value warns, not just 'hidden'
      err.mockClear();
      activateCapturing({ PI_REFLEX_EXPOSURE: "direct" });
      activateCapturing({ PI_REFLEX_EXPOSURE: "codemode" });
      activateCapturing({ PI_REFLEX_EXPOSURE: "deferred" });
      expect(err).not.toHaveBeenCalled(); // recognized values stay silent
    } finally {
      err.mockRestore();
    }
  });
});

describe("startup banner (on by default; PI_REFLEX_QUIET=1 silences)", () => {
  async function sessionNotifications(env: Record<string, string | undefined>, opts: ActivateOpts = {}): Promise<Notification[]> {
    const { handlers } = activateCapturing(env, undefined, opts);
    const calls: Notification[] = [];
    const handler = handlers.get("session_start");
    expect(handler).toBeDefined();
    await handler!(undefined, { ui: { notify: (message: string, level = "info") => calls.push({ message, level }) } });
    return calls;
  }

  it("is shown by default and with PI_REFLEX_QUIET=0", async () => {
    for (const env of [{}, { PI_REFLEX_QUIET: "0" }]) {
      const calls = await sessionNotifications(env);
      expect(calls).toHaveLength(1);
      expect(calls[0]!.message).toMatch(/pi-reflex ready/);
      expect(calls[0]!.level).toBe("info");
    }
  });

  it("is silenced with PI_REFLEX_QUIET=1", async () => {
    expect(await sessionNotifications({ PI_REFLEX_QUIET: "1" })).toEqual([]);
  });
});

describe("deferred exposure without tool_search (session_start guard)", () => {
  const QUIET = { PI_REFLEX_EXPOSURE: "deferred", PI_REFLEX_QUIET: "1" };

  it("no warning when tool_search is active", async () => {
    const { handlers, setActiveCalls } = activateCapturing({ ...QUIET }, undefined, { activeTools: ["tool_search"] });
    const calls: Notification[] = [];
    await handlers.get("session_start")!(undefined, { ui: { notify: (m: string, l: string) => calls.push({ message: m, level: l }) } });
    expect(calls).toEqual([]);
    expect(setActiveCalls).toEqual([]);
  });

  it("warns (no self-heal) when codemode is active but tool_search is not", async () => {
    const { handlers, setActiveCalls } = activateCapturing({ ...QUIET }, undefined, { activeTools: ["codemode"] });
    const calls: Notification[] = [];
    await handlers.get("session_start")!(undefined, { ui: { notify: (m: string, l: string) => calls.push({ message: m, level: l }) } });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.level).toBe("warning");
    expect(calls[0]!.message).toMatch(/already know their names/);
    expect(calls[0]!.message).toMatch(/\+tool_search/);
    expect(setActiveCalls).toEqual([]);
  });

  it("activates tool_search when nothing could reach the tools", async () => {
    const { handlers, setActiveCalls } = activateCapturing({ ...QUIET });
    const calls: Notification[] = [];
    await handlers.get("session_start")!(undefined, { ui: { notify: (m: string, l: string) => calls.push({ message: m, level: l }) } });
    expect(setActiveCalls).toEqual([["tool_search"]]);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.level).toBe("warning");
    expect(calls[0]!.message).toMatch(/unreachable this session/);
    expect(calls[0]!.message).toMatch(/Activated tool_search/);
  });

  it("warns without self-heal when the tool_search builtin is disabled", async () => {
    const { handlers, setActiveCalls } = activateCapturing({ ...QUIET }, undefined, { allToolNames: null });
    const calls: Notification[] = [];
    await handlers.get("session_start")!(undefined, { ui: { notify: (m: string, l: string) => calls.push({ message: m, level: l }) } });
    expect(setActiveCalls).toEqual([]);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.level).toBe("warning");
    expect(calls[0]!.message).toMatch(/unreachable this session/);
  });

  it("stays silent without deferred exposure (default direct)", async () => {
    const { handlers, setActiveCalls } = activateCapturing({ PI_REFLEX_QUIET: "1" }, undefined, {});
    const calls: Notification[] = [];
    await handlers.get("session_start")!(undefined, { ui: { notify: (m: string, l: string) => calls.push({ message: m, level: l }) } });
    expect(calls).toEqual([]);
    expect(setActiveCalls).toEqual([]);
  });

  it("skips the check on hosts without tool introspection (pi ≥ 0.99)", async () => {
    const { handlers } = activateCapturing({ ...QUIET }, undefined, { noIntrospection: true });
    const calls: Notification[] = [];
    await expect(
      handlers.get("session_start")!(undefined, { ui: { notify: (m: string, l: string) => calls.push({ message: m, level: l }) } }),
    ).resolves.toBeUndefined();
    expect(calls).toEqual([]);
  });
});

describe("isError results with structured recovery (pi ≥ 1.0 codemode contract)", () => {
  it("engine failure returns isError + { type: 'error', recovery } instead of throwing", async () => {
    const { tools } = activateCapturing({}, (async (_name: EngineName) => {
      throw new Error("no artifacts");
    }) as ExtensionDeps["loadEngine"]);
    const judge = tools.find((t) => t.name === "reflex_judge")!;
    const res = await judge.execute("id", { state: "x", question: "y?" });
    expect(res.isError).toBe(true);
    expect(res.structuredContent).toMatchObject({ type: "error" });
    expect(String(res.structuredContent?.error)).toMatch(/no artifacts/);
    expect(String(res.structuredContent?.recovery)).toMatch(/PI_REFLEX_ARTIFACTS/);
    expect(res.content[0]?.text).toContain("no artifacts");
  });

  it("malformed params surface as isError results (rate < 2 levels)", async () => {
    const { tools } = activateCapturing({}, async () => fakeEngine());
    const rate = tools.find((t) => t.name === "reflex_rate")!;
    const res = await rate.execute("id", { state: "x", instructions: "y", levels: ["only"] });
    expect(res.isError).toBe(true);
    expect(res.structuredContent).toMatchObject({ type: "error", error: expect.stringMatching(/at least 2 levels/) });
  });

  it("success path still returns structuredContent without isError", async () => {
    const { tools } = activateCapturing({}, async () => fakeEngine());
    const judge = tools.find((t) => t.name === "reflex_judge")!;
    const res = await judge.execute("id", { state: "x", question: "y?" });
    expect(res.isError).toBeUndefined();
    expect(res.structuredContent).toMatchObject({ type: "bool", probability: 0.5 });
  });
});
