/**
 * pi 1.0 alignment (feat/pi-1.0-alignment):
 * - PI_REFLEX_EXPOSURE accepts `deferred` next to `codemode` (unknown → default direct)
 * - startup banner is quiet by default; opt-in via PI_REFLEX_QUIET=0
 * - tool failures return isError results with structured recovery payloads
 *   (codemode scripts read structuredContent) instead of throwing
 */
import { describe, expect, it } from "vitest";
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

function activateCapturing(env: Record<string, string | undefined>, loadEngine?: ExtensionDeps["loadEngine"]) {
  const tools: CapturedTool[] = [];
  const handlers = new Map<string, (event: unknown, ctx: unknown) => Promise<unknown>>();
  const pi = {
    registerProvider: () => {},
    registerVirtualModel: () => {},
    registerTool: (def: CapturedTool) => tools.push(def),
    registerCommand: () => {},
    registerMcpServer: () => {},
    registerToolRenderer: () => {},
    on: (event: string, handler: (event: unknown, ctx: unknown) => Promise<unknown>) => handlers.set(event, handler),
  } as never;
  activate(pi, { env, loadEngine });
  return { tools, handlers };
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
});

describe("startup banner (pi quietStartup alignment)", () => {
  async function bannerCalls(env: Record<string, string | undefined>): Promise<string[]> {
    const { handlers } = activateCapturing(env);
    const calls: string[] = [];
    const handler = handlers.get("session_start");
    expect(handler).toBeDefined();
    await handler!(undefined, { ui: { notify: (m: string) => calls.push(m) } });
    return calls;
  }

  it("is quiet by default and with PI_REFLEX_QUIET=1", async () => {
    expect(await bannerCalls({})).toEqual([]);
    expect(await bannerCalls({ PI_REFLEX_QUIET: "1" })).toEqual([]);
  });

  it("is shown with PI_REFLEX_QUIET=0", async () => {
    const calls = await bannerCalls({ PI_REFLEX_QUIET: "0" });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatch(/pi-reflex ready/);
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
