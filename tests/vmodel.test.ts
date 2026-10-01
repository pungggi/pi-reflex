import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { SystemOneResult } from "../src/core/types.js";
import {
  bumpTier,
  latestUserText,
  parseModelRef,
  routeUserTurn,
  tierFromEnv,
  tierThinkingLevel,
} from "../src/extension/vmodel.js";

function fakeLoader(opts: { choice?: string; noul?: number }): () => Promise<Engine> {
  return async () =>
    ({
      name: "fake",
      async systemOne(_state: unknown, questions: Parameters<Engine["systemOne"]>[1]): Promise<SystemOneResult> {
        const answers: SystemOneResult["answers"] = {};
        for (const [qid, q] of Object.entries(questions)) {
          if (q.type === "choice") {
            (answers as Record<string, unknown>)[qid] = { type: "choice", choice: opts.choice ?? "trivial", probabilities: {}, confidence: 0.5, action: { act_probability: 0.5 } };
          } else {
            (answers as Record<string, unknown>)[qid] = { type: "noul", noul: opts.noul ?? 0.1, confidence: 0.8, action: { act_probability: 0.5 } };
          }
        }
        return { model: "fake", answers, usage: { input_tokens: 9, output_tokens: 0 } };
      },
      async batchQuestion() {
        return [];
      },
    }) as unknown as Engine;
}

const FAKE_MODEL = { id: "small-model", provider: "prov" } as never;

describe("helpers", () => {
  it("parseModelRef splits provider/id", () => {
    expect(parseModelRef("anthropic/claude-haiku-4-5")).toEqual({ provider: "anthropic", id: "claude-haiku-4-5" });
    expect(parseModelRef("no-slash")).toBeUndefined();
    expect(parseModelRef("/leading")).toBeUndefined();
  });
  it("tierFromEnv reads the right variable", () => {
    expect(tierFromEnv({ PI_REFLEX_TIER_SMALL: "a/b" }, "small")).toBe("a/b");
    expect(tierFromEnv({}, "frontier")).toBeUndefined();
  });
  it("bumpTier escalates only on high effort", () => {
    expect(bumpTier("small", false)).toBe("small");
    expect(bumpTier("small", true)).toBe("mid");
    expect(bumpTier("mid", true)).toBe("frontier");
    expect(bumpTier("frontier", true)).toBe("frontier");
  });
  it("tierThinkingLevel maps tier → effort", () => {
    expect(tierThinkingLevel("small")).toBe("low");
    expect(tierThinkingLevel("mid")).toBe("medium");
    expect(tierThinkingLevel("frontier")).toBe("high");
  });
  it("latestUserText finds the last user message (string and blocks)", () => {
    const msgs = [
      { role: "system", content: "sys" },
      { role: "user", content: "first" },
      { role: "assistant", content: [{ type: "text", text: "hi" }] },
      { role: "user", content: [{ type: "text", text: "hello" }, { type: "text", text: "world" }] },
    ] as never;
    expect(latestUserText(msgs)).toBe("hello\nworld");
  });
});

describe("routeUserTurn", () => {
  const deps = {
    env: {
      PI_REFLEX_TIER_SMALL: "prov/small-model",
      PI_REFLEX_TIER_MID: "prov/mid-model",
      PI_REFLEX_TIER_FRONTIER: "prov/frontier-model",
    },
    find: (provider: string, id: string) => (provider === "prov" ? ({ id, provider } as never) : undefined),
  };

  it("routes a trivial message to the small tier", async () => {
    const out = await routeUserTurn(fakeLoader({ choice: "trivial", noul: 0.1 }), "what does this flag mean?", "low", deps);
    expect(out.model).toMatchObject({ id: "small-model" });
    expect(out.thinkingLevel).toBe("low");
    expect(out.state.tier).toBe("small");
  });
  it("routes a complex message to the frontier tier", async () => {
    const out = await routeUserTurn(fakeLoader({ choice: "complex", noul: 0.1 }), "redesign the scheduler", "low", deps);
    expect(out.model).toMatchObject({ id: "frontier-model" });
    expect(out.state.tier).toBe("frontier");
  });
  it("high effort bumps one tier", async () => {
    const out = await routeUserTurn(fakeLoader({ choice: "trivial", noul: 0.1 }), "quick question", "high", deps);
    expect(out.model).toMatchObject({ id: "mid-model" });
    expect(out.thinkingLevel).toBe("medium");
  });
  it("falls back to the previous model when the tier env is unmapped", async () => {
    const out = await routeUserTurn(fakeLoader({ choice: "trivial", noul: 0.1 }), "x", "low", { env: {}, find: deps.find }, FAKE_MODEL);
    expect(out.model).toBe(FAKE_MODEL);
  });
  it("throws a setup hint when nothing resolves", async () => {
    await expect(routeUserTurn(fakeLoader({ choice: "trivial", noul: 0.1 }), "x", "low", { env: {}, find: () => undefined })).rejects.toThrow(
      /PI_REFLEX_TIER_SMALL/,
    );
  });
  it("engine failure degrades to mid tier instead of blocking the turn", async () => {
    const broken = async () => {
      throw new Error("engine down");
    };
    const out = await routeUserTurn(broken, "anything", "low", deps, FAKE_MODEL);
    expect(out.state.reason).toContain("engine unavailable");
    expect(out.model).toMatchObject({ id: "mid-model" });
  });
});
