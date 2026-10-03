import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { SystemOneResult } from "../src/core/types.js";
import type { EngineName } from "../src/engine/download.js";
import activate, { type ExtensionDeps } from "../src/extension/index.js";
import type { ProviderConfig } from "@earendil-works/pi-coding-agent";

/** Minimal fake engine answering noul=0.5 for everything. */
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

interface Captured {
  provider?: { name: string; config: ProviderConfig };
}

function activateWith(deps: ExtensionDeps): Captured {
  const captured: Captured = {};
  const pi = {
    registerProvider: (name: string, config: ProviderConfig) => (captured.provider = { name, config }),
    registerVirtualModel: () => {},
    registerTool: () => {},
    registerCommand: () => {},
    registerMcpServer: () => {},
    registerToolRenderer: () => {},
    on: () => {},
  } as never;
  activate(pi, deps);
  return captured;
}

const CONTEXT = { state: {}, questions: { q: { type: "bool" as const, instructions: "?" } } };

function classify(captured: Captured, modelId: string) {
  const impl = captured.provider?.config?.classifiers?.["reflex-onnx"];
  if (!impl) throw new Error("classifier impl not registered");
  return impl.classify({ id: modelId } as never, CONTEXT as never);
}

describe("engine slot (PR#2 reviews #1 and #6)", () => {
  it("PR#2 #1: a failed engine load does not poison other engines", async () => {
    const captured = activateWith({
      env: {},
      loadEngine: async (name: EngineName) => {
        if (name === "english") throw new Error("english artifacts missing");
        return fakeEngine();
      },
    });
    await expect(classify(captured, "english")).rejects.toThrow(/english/);
    // multilingual still loads and answers — no shared error gate.
    const res = await classify(captured, "multilingual");
    expect(res.answers.q).toEqual({ type: "bool", probability: 0.5 });
    // and english keeps failing (per-name, not global).
    await expect(classify(captured, "english")).rejects.toThrow(/english/);
  });

  it("PR#2 #6: concurrent cold loads coalesce into one engine load", async () => {
    let loads = 0;
    const captured = activateWith({
      env: {},
      loadEngine: async () => {
        loads++;
        await new Promise((r) => setTimeout(r, 20));
        return fakeEngine();
      },
    });
    const [a, b] = await Promise.all([classify(captured, "multilingual"), classify(captured, "multilingual")]);
    expect(loads).toBe(1);
    expect(a.answers.q).toEqual({ type: "bool", probability: 0.5 });
    expect(b.answers.q).toEqual({ type: "bool", probability: 0.5 });
  });

  it("PR#2 #6: a failed shared load resets so a later call retries", async () => {
    let fail = true;
    let loads = 0;
    const captured = activateWith({
      env: {},
      loadEngine: async () => {
        loads++;
        if (fail) throw new Error("transient");
        return fakeEngine();
      },
    });
    await expect(classify(captured, "multilingual")).rejects.toThrow(/transient/);
    fail = false;
    const res = await classify(captured, "multilingual");
    expect(loads).toBe(2);
    expect(res.answers.q).toEqual({ type: "bool", probability: 0.5 });
  });
});
