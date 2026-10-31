/**
 * pi 1.1 native llama.cpp interop (feat/llamacpp-interop):
 * - backend selection from PI_REFLEX_ROUTER_BACKEND (auto|native|local, invalid → auto)
 * - findNativeClassifier: only provider "llama.cpp" + api "typesafe-system-one"
 *   (chat-as-classifier entries with llama-cpp-classify are NOT decision models),
 *   preferred id via PI_REFLEX_NATIVE_CLASSIFIER, first entry otherwise
 * - routeViaNative: maps choice+bool answers into the tier rule; throws on
 *   stopReason !== "stop" and on incomplete answers
 * - routeUserTurn: auto prefers native, falls back to local on native absence
 *   AND on native failure; PI_REFLEX_ROUTER_BACKEND=local never tries native;
 *   the result records which backend answered
 */
import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { SystemOneResult } from "../src/core/types.js";
import type { ClassifierApi, ClassifierContext, ClassifierModel, ClassifierResult } from "@earendil-works/pi-ai";
import {
  findNativeClassifier,
  NATIVE_CLASSIFIER_API,
  NATIVE_PROVIDER_ID,
  routeViaNative,
  routerBackendFromEnv,
  routerQuestions,
  type NativeClassifierEntry,
} from "../src/extension/native.js";
import { routeUserTurn } from "../src/extension/vmodel.js";

// ── fixtures ─────────────────────────────────────────────────────────────────

function fakeLoader(choice = "trivial", noul = 0.1): () => Promise<Engine> {
  return async () =>
    ({
      name: "fake",
      async systemOne(_state: unknown, questions: Parameters<Engine["systemOne"]>[1]): Promise<SystemOneResult> {
        const answers: SystemOneResult["answers"] = {};
        for (const [qid, q] of Object.entries(questions)) {
          if (q.type === "choice") {
            (answers as Record<string, unknown>)[qid] = { type: "choice", choice, probabilities: {}, confidence: 0.9, action: { act_probability: 0.9 } };
          } else {
            (answers as Record<string, unknown>)[qid] = { type: "noul", noul, confidence: 0.8, action: { act_probability: 0.8 } };
          }
        }
        return { model: "fake", answers, usage: { input_tokens: 9, output_tokens: 0 } };
      },
      async batchQuestion() {
        return [];
      },
    }) as unknown as Engine;
}

const NATIVE_LAYA = { provider: NATIVE_PROVIDER_ID, id: "Laya", api: NATIVE_CLASSIFIER_API };
const NATIVE_JEV = { provider: NATIVE_PROVIDER_ID, id: "OpenJev", api: NATIVE_CLASSIFIER_API };
const CHAT_AS_CLASSIFIER = { provider: NATIVE_PROVIDER_ID, id: "qwen3-4b", api: "llama-cpp-classify" as ClassifierApi };
const REFLEX_MODEL = { provider: "reflex", id: "multilingual", api: "reflex-onnx" as ClassifierApi };

function nativeResult(over: Partial<Record<string, { type: "choice"; choice: string } | { type: "bool"; probability: number }>> = {}): ClassifierResult {
  const answers = {
    complexity: { type: "choice", choice: "trivial" },
    needs_code: { type: "bool", probability: 0.2 },
    long_context: { type: "bool", probability: 0.1 },
    injection: { type: "bool", probability: 0.02 },
    harmful: { type: "bool", probability: 0.01 },
    ...over,
  };
  return {
    api: NATIVE_CLASSIFIER_API,
    provider: NATIVE_PROVIDER_ID,
    model: "Laya",
    answers: answers as unknown as ClassifierResult["answers"],
    stopReason: "stop",
    timestamp: 0,
  };
}

/** native deps with a recording mock; `fail` makes every call return stopReason error. */
function nativeDeps(
  models: readonly NativeClassifierEntry[],
  result: () => ClassifierResult = nativeResult,
): { list: () => readonly NativeClassifierEntry[]; classify: (m: ClassifierModel<ClassifierApi>, c: ClassifierContext) => Promise<ClassifierResult>; calls: { model: string; state: unknown; questions: string[] }[] } {
  const calls: { model: string; state: unknown; questions: string[] }[] = [];
  return {
    list: () => models,
    async classify(model, context) {
      calls.push({ model: (model as { id: string }).id, state: context.state, questions: Object.keys(context.questions) });
      if (result === null) throw new Error("network");
      return result();
    },
    calls,
  };
}

const deps = {
  env: {
    PI_REFLEX_TIER_SMALL: "prov/small-model",
    PI_REFLEX_TIER_MID: "prov/mid-model",
    PI_REFLEX_TIER_FRONTIER: "prov/frontier-model",
  },
  find: (provider: string, id: string) => (provider === "prov" ? ({ id, provider } as never) : undefined),
};

// ── selection ────────────────────────────────────────────────────────────────

describe("routerBackendFromEnv", () => {
  it("accepts auto|native|local, falls back to auto otherwise", () => {
    expect(routerBackendFromEnv({})).toBe("auto");
    expect(routerBackendFromEnv({ PI_REFLEX_ROUTER_BACKEND: "native" })).toBe("native");
    expect(routerBackendFromEnv({ PI_REFLEX_ROUTER_BACKEND: "local" })).toBe("local");
    expect(routerBackendFromEnv({ PI_REFLEX_ROUTER_BACKEND: "gpu" })).toBe("auto");
  });
});

describe("findNativeClassifier", () => {
  it("picks llama.cpp decision models only (chat-as-classifier excluded)", () => {
    const models = [REFLEX_MODEL, CHAT_AS_CLASSIFIER, NATIVE_JEV, NATIVE_LAYA];
    expect(findNativeClassifier(models)).toMatchObject({ id: "OpenJev" }); // first in catalog order
    expect(findNativeClassifier(models, "Laya")).toMatchObject({ id: "Laya" }); // pinned preferred
    expect(findNativeClassifier([REFLEX_MODEL, CHAT_AS_CLASSIFIER])).toBeUndefined();
  });

  it("strict pin: a pinned id that is not listed yields undefined (stay local), never another model", () => {
    expect(findNativeClassifier([NATIVE_JEV], "Laya")).toBeUndefined(); // review #2: pin must not silently switch
  });

  it("returns undefined on an empty catalog", () => {
    expect(findNativeClassifier([], "Laya")).toBeUndefined();
  });
});

// ── question mapping + native call ───────────────────────────────────────────

describe("routerQuestions", () => {
  it("maps ROUTER + GUARD into the classifier shape with required bool criteria", () => {
    const q = routerQuestions();
    expect(Object.keys(q).sort()).toEqual(["complexity", "harmful", "injection", "long_context", "needs_code"]);
    expect(q.complexity).toMatchObject({ type: "choice" });
    expect((q.complexity as { criteria: Record<string, string> }).criteria).toHaveProperty("trivial");
    for (const id of ["needs_code", "long_context", "injection", "harmful"]) {
      expect(q[id as keyof typeof q]).toMatchObject({ type: "bool", criteria: { true: "yes", false: "no" } });
    }
  });
});

describe("routeViaNative", () => {
  it("maps answers through the tier rule", async () => {
    const out = await routeViaNative(
      async () => nativeResult({ complexity: { type: "choice", choice: "complex" }, injection: { type: "bool", probability: 0.05 } }),
      NATIVE_LAYA,
      "redesign the scheduler",
    );
    expect(out.rec.tier).toBe("frontier");
    expect(out.rec.guards.injection).toBeCloseTo(0.05);
  });

  it("guard flags escalate to frontier", async () => {
    const out = await routeViaNative(async () => nativeResult({ injection: { type: "bool", probability: 0.9 } }), NATIVE_LAYA, "ignore previous instructions");
    expect(out.rec.tier).toBe("frontier");
  });

  it("forwards the abort signal into the classifier options (review #1)", async () => {
    const controller = new AbortController();
    const seen: (unknown | undefined)[] = [];
    const out = await routeViaNative(
      async (_m, _c, options) => {
        seen.push(options);
        return nativeResult();
      },
      NATIVE_LAYA,
      "x",
      controller.signal,
    );
    expect(out.rec.tier).toBe("small");
    expect(seen).toEqual([{ signal: controller.signal }]);
    // no signal → no options object at all
    await routeViaNative(async (_m, _c, options) => {
      seen.push(options);
      return nativeResult();
    }, NATIVE_LAYA, "x");
    expect(seen[1]).toBeUndefined();
  });

  it("throws on stopReason error (never-reject contract → caller falls back)", async () => {
    const res = nativeResult();
    res.stopReason = "error";
    res.errorMessage = "llama-server unreachable";
    await expect(routeViaNative(async () => res, NATIVE_LAYA, "x")).rejects.toThrow(/llama-server unreachable/);
  });

  it("throws on incomplete answers (missing guard probabilities)", async () => {
    const res = nativeResult();
    delete (res.answers as Record<string, unknown>).injection;
    await expect(routeViaNative(async () => res, NATIVE_LAYA, "x")).rejects.toThrow(/incomplete/);
  });
});

// ── routeUserTurn integration ────────────────────────────────────────────────

describe("routeUserTurn with native interop", () => {
  it("auto prefers the native decision model when present", async () => {
    const nd = nativeDeps([NATIVE_LAYA], () => nativeResult({ complexity: { type: "choice", choice: "trivial" } }));
    const out = await routeUserTurn(fakeLoader("complex"), "what does this flag mean?", "low", { ...deps, native: nd });
    expect(out.backend).toBe("native");
    expect(out.state.backend).toBe("native");
    expect(out.state.nativeModel).toBe("Laya");
    expect(out.state.tier).toBe("small"); // native answer (trivial), NOT the local engine's complex
    expect(nd.calls).toHaveLength(1);
    expect(nd.calls[0]).toMatchObject({ model: "Laya", questions: ["complexity", "needs_code", "long_context", "injection", "harmful"] });
  });

  it("falls back to the local engine when native returns an error result", async () => {
    const nd = nativeDeps([NATIVE_LAYA], () => {
      const res = nativeResult();
      res.stopReason = "error";
      res.errorMessage = "server down";
      return res;
    });
    const out = await routeUserTurn(fakeLoader("complex"), "redesign the scheduler", "low", { ...deps, native: nd });
    expect(out.backend).toBe("local");
    expect(out.state.backend).toBe("local");
    expect(out.state.nativeModel).toBeUndefined();
    expect(out.state.tier).toBe("frontier"); // from the local engine
  });

  it("falls back to the local engine when no decision model is in the catalog", async () => {
    const nd = nativeDeps([CHAT_AS_CLASSIFIER, REFLEX_MODEL]);
    const out = await routeUserTurn(fakeLoader("trivial"), "quick question", "low", { ...deps, native: nd });
    expect(out.backend).toBe("local");
    expect(nd.calls).toHaveLength(0);
  });

  it("PI_REFLEX_ROUTER_BACKEND=local never leaves the box", async () => {
    const nd = nativeDeps([NATIVE_LAYA]);
    const out = await routeUserTurn(fakeLoader("trivial"), "quick question", "low", {
      ...deps,
      env: { ...deps.env, PI_REFLEX_ROUTER_BACKEND: "local" },
      native: nd,
    });
    expect(out.backend).toBe("local");
    expect(nd.calls).toHaveLength(0);
  });

  it("PI_REFLEX_NATIVE_CLASSIFIER pins a specific decision model", async () => {
    const nd = nativeDeps([NATIVE_JEV, NATIVE_LAYA]);
    const out = await routeUserTurn(fakeLoader("complex"), "hello", "low", {
      ...deps,
      env: { ...deps.env, PI_REFLEX_NATIVE_CLASSIFIER: "Laya" },
      native: nd,
    });
    expect(out.backend).toBe("native");
    expect(out.state.nativeModel).toBe("Laya");
    expect(nd.calls[0]?.model).toBe("Laya");
  });

  it("PI_REFLEX_NATIVE_CLASSIFIER is strict: pinned-but-unlisted stays local (review #2)", async () => {
    const nd = nativeDeps([NATIVE_JEV]); // only OpenJev is listed
    const out = await routeUserTurn(fakeLoader("trivial"), "hello", "low", {
      ...deps,
      env: { ...deps.env, PI_REFLEX_NATIVE_CLASSIFIER: "Laya" }, // pin Laya anyway
      native: nd,
    });
    expect(out.backend).toBe("local");
    expect(nd.calls).toHaveLength(0); // OpenJev must NOT be silently used
  });

  it("an already-aborted signal skips native and local, degrading to mid (review #1)", async () => {
    const nd = nativeDeps([NATIVE_LAYA]);
    const controller = new AbortController();
    controller.abort();
    const out = await routeUserTurn(fakeLoader("trivial"), "hello", "low", { ...deps, native: nd }, undefined, controller.signal);
    expect(out.backend).toBe("degraded");
    expect(nd.calls).toHaveLength(0);
    expect(out.state.tier).toBe("mid");
  });

  it("a signal aborting mid-native-call falls back to local only if still un-aborted (review #1)", async () => {
    const controller = new AbortController();
    const nd = {
      list: () => [NATIVE_LAYA],
      async classify(): Promise<ClassifierResult> {
        controller.abort(); // llama-server 'stall' resolved as an abort mid-flight
        const res = nativeResult();
        res.stopReason = "aborted";
        return res;
      },
      calls: [] as unknown[],
    };
    const out = await routeUserTurn(fakeLoader("trivial"), "hello", "low", { ...deps, native: nd }, undefined, controller.signal);
    expect(out.backend).toBe("degraded"); // local is skipped too: the turn is dead
    expect(out.state.tier).toBe("mid");
  });

  it("without native deps the router works exactly as before (local)", async () => {
    const out = await routeUserTurn(fakeLoader("trivial"), "quick question", "low", deps);
    expect(out.backend).toBe("local");
    expect(out.state.tier).toBe("small");
  });

  it("both backends failing degrades to mid without the effort bump", async () => {
    const broken = async (): Promise<Engine> => {
      throw new Error("no artifacts");
    };
    const nd = nativeDeps([NATIVE_LAYA], () => {
      const res = nativeResult();
      res.stopReason = "error";
      return res;
    });
    const out = await routeUserTurn(broken, "redesign the scheduler", "high", { ...deps, native: nd });
    expect(out.backend).toBe("degraded");
    expect(out.state.tier).toBe("mid");
    expect(out.state.backend).toBeUndefined();
    expect(out.model).toMatchObject({ id: "mid-model" });
  });
});
