import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { SystemOneResult } from "../src/core/types.js";
import {
  createReflexClassifier,
  mapAnswer,
  mapQuestion,
  mapQuestions,
  reflexClassifierModels,
  registerReflexProvider,
  REFLEX_PROVIDER_ID,
} from "../src/extension/provider.js";

function fakeEngine(answer: (qid: string) => SystemOneResult["answers"][string]): Engine {
  return {
    name: "fake",
    async systemOne(_state: unknown, questions: Parameters<Engine["systemOne"]>[1]): Promise<SystemOneResult> {
      const answers: SystemOneResult["answers"] = {};
      for (const qid of Object.keys(questions)) answers[qid] = answer(qid);
      return { model: "fake", answers, usage: { input_tokens: 11, output_tokens: 0 } };
    },
    async batchQuestion() {
      return [];
    },
  } as unknown as Engine;
}

describe("classifier question mapping", () => {
  it("choice and score pass through 1:1", () => {
    expect(mapQuestion({ type: "choice", instructions: "Which?", criteria: { a: "x", b: "y" } })).toEqual({
      type: "choice",
      instructions: "Which?",
      criteria: { a: "x", b: "y" },
    });
    expect(mapQuestion({ type: "score", instructions: "How bad?", criteria: ["low", "high"] })).toEqual({
      type: "score",
      instructions: "How bad?",
      criteria: ["low", "high"],
    });
  });
  it("bool maps to noul with true/false criteria", () => {
    expect(mapQuestion({ type: "bool", instructions: "Is it?", criteria: { true: "yes", false: "no" } })).toEqual({
      type: "noul",
      instructions: "Is it?",
      criteria: { true: "yes", false: "no" },
    });
  });
  it("mapQuestions preserves question ids", () => {
    const qs = mapQuestions({ a: { type: "bool", instructions: "q", criteria: { true: "yes", false: "no" } }, b: { type: "score", instructions: "s", criteria: ["x", "y"] } });
    expect(Object.keys(qs)).toEqual(["a", "b"]);
    expect(qs.a?.type).toBe("noul");
    expect(qs.b?.type).toBe("score");
  });
});

describe("classifier answer mapping", () => {
  it("noul → bool probability; choice/score keep shape", () => {
    expect(mapAnswer({ type: "noul", noul: 0.83, confidence: 0.9 })).toEqual({ type: "bool", probability: 0.83 });
    expect(mapAnswer({ type: "choice", choice: "a", probabilities: { a: 0.7 }, confidence: 0.6 })).toEqual({
      type: "choice",
      choice: "a",
      probabilities: { a: 0.7 },
      confidence: 0.6,
    });
    expect(mapAnswer({ type: "score", score: 1.5, confidence: 0.4 })).toEqual({ type: "score", score: 1.5, confidence: 0.4 });
  });
});

describe("createReflexClassifier.classify", () => {
  it("returns a pi ClassifierResult with mapped answers and usage", async () => {
    const impl = createReflexClassifier(async () =>
      fakeEngine(() => ({ type: "noul", noul: 0.77, confidence: 0.8, action: { act_probability: 0.5 } })),
    );
    const model = { id: "multilingual", api: "reflex-onnx" } as never;
    const res = await impl.classify(model, {
      state: { message: "ship it" },
      questions: { ok: { type: "bool", instructions: "Does the user approve?", criteria: { true: "yes", false: "no" } } },
    });
    expect(res.stopReason).toBe("stop");
    expect(res.provider).toBe("reflex");
    expect(res.model).toBe("multilingual");
    expect(res.answers.ok).toEqual({ type: "bool", probability: 0.77 });
    expect(res.usage).toMatchObject({ input: 11, output: 0, totalTokens: 11, cost: { total: 0 } });
  });

  it("propagates engine failures (ModelRuntime converts them to error results)", async () => {
    const impl = createReflexClassifier(async () => {
      throw new Error("no artifacts");
    });
    const model = { id: "english", api: "reflex-onnx" } as never;
    await expect(
      impl.classify(model, { state: {}, questions: { q: { type: "bool", instructions: "?", criteria: { true: "yes", false: "no" } } } }),
    ).rejects.toThrow(/no artifacts/);
  });

  // pi ≥ 1.4: ClassifierContext gained `images`; pi's runtime refuses them for
  // input:["text"] models, but direct callers must not get a silently
  // image-blind text answer either.
  it("rejects image contexts with a hosted-classifier pointer (pi ≥ 1.4 image contract)", async () => {
    let engineCalled = false;
    const impl = createReflexClassifier(async () => {
      engineCalled = true;
      return fakeEngine(() => ({ type: "noul", noul: 0.9, confidence: 0.9, action: { act_probability: 0.5 } }));
    });
    const model = { id: "multilingual", api: "reflex-onnx" } as never;
    const image = { type: "image" as const, data: "aGk=", mimeType: "image/png" };
    await expect(
      impl.classify(model, {
        state: {},
        images: [image],
        questions: { q: { type: "bool", instructions: "?", criteria: { true: "yes", false: "no" } } },
      }),
    ).rejects.toThrow(/text-only.*judge images/s);
    expect(engineCalled).toBe(false); // refused before any engine work
  });
});

describe("reflexClassifierModels", () => {
  it("registers three local classifier models with zero cost", () => {
    const models = reflexClassifierModels();
    expect(models.map((m) => m.id)).toEqual(["multilingual", "english", "typed-decisions"]);
    for (const m of models) {
      expect(m.type).toBe("classifier");
      expect(m.api).toBe("reflex-onnx");
      expect(m.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
      expect(m.contextWindow).toBeGreaterThan(0);
      // pi ≥ 1.4 image contract: with input:["text"] (no "image"), pi's
      // ModelRuntime returns an ERROR RESULT for ClassifierContext.images
      // instead of dispatching — codemode scripts can rely on that.
      expect(m.input).toEqual(["text"]);
    }
  });
});

describe("registerReflexProvider", () => {
  it("passes a truthy baseUrl — pi's composer rejects custom models without one", () => {
    const captured: { id: string; config: Record<string, unknown> }[] = [];
    const pi = { registerProvider: (id: string, config: Record<string, unknown>) => captured.push({ id, config }) } as never;
    registerReflexProvider(pi, () => Promise.resolve(fakeEngine(() => ({ type: "noul", noul: 0.5, confidence: 0.9, action: { act_probability: 0.5 } }))));
    expect(captured).toHaveLength(1);
    expect(captured[0]!.id).toBe(REFLEX_PROVIDER_ID);
    // Regression: pi throws 'Provider reflex: "baseUrl" is required when defining custom models.'
    expect(captured[0]!.config.baseUrl).toBeTruthy();
    expect(captured[0]!.config.apiKey).toBeTruthy();
    expect((captured[0]!.config.models as { id: string }[]).map((m) => m.id)).toEqual(["multilingual", "english", "typed-decisions"]);
  });
});
