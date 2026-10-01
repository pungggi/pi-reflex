import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { SystemOneResult } from "../src/core/types.js";
import {
  createReflexClassifier,
  mapAnswer,
  mapQuestion,
  mapQuestions,
  reflexClassifierModels,
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
    const qs = mapQuestions({ a: { type: "bool", instructions: "q" }, b: { type: "score", instructions: "s", criteria: ["x", "y"] } });
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
      questions: { ok: { type: "bool", instructions: "Does the user approve?" } },
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
      impl.classify(model, { state: {}, questions: { q: { type: "bool", instructions: "?" } } }),
    ).rejects.toThrow(/no artifacts/);
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
    }
  });
});
