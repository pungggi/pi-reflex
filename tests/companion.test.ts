import { describe, expect, it } from "vitest";
import { createCreateGate, createDedupeSimilarity, createInjectionRelevance, createImportanceRescorer, type EngineLike } from "../src/harness/companion.js";
import type { Answer, NoulAnswer, Questions, ScoreAnswer, SystemOneResult } from "../src/core/types.js";

/** Deterministic fake engine: noul posterior = 0.9, score = 1.5, choice = first option. */
function fakeEngine(opts: { noul?: number; score?: number; delayMs?: number } = {}): EngineLike & { calls: { kind: string; count: number } } {
  const calls = { kind: "none", count: 0 };
  return {
    name: "fake",
    calls,
    async systemOne(state: unknown, questions: Questions): Promise<SystemOneResult> {
      calls.count++;
      calls.kind = `systemOne:${Object.keys(questions).join(",")}`;
      const answers: SystemOneResult["answers"] = {};
      for (const [qid, q] of Object.entries(questions)) {
        if (q.type === "noul") {
          (answers as Record<string, NoulAnswer>)[qid] = { type: "noul", noul: opts.noul ?? 0.9, confidence: 0.9, action: { act_probability: 0.5 } };
        } else if (q.type === "score") {
          (answers as Record<string, ScoreAnswer>)[qid] = {
            type: "score",
            score: opts.score ?? 1.5,
            legend: {},
            probabilities: { "0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25 },
            confidence: 0,
            action: { act_probability: 0.5 },
          };
        }
      }
      return { model: "fake", answers, usage: { input_tokens: 10, output_tokens: 0 } };
    },
    async batchQuestion(states: readonly unknown[]): Promise<Answer[]> {
      calls.count++;
      calls.kind = "batchQuestion";
      return states.map(() => ({
        type: "score",
        score: opts.score ?? 1.5,
        legend: {},
        probabilities: { "0": 0.5, "1": 0.5 },
        confidence: 0,
        action: { act_probability: 0.5 },
      }) as ScoreAnswer);
    },
  };
}

const CALIB = {
  p: [0.98, 0.02, 0.97, 0.03, 0.99, 0.01, 0.96, 0.04, 0.985, 0.015, 0.975, 0.025],
  labels: [true, false, true, false, true, false, true, false, true, false, true, false],
};

describe("D1 — createDedupeSimilarity", () => {
  it("calibrated: confident posterior acts with score", async () => {
    const e = fakeEngine({ noul: 0.98 });
    const sim = createDedupeSimilarity({ engine: e, noulCalibration: CALIB, alpha: 0.1 });
    const r = await sim("item a", "item b");
    expect(r).toEqual({ score: 0.98 });
    expect(r.abstain).toBeUndefined();
  });
  it("uncertain posterior (0.5) abstains", async () => {
    const e = fakeEngine({ noul: 0.5 });
    const sim = createDedupeSimilarity({ engine: e, noulCalibration: CALIB, alpha: 0.1 });
    expect(await sim("x", "y")).toEqual({ score: 0.5, abstain: true });
  });
  it("uncalibrated: plain numeric score (contract §2 compat)", async () => {
    const e = fakeEngine({ noul: 0.42 });
    const sim = createDedupeSimilarity({ engine: e });
    expect(await sim("x", "y")).toEqual({ score: 0.42 });
  });
  it("caches by content pair — second identical pair costs no engine call", async () => {
    const e = fakeEngine();
    const sim = createDedupeSimilarity({ engine: e });
    await sim("alpha", "beta");
    await sim("alpha", "beta");
    expect(e.calls.count).toBe(1);
    await sim("beta", "alpha"); // order-sensitive key: new call
    expect(e.calls.count).toBe(2);
  });
});

describe("D2 — createCreateGate", () => {
  it("confident durable → act (no escalate)", async () => {
    const gate = createCreateGate({ engine: fakeEngine({ noul: 0.97 }), noulCalibration: CALIB });
    const r = await gate("Store retry policy in docs");
    expect(r).toMatchObject({ pTrue: 0.97, abstain: false, escalate: false, calibrated: true });
  });
  it("uncertain → abstain ⇒ escalate to agent (rule zero)", async () => {
    const gate = createCreateGate({ engine: fakeEngine({ noul: 0.5 }), noulCalibration: CALIB });
    const r = await gate("Rename all variables");
    expect(r.abstain).toBe(true);
    expect(r.escalate).toBe(true);
  });
  it("uncalibrated reports calibrated:false (harness threshold decides)", async () => {
    const gate = createCreateGate({ engine: fakeEngine({ noul: 0.6 }) });
    expect((await gate("d")).calibrated).toBe(false);
  });
});

describe("D3 — createImportanceRescorer", () => {
  it("batch scores with conformal interval when calibrated", async () => {
    const rescore = createImportanceRescorer({
      engine: fakeEngine({ score: 2 }),
      scoreCalibration: { pred: [0, 1, 2, 3, 0, 1, 2, 3, 1, 2], truth: [0, 1, 2, 3, 0, 1, 2, 3, 1, 2] },
    });
    const out = await rescore(["item 1", "item 2"]);
    expect(out).toHaveLength(2);
    expect(out[0]?.score).toBe(2);
    expect(out[0]?.interval).toBeDefined();
  });
  it("empty items → empty result, no engine call", async () => {
    const e = fakeEngine();
    const rescore = createImportanceRescorer({ engine: e });
    expect(await rescore([])).toEqual([]);
    expect(e.calls.count).toBe(0);
  });
});

describe("D4 — createInjectionRelevance", () => {
  it("≤3 items: one batch call within budget", async () => {
    const rel = createInjectionRelevance({ engine: fakeEngine(), maxItems: 3, budgetMs: 10_000 });
    const r = await rel(["a", "b", "c"], "current task");
    expect(r.fallback).toBe(false);
    expect(r.scores).toEqual([1.5, 1.5, 1.5]);
  });
  it(">3 items: immediate importance-order fallback, zero engine calls (measured viability)", async () => {
    const e = fakeEngine();
    const rel = createInjectionRelevance({ engine: e, maxItems: 3 });
    const r = await rel(["a", "b", "c", "d"], "task");
    expect(r.fallback).toBe(true);
    expect(r.reason).toContain("importance-order");
    expect(e.calls.count).toBe(0);
  });
  it("over budget: fallback with honest reason", async () => {
    const rel = createInjectionRelevance({ engine: fakeEngine(), maxItems: 3, budgetMs: 0 });
    const r = await rel(["a"], "task");
    expect(r.fallback).toBe(true);
    expect(r.reason).toContain("over budget");
  });
});
