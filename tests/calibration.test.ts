import { describe, expect, it } from "vitest";
import { calibratedSoftmax, confidenceFromProbs, round4, tempBucket, temperatureFor } from "../src/core/calibration.js";
import { buildAnswer, softmax } from "../src/core/answers.js";
import { toInternal, type QuestionDef } from "../src/core/types.js";

describe("confidenceFromProbs", () => {
  it("uniform distribution → 0, one-hot → 1", () => {
    expect(confidenceFromProbs([0.25, 0.25, 0.25, 0.25], 4)).toBeCloseTo(0, 10);
    expect(confidenceFromProbs([1, 0, 0], 3)).toBeCloseTo(1, 10);
  });
  it("k<2 returns 1", () => {
    expect(confidenceFromProbs([1], 1)).toBe(1);
  });
  it("handles zeros without NaN", () => {
    const c = confidenceFromProbs([0.5, 0.5, 0, 0], 4);
    expect(Number.isFinite(c)).toBe(true);
  });
});

describe("tempBucket", () => {
  it("buckets by qtype and option count", () => {
    expect(tempBucket(0, 2)).toBe("choice:2");
    expect(tempBucket(0, 5)).toBe("choice:3-5");
    expect(tempBucket(0, 10)).toBe("choice:6-10");
    expect(tempBucket(0, 11)).toBe("choice:11+");
    expect(tempBucket(1, 4)).toBe("score:3-5");
    expect(tempBucket(2, 2)).toBe("noul:2");
  });
});

describe("temperatureFor", () => {
  const temps = {
    temperature: [1.0, 1.2, 1.05],
    temperatureByOptions: { "noul:2": 1.03, "choice:3-5": 1.17 },
  };
  it("uses bucket override when present, per-qtype fallback otherwise", () => {
    expect(temperatureFor(temps, 2, 2)).toBe(1.03);
    expect(temperatureFor(temps, 0, 4)).toBe(1.17);
    expect(temperatureFor(temps, 0, 12)).toBe(1.0);
    expect(temperatureFor(temps, 1, 4)).toBe(1.2);
  });
});

describe("calibratedSoftmax", () => {
  it("is stable and sums to 1", () => {
    const p = calibratedSoftmax([1000, 999, 0], 3, 1.17);
    const s = p.reduce((a, b) => a + b, 0);
    expect(s).toBeCloseTo(1, 10);
    expect(p[0]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[2]);
  });
  it("higher temperature flattens", () => {
    const cold = calibratedSoftmax([2, 0], 2, 0.5);
    const hot = calibratedSoftmax([2, 0], 2, 4);
    expect(cold[0]).toBeGreaterThan(hot[0]);
  });
});

describe("buildAnswer", () => {
  const temps = { temperature: [1, 1, 1], temperatureByOptions: {} };

  it("choice: argmax label, probabilities, entropy confidence, act prob", () => {
    const q = toInternal({ type: "choice", instructions: "i", criteria: { billing: "b", tech: "t", sales: "s" } } as QuestionDef);
    const a = buildAnswer(q, [0.2, 4.5, 0.1], [0.7, 0.3], temps);
    if (a.type !== "choice") throw new Error("expected choice");
    expect(a.choice).toBe("tech");
    expect(a.probabilities.billing + a.probabilities.tech + a.probabilities.sales).toBeCloseTo(1, 3);
    expect(a.confidence).toBeGreaterThan(0.5);
    expect(a.action.act_probability).toBeCloseTo(Math.exp(0.7) / (Math.exp(0.7) + Math.exp(0.3)), 4);
  });

  it("score: expected value over levels", () => {
    const q = toInternal({ type: "score", instructions: "i", criteria: ["low", "mid", "high"] } as QuestionDef);
    const a = buildAnswer(q, [0.1, 0.2, 3.0], [0, 1], temps);
    if (a.type !== "score") throw new Error("expected score");
    // strongly favors level 2 → score near 2
    expect(a.score).toBeGreaterThan(1.8);
    expect(a.legend).toEqual({ "0": "low", "1": "mid", "2": "high" });
    expect(Object.keys(a.probabilities)).toEqual(["0", "1", "2"]);
  });

  it("noul: P(true) from index 1, confidence = max(p, 1-p)", () => {
    const q = toInternal({ type: "noul", instructions: "urgent?" } as QuestionDef);
    const a = buildAnswer(q, [-1.0, 2.0], [1.0, 0.5], temps);
    if (a.type !== "noul") throw new Error("expected noul");
    const p1 = Math.exp(2) / (Math.exp(-1) + Math.exp(2));
    expect(a.noul).toBeCloseTo(p1, 3);
    expect(a.confidence).toBeCloseTo(Math.max(p1, 1 - p1), 3);
  });

  it("rounds to 4 decimals like laya", () => {
    const q = toInternal({ type: "noul", instructions: "x" } as QuestionDef);
    const a = buildAnswer(q, [0, 0], [0, 0], temps);
    if (a.type !== "noul") throw new Error("expected noul");
    expect(a.noul).toBe(0.5);
    expect(String(a.noul.split?.(".")?.[1]?.length ?? 1)).toMatch(/^[01]$/);
  });
});

describe("softmax/round4", () => {
  it("softmax sums to 1", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });
  it("round4 halves away from zero", () => {
    expect(round4(0.12345)).toBe(0.1235);
    expect(round4(1.00004)).toBe(1);
  });
});
