import { describe, expect, it } from "vitest";
import { ChoiceConformal, escalationThreshold, NoulConformal, ScoreConformal } from "../src/core/conformal.js";

/**
 * H4 golden tests (CONTRACT-harness.md §5.4 determinism invariant): fixed calibration
 * inputs, inline-literal expected outputs. These pin the exact qhat/threshold math —
 * if one of these fails, conformal guarantees may have drifted and the contract's
 * "coordinated change" rule kicks in.
 */

const P = [
  [0.9, 0.1],
  [0.2, 0.8],
  [0.6, 0.4],
  [0.55, 0.45],
];
const L = [0, 1, 0, 1];

describe("ChoiceConformal golden", () => {
  it("alpha=0.2 → qhat=0.55, sets per literal expectation", () => {
    const c = new ChoiceConformal().fit(P, L, 0.2);
    expect(c.threshold).toBe(0.55);
    expect(c.toJSON()).toEqual({ n: 4, alpha: 0.2, qhat: 0.55 });
    expect(c.set([0.6, 0.4])).toEqual([0]); // singleton → act
    expect(c.set([0.9, 0.1])).toEqual([0]);
    expect(c.set([0.5, 0.5])).toEqual([0, 1]); // both → abstain
    expect(c.set([0.2, 0.8])).toEqual([1]);
  });
  it("small n can legitimately produce an empty set → contract ABSTAIN", () => {
    const c = new ChoiceConformal().fit([[0.9, 0.1], [0.2, 0.8]], [0, 1], 0.5);
    expect(c.set([0.5, 0.5])).toEqual([]); // empty ⇒ abstain (never a bug)
  });
});

describe("NoulConformal golden", () => {
  it("alpha=0.2 → qhat=0.55, label sets per literal expectation", () => {
    const n = new NoulConformal().fit([0.9, 0.2, 0.6, 0.55], [true, false, true, false], 0.2);
    expect(n.toJSON().qhat).toBeCloseTo(0.55, 12);
    expect(n.set(0.9)).toEqual([false, true]); // singleton true → act
    expect(n.set(0.1)).toEqual([true, false]); // singleton false → act
    expect(n.set(0.5)).toEqual([true, true]); // both → abstain
  });
});

describe("ScoreConformal golden", () => {
  it("alpha=0.25 → halfWidth=1, interval literal (exact-float fixture)", () => {
    const s = new ScoreConformal().fit([1, 2, 3, 4], [1, 1.5, 4, 3.5], 0.25);
    expect(s.toJSON()).toEqual({ alpha: 0.25, halfWidth: 1 });
    expect(s.interval(2)).toEqual({ lo: 1, hi: 3 });
  });
  it("rank > n → unbounded interval, JSON-safe serialization", () => {
    const s = new ScoreConformal().fit([1], [2], 0.25);
    expect(s.toJSON()).toEqual({ alpha: 0.25, halfWidth: "Infinity" });
  });
});

describe("escalationThreshold golden", () => {
  it("clean calibration → full coverage", () => {
    expect(escalationThreshold([0.9, 0.8, 0.7], [false, false, false], 0.05)).toEqual({
      threshold: 0.7,
      coverage: 1,
      risk: 0,
    });
  });
  it("one error → threshold at the last safe prefix", () => {
    const r = escalationThreshold([0.9, 0.8, 0.7], [false, true, false], 0.05);
    expect(r.threshold).toBe(0.9);
    expect(r.coverage).toBeCloseTo(1 / 3, 12);
    expect(r.risk).toBe(0);
  });
  it("degenerate: nothing automatable → threshold 2, coverage 0, risk honestly 1 (M8)", () => {
    expect(escalationThreshold([0.9], [true], 0.05)).toEqual({ threshold: 2, coverage: 0, risk: 1 });
  });
});

describe("conformal input validation (M1) — throw, never silently NaN", () => {
  it("rejects empty/mismatched calibration", () => {
    expect(() => new ChoiceConformal().fit([], [], 0.1)).toThrow(RangeError);
    expect(() => new ChoiceConformal().fit(P, [0, 1], 0.1)).toThrow(RangeError);
    expect(() => new NoulConformal().fit([], [], 0.1)).toThrow(RangeError);
    expect(() => new ScoreConformal().fit([], [1], 0.1)).toThrow(RangeError);
    expect(() => escalationThreshold([], [], 0.1)).toThrow(RangeError);
  });
  it("rejects out-of-range labels", () => {
    expect(() => new ChoiceConformal().fit(P, [0, 1, 2, 0], 0.1)).toThrow(/labels\[2\]/);
    expect(() => new ChoiceConformal().fit(P, [0, 1, -1, 0], 0.1)).toThrow(/labels\[2\]/);
  });
  it("rejects alpha outside (0,1)", () => {
    expect(() => new ChoiceConformal().fit(P, L, 0)).toThrow(RangeError);
    expect(() => new ChoiceConformal().fit(P, L, 1)).toThrow(RangeError);
    expect(() => new ChoiceConformal().fit(P, L, Number.NaN)).toThrow(RangeError);
    expect(() => new NoulConformal().fit([0.5], [true], 1.0)).toThrow(RangeError);
    expect(() => new ScoreConformal().fit([1], [1], 0)).toThrow(RangeError);
    expect(() => escalationThreshold([0.5], [false], 0)).toThrow(RangeError);
  });
  it("rejects NaN/Infinity in inputs", () => {
    expect(() => new ChoiceConformal().fit([[Number.NaN, 0.5]], [0], 0.1)).toThrow(/finite/);
    expect(() => new NoulConformal().fit([Number.NaN], [true], 0.1)).toThrow(/finite/);
    expect(() => new ScoreConformal().fit([Number.POSITIVE_INFINITY], [1], 0.1)).toThrow(/finite/);
    expect(() => escalationThreshold([Number.NaN], [false], 0.05)).toThrow(/finite/);
  });
});
