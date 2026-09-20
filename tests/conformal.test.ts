import { describe, expect, it } from "vitest";
import { ChoiceConformal, escalationThreshold, NoulConformal, ScoreConformal } from "../src/core/conformal.js";

/** Deterministic pseudo-random for reproducible tests. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("ChoiceConformal", () => {
  it("achieves ≥ 1−α coverage on exchangeable synthetic data", () => {
    const rng = mulberry32(42);
    const K = 5;
    const mk = (): { probs: number[]; label: number } => {
      const label = Math.floor(rng() * K);
      const raw = Array.from({ length: K }, () => rng());
      raw[label] += 1.5;
      const s = raw.reduce((a, b) => a + b, 0);
      return { probs: raw.map((v) => v / s), label };
    };
    const calib = Array.from({ length: 2000 }, mk);
    const test = Array.from({ length: 4000 }, mk);
    const conformal = new ChoiceConformal().fit(
      calib.map((c) => c.probs),
      calib.map((c) => c.label),
      0.1,
    );
    let covered = 0;
    for (const t of test) if (conformal.set(t.probs).includes(t.label)) covered++;
    const coverage = covered / test.length;
    expect(coverage).toBeGreaterThanOrEqual(0.9);
    expect(coverage).toBeLessThanOrEqual(0.95); // not trivially full sets
  });

  it("rank > n gives full-set fallback", () => {
    // n=2, α=0.25 → rank = ceil(3·0.75) = 3 > 2 → q̂ = 1 → always the full label set
    const c = new ChoiceConformal().fit([[0.9, 0.1], [0.2, 0.8]], [0, 1], 0.25);
    expect(c.set([0.5, 0.5]).length).toBe(2);
  });

  it("small calibration sets can legitimately produce empty sets (finite-sample honesty)", () => {
    // n=2, α=0.5 → rank = 2 ≤ n → q̂ = max score; a 50/50 input falls below it
    const c = new ChoiceConformal().fit([[0.9, 0.1], [0.2, 0.8]], [0, 1], 0.5);
    expect(c.set([0.5, 0.5]).length).toBe(0);
  });
});

describe("NoulConformal", () => {
  it("covers binary truth at the nominal rate", () => {
    const rng = mulberry32(7);
    const calib = Array.from({ length: 2000 }, () => {
      const truth = rng() > 0.5;
      const p = Math.min(0.99, Math.max(0.01, truth ? 0.6 + 0.4 * rng() : 0.4 * rng()));
      return { p, truth };
    });
    const noul = new NoulConformal().fit(
      calib.map((c) => c.p),
      calib.map((c) => c.truth),
      0.1,
    );
    let covered = 0;
    for (const c of calib) {
      const [f, t] = noul.set(c.p);
      if ((c.truth && t) || (!c.truth && f)) covered++;
    }
    expect(covered / calib.length).toBeGreaterThanOrEqual(0.9);
  });
});

describe("ScoreConformal", () => {
  it("interval covers residuals at the nominal rate", () => {
    const rng = mulberry32(11);
    const calib = Array.from({ length: 1500 }, () => {
      const pred = 4 * rng();
      return { pred, truth: Math.min(4, Math.max(0, pred + (rng() - 0.5))) };
    });
    const sc = new ScoreConformal().fit(
      calib.map((c) => c.pred),
      calib.map((c) => c.truth),
      0.1,
    );
    let covered = 0;
    for (const c of calib) {
      const iv = sc.interval(c.pred);
      if (c.truth >= iv.lo && c.truth <= iv.hi) covered++;
    }
    expect(covered / calib.length).toBeGreaterThanOrEqual(0.9);
  });
});

describe("escalationThreshold (conformal risk control)", () => {
  it("finds max coverage with risk ≤ α", () => {
    const rng = mulberry32(3);
    const confidences: number[] = [];
    const errors: boolean[] = [];
    for (let i = 0; i < 3000; i++) {
      const conf = rng();
      const err = conf < 0.5 ? rng() < 0.1 : rng() < 0.01; // sharp separation
      confidences.push(conf);
      errors.push(err);
    }
    const r = escalationThreshold(confidences, errors, 0.05);
    expect(r.risk).toBeLessThanOrEqual(0.05 + 1e-9);
    expect(r.coverage).toBeGreaterThan(0.4);
  });

  it("degenerate all-error data automates nothing", () => {
    const r = escalationThreshold([0.9, 0.8, 0.7], [true, true, true], 0.05);
    expect(r.coverage).toBe(0);
  });
});
