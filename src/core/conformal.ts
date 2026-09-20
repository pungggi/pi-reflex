/**
 * Conformal guarantee layer — pi-reflex's value-add over laya/von.
 *
 * laya/von expose heuristic confidence gating ("escalate if confidence < 0.85").
 * We wrap the same calibrated probabilities in split-conformal prediction sets with
 * finite-sample coverage guarantees (Vovk et al. 2005; Angelopoulos & Bates 2021):
 *
 *     P(y_true ∈ S(x)) ≥ 1 − α   for exchangeable calibration & test data.
 *
 * Contract mapping (CONTRACT-harness.md rule zero): **an empty or multi-element
 * prediction set means ABSTAIN** — act only on a singleton. Empty sets are rare but
 * legitimate at small calibration sizes (finite-sample honesty), never a bug.
 *
 * References: arXiv:2107.07511 (gentle intro), arXiv:2009.14193 (prediction sets),
 * arXiv:2208.02814 (conformal risk control), arXiv:1705.08500 (selective classification).
 */

function validateAlpha(alpha: number): void {
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new RangeError(`alpha must be in the open interval (0, 1); got ${alpha}`);
  }
}

function validateFinite(values: readonly number[], what: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new RangeError(`${what}[${i}] must be finite; got ${values[i]}`);
    }
  }
}

/** Split-conformal prediction sets for `choice` (score = 1 − p_true_label). */
export class ChoiceConformal {
  private qhat = 1;
  private n = 0;
  private alpha = 0.1;

  /** Fit on calibration data: per-example probability vectors and true label indices. */
  fit(probs: number[][], labels: number[], alpha = 0.1): this {
    validateAlpha(alpha);
    if (probs.length === 0 || probs.length !== labels.length) {
      throw new RangeError("calibration data must be non-empty and probs/labels must have equal length");
    }
    const scores = probs.map((p, i) => {
      if (!Array.isArray(p) || p.length === 0) throw new RangeError(`probs[${i}] must be a non-empty array`);
      validateFinite(p, `probs[${i}]`);
      const label = labels[i];
      if (!Number.isInteger(label) || (label as number) < 0 || (label as number) >= p.length) {
        throw new RangeError(`labels[${i}] must be an integer index into probs[${i}] (0..${p.length - 1}); got ${label}`);
      }
      return 1 - p[label as number]!; // guarded by the label bounds check above
    });
    this.alpha = alpha;
    this.n = probs.length;
    const sorted = [...scores].sort((a, b) => a - b);
    const rank = Math.ceil((this.n + 1) * (1 - alpha)); // 1-indexed order statistic
    // rank ∈ [1..n] here (alpha ∈ (0,1) ⇒ 1 ≤ ceil((n+1)(1-α)) ≤ n+1; > n ⇒ full set):
    this.qhat = rank > this.n ? 1 : sorted[rank - 1]!;
    return this;
  }

  /** Prediction set: indices with p_k ≥ 1 − q̂. Guaranteed coverage ≥ 1−α on exchangeable data. */
  set(probs: number[]): number[] {
    const floor = 1 - this.qhat;
    const out: number[] = [];
    for (let k = 0; k < probs.length; k++) {
      if ((probs[k] ?? Number.NaN) >= floor) out.push(k);
    }
    return out;
  }

  get threshold(): number {
    return this.qhat;
  }

  toJSON(): { n: number; alpha: number; qhat: number } {
    return { n: this.n, alpha: this.alpha, qhat: this.qhat };
  }
}

/** Split-conformal for `noul` (binary): returns the guaranteed-coverage label set {0,1}, {0}, {1}. */
export class NoulConformal {
  private inner = new ChoiceConformal();

  /** Fit on calibration data: P(true) values and boolean labels. */
  fit(pTrue: number[], labels: boolean[], alpha = 0.1): this {
    validateAlpha(alpha);
    if (pTrue.length === 0 || pTrue.length !== labels.length) {
      throw new RangeError("calibration data must be non-empty and pTrue/labels must have equal length");
    }
    validateFinite(pTrue, "pTrue");
    this.inner.fit(
      pTrue.map((p) => [1 - p, p]),
      labels.map((b) => (b ? 1 : 0)),
      alpha,
    );
    return this;
  }

  /** Label set as [falseIncluded, trueIncluded]. Singleton ⇒ act; empty or both ⇒ abstain. */
  set(pTrue: number): [boolean, boolean] {
    const s = this.inner.set([1 - pTrue, pTrue]);
    return [s.includes(0), s.includes(1)];
  }

  toJSON() {
    return this.inner.toJSON();
  }
}

/**
 * Conformal interval for `score`: symmetric interval around the expected level with
 * marginal coverage ≥ 1−α (score = |E[L] − y_true| residual quantile).
 */
export class ScoreConformal {
  private halfWidth = Infinity;
  private alpha = 0.1;

  fit(predicted: number[], truth: number[], alpha = 0.1): this {
    validateAlpha(alpha);
    if (predicted.length === 0 || predicted.length !== truth.length) {
      throw new RangeError("calibration data must be non-empty and predicted/truth must have equal length");
    }
    validateFinite(predicted, "predicted");
    validateFinite(truth, "truth");
    this.alpha = alpha;
    const residuals = predicted.map((p, i) => Math.abs(p - truth[i]!)).sort((a, b) => a - b); // equal lengths validated
    const n = residuals.length;
    const rank = Math.ceil((n + 1) * (1 - alpha));
    this.halfWidth = rank > n ? Infinity : residuals[rank - 1]!;
    return this;
  }

  interval(expected: number): { lo: number; hi: number } {
    return { lo: expected - this.halfWidth, hi: expected + this.halfWidth };
  }

  /** halfWidth is a number, or the string "Infinity" for the unbounded fallback (JSON-safe). */
  toJSON(): { alpha: number; halfWidth: number | string } {
    return {
      alpha: this.alpha,
      halfWidth: Number.isFinite(this.halfWidth) ? this.halfWidth : "Infinity",
    };
  }
}

/**
 * Conformal-risk-control escalation threshold: pick the highest automation coverage
 * whose empirical risk on calibration data stays ≤ alpha (greedy, per arXiv:2208.02814 §5).
 *
 * Returns a confidence threshold: automate while confidence ≥ threshold.
 * When nothing can be automated within the risk budget, threshold is 2 (never met)
 * and `risk` is reported as 1 — there is no data backing a lower risk claim.
 */
export function escalationThreshold(
  confidences: number[],
  errors: boolean[],
  alpha = 0.05,
): { threshold: number; coverage: number; risk: number } {
  validateAlpha(alpha);
  const n = confidences.length;
  if (n === 0 || confidences.length !== errors.length) {
    throw new RangeError("calibration data must be non-empty and confidences/errors must have equal length");
  }
  validateFinite(confidences, "confidences");
  const order = confidences.map((_, i) => i).sort((a, b) => confidences[b]! - confidences[a]!);

  let taken = 0;
  let errs = 0;
  let best = { threshold: 2, coverage: 0, risk: 1 };

  for (const i of order) {
    taken++;
    if (errors[i] === true) errs++;
    const risk = errs / taken;
    if (risk <= alpha) best = { threshold: confidences[i]!, coverage: taken / n, risk };
  }
  return best;
}
