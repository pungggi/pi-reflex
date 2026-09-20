/**
 * Conformal guarantee layer — pi-jev's value-add over laya/von.
 *
 * laya/von expose heuristic confidence gating ("escalate if confidence < 0.85").
 * We wrap the same calibrated probabilities in split-conformal prediction sets with
 * finite-sample coverage guarantees (Vovk et al. 2005; Angelopoulos & Bates 2021):
 *
 *     P(y_true ∈ S(x)) ≥ 1 − α   for exchangeable calibration & test data.
 *
 * References: arXiv:2107.07511 (gentle intro), arXiv:2009.14193 (prediction sets),
 * arXiv:2208.02814 (conformal risk control), arXiv:1705.08500 (selective classification).
 */

/** Split-conformal prediction sets for `choice` (score = 1 − p_true_label). */
export class ChoiceConformal {
  private qhat = 1;
  private n = 0;
  private alpha = 0.1;

  /** Fit on calibration data: per-example probability vectors and true label indices. */
  fit(probs: number[][], labels: number[], alpha = 0.1): this {
    if (probs.length !== labels.length || probs.length === 0) throw new Error("empty or mismatched calibration data");
    this.alpha = alpha;
    this.n = probs.length;
    const scores = probs.map((p, i) => 1 - p[labels[i]]);
    const sorted = [...scores].sort((a, b) => a - b);
    const rank = Math.ceil((this.n + 1) * (1 - alpha)); // 1-indexed order statistic
    this.qhat = rank > this.n ? 1 : sorted[rank - 1];
    return this;
  }

  /** Prediction set: indices with p_k ≥ 1 − q̂. Guaranteed coverage ≥ 1−α on exchangeable data. */
  set(probs: number[]): number[] {
    const out: number[] = [];
    for (let k = 0; k < probs.length; k++) if (probs[k] >= 1 - this.qhat) out.push(k);
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
    this.inner.fit(
      pTrue.map((p) => [1 - p, p]),
      labels.map((b) => (b ? 1 : 0)),
      alpha,
    );
    return this;
  }

  /** Label set as [falseIncluded, trueIncluded]. */
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
    if (predicted.length !== truth.length || predicted.length === 0) throw new Error("empty or mismatched calibration data");
    this.alpha = alpha;
    const residuals = predicted.map((p, i) => Math.abs(p - truth[i])).sort((a, b) => a - b);
    const n = residuals.length;
    const rank = Math.ceil((n + 1) * (1 - alpha));
    this.halfWidth = rank > n ? Infinity : residuals[rank - 1];
    return this;
  }

  interval(expected: number): { lo: number; hi: number } {
    return { lo: expected - this.halfWidth, hi: expected + this.halfWidth };
  }

  toJSON(): { alpha: number; halfWidth: number } {
    return { alpha: this.alpha, halfWidth: this.halfWidth };
  }
}

/**
 * Conformal-risk-control escalation threshold: pick the highest automation coverage
 * whose empirical risk on calibration data stays ≤ alpha (greedy, per arXiv:2208.02814 §5).
 *
 * Returns a confidence threshold: automate while confidence ≥ threshold.
 */
export function escalationThreshold(
  confidences: number[],
  errors: boolean[],
  alpha = 0.05,
): { threshold: number; coverage: number; risk: number } {
  const n = confidences.length;
  if (n === 0 || confidences.length !== errors.length) throw new Error("empty or mismatched calibration data");
  const order = confidences.map((c, i) => i).sort((a, b) => confidences[b] - confidences[a]);

  let taken = 0;
  let errs = 0;
  let best = { threshold: 2 /* automate nothing */, coverage: 0, risk: 0 };

  for (const i of order) {
    taken++;
    if (errors[i]) errs++;
    const risk = errs / taken;
    if (risk <= alpha) best = { threshold: confidences[i], coverage: taken / n, risk };
  }
  return best;
}
