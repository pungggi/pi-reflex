/**
 * CONTRACT-harness.md §2 — the similarity-seam adapter (pure, A1: no engine needed).
 *
 * Maps a calibrated `noul` posterior + its NoulConformal prediction set onto the
 * pi-continual-harness `SimilarityResult` seam:
 *
 *     { score: number; abstain?: boolean }   // abstain: true = keep both, never merge/delete
 *
 * Rule zero (contract §1): singleton prediction set → act; anything larger → abstain.
 * An empty set (rare, small-n calibration) is also an abstain.
 */
import { round4 } from "../core/calibration.js";

export interface SimilarityResult {
  /** similarity in [0,1], compared against dedupe.threshold — the calibrated P("same fact") */
  score: number;
  /** true = keep both; never merge or delete. Omitted when acting on a singleton set. */
  abstain?: boolean;
}

/**
 * Convert a noul posterior + prediction set into the harness seam shape.
 *
 * @param pTrue    calibrated P(true) — the "same durable fact" probability
 * @param labelSet prediction set from `NoulConformal.set(pTrue)`: [falseIncluded, trueIncluded]
 * @returns `{ score }` (act) when the set is a singleton, `{ score, abstain: true }` otherwise
 */
export function toSimilarity(pTrue: number, labelSet: [boolean, boolean]): SimilarityResult {
  if (!Number.isFinite(pTrue) || pTrue < 0 || pTrue > 1) {
    throw new RangeError(`pTrue must be a finite probability in [0, 1]; got ${pTrue}`);
  }
  const [falseIn, trueIn] = labelSet;
  const singleton = falseIn !== trueIn;
  const score = round4(pTrue);
  return singleton ? { score } : { score, abstain: true };
}

/** Convenience: run the conformal layer and map in one call (engine-free, once fitted). */
export function toSimilarityFromSet(pTrue: number, set: (p: number) => [boolean, boolean]): SimilarityResult {
  return toSimilarity(pTrue, set(pTrue));
}
