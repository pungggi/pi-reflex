/**
 * The pi-continual-harness companion (CONTRACT-harness.md call-sites D1–D4).
 *
 * Every factory takes an EngineLike (the real Engine satisfies it) so companions are
 * testable without artifacts, plus optional calibration data for the conformal layer.
 * Soft-fail everywhere: no calibration → plain scores (contract §2 keeps numeric
 * returns working); over budget → the contract's own fallback, never a blocked turn.
 */
import { NoulConformal, ScoreConformal } from "../core/conformal.js";
import type { Answer, NoulAnswer, Questions, QuestionDef, ScoreAnswer, SystemOneResult } from "../core/types.js";
import { toSimilarity, type SimilarityResult } from "./similarity.js";
import { contentKey } from "../engine/download.js";

/** The subset of Engine the companions need (Engine satisfies this structurally). */
export interface EngineLike {
  name: string;
  systemOne(state: unknown, questions: Questions): Promise<SystemOneResult>;
  batchQuestion(states: readonly unknown[], question: Questions[string]): Promise<Answer[]>;
}

class LruCache<T> {
  private map = new Map<string, T>();
  constructor(private readonly max: number) {}
  get(key: string): T | undefined {
    const v = this.map.get(key);
    if (v !== undefined) {
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }
  set(key: string, value: T): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.max) {
      const oldest = this.map.keys().next().value;
      if (oldest === undefined) break;
      this.map.delete(oldest);
    }
  }
}

export interface CompanionOptions {
  engine: EngineLike;
  /** dedupe-pairs calibration: P("same fact") values + true labels */
  noulCalibration?: { p: number[]; labels: boolean[] };
  /** importance calibration: predicted vs truth levels */
  scoreCalibration?: { pred: number[]; truth: number[] };
  alpha?: number;
  budgetMs?: number;
  cacheSize?: number;
}

const D1_QUESTION: Questions = { same: { type: "noul", instructions: "Are these two items the same durable fact?" } };
const D2_QUESTION: Questions = { durable: { type: "noul", instructions: "Is this proposed change durable and reusable across future sessions?" } };
const D3_QUESTION: QuestionDef = {
  type: "score",
  instructions: "How fit is this stored item for future reuse?",
  criteria: ["unfit", "marginal", "useful", "highly reusable"],
};
const D4_QUESTION: QuestionDef = {
  type: "score",
  instructions: "How relevant is this item to the current task?",
  criteria: ["irrelevant", "relevant"],
};

/**
 * D1 — dedupe similarity for the harness seam. Contents are immutable between
 * mutations, so results cache by content key (contract §3 sanctions this).
 * Uncalibrated → plain numeric score (still a valid seam return).
 */
export function createDedupeSimilarity(opts: CompanionOptions) {
  const cache = new LruCache<SimilarityResult>(opts.cacheSize ?? 4096);
  const conformal = opts.noulCalibration ? new NoulConformal().fit(opts.noulCalibration.p, opts.noulCalibration.labels, opts.alpha ?? 0.1) : null;
  const budgetMs = opts.budgetMs ?? 250;
  return async function similarity(a: string, b: string): Promise<SimilarityResult> {
    const key = contentKey(a, b);
    const hit = cache.get(key);
    if (hit) return hit;
    const t0 = performance.now();
    const res = await opts.engine.systemOne({ a, b, kind: "memory" }, D1_QUESTION);
    const elapsed = performance.now() - t0;
    const pTrue = (res.answers.same as NoulAnswer).noul;
    let out: SimilarityResult;
    if (elapsed > budgetMs) {
      // over budget: still return the score (a decision was made) but never cache-expand;
      // the harness treats slow-but-answered the same as answered.
      out = conformal ? toSimilarity(pTrue, conformal.set(pTrue)) : { score: pTrue };
    } else {
      out = conformal ? toSimilarity(pTrue, conformal.set(pTrue)) : { score: pTrue };
    }
    cache.set(key, out);
    return out;
  };
}

/** D2 — create-gate. Abstain ⇒ escalate to the agent (never silently accept/reject). */
export function createCreateGate(opts: CompanionOptions) {
  const conformal = opts.noulCalibration ? new NoulConformal().fit(opts.noulCalibration.p, opts.noulCalibration.labels, opts.alpha ?? 0.1) : null;
  return async function createGate(delta: string, evidence?: string): Promise<{ pTrue: number; abstain: boolean; escalate: boolean; calibrated: boolean }> {
    const res = await opts.engine.systemOne({ delta, evidence: evidence ?? null }, D2_QUESTION);
    const pTrue = (res.answers.durable as NoulAnswer).noul;
    const set = conformal?.set(pTrue);
    // singleton (falseIn !== trueIn) ⇒ act; multi/empty set ⇒ abstain ⇒ escalate (rule zero)
    const abstain = set ? set[0] === set[1] : false;
    return { pTrue, abstain, escalate: abstain, calibrated: conformal !== null };
  };
}

/** D3 — importance re-score (unbounded budget; batch job). */
export function createImportanceRescorer(opts: CompanionOptions) {
  const conformal = opts.scoreCalibration ? new ScoreConformal().fit(opts.scoreCalibration.pred, opts.scoreCalibration.truth, opts.alpha ?? 0.1) : null;
  return async function rescore(items: readonly string[]): Promise<Array<{ score: number; interval?: { lo: number; hi: number } }>> {
    if (items.length === 0) return [];
    const answers = (await opts.engine.batchQuestion(items.map((content) => ({ item: content })), D3_QUESTION)) as ScoreAnswer[];
    return answers.map((a) => ({ score: a.score, interval: conformal ? conformal.interval(a.score) : undefined }));
  };
}

export interface RelevanceOutcome {
  /** true ⇒ harness must use its importance-order fallback (contract §3) */
  fallback: boolean;
  reason: string;
  scores?: number[];
  elapsedMs?: number;
}

/**
 * D4 — injection relevance, ≤100 ms TOTAL at turn start. The measured viability is
 * ≤3 items on multilingual-int8 (BENCHMARKS.md); more items ⇒ immediate fallback
 * without spending the budget on a doomed call.
 */
export function createInjectionRelevance(opts: { engine: EngineLike; maxItems?: number; budgetMs?: number }) {
  const maxItems = opts.maxItems ?? 3;
  const budgetMs = opts.budgetMs ?? 100;
  return async function relevance(items: readonly string[], task: string): Promise<RelevanceOutcome> {
    if (items.length === 0) return { fallback: true, reason: "no items" };
    if (items.length > maxItems) {
      return { fallback: true, reason: `${items.length} items > ${maxItems} (measured CPU viability) — importance-order fallback` };
    }
    const t0 = performance.now();
    const answers = (await opts.engine.batchQuestion(items.map((content) => ({ item: content, task })), D4_QUESTION)) as ScoreAnswer[];
    const elapsedMs = performance.now() - t0;
    if (elapsedMs > budgetMs) {
      return { fallback: true, reason: `over budget (${elapsedMs.toFixed(0)} ms > ${budgetMs} ms)`, elapsedMs };
    }
    return { fallback: false, reason: `${items.length} item(s) scored in ${elapsedMs.toFixed(0)} ms`, scores: answers.map((a) => a.score), elapsedMs };
  };
}
