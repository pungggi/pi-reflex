/**
 * Calibration math — exact port of laya's temperature bucketing and entropy confidence.
 */

export const QTYPES = { choice: 0, score: 1, noul: 2 } as const;
export type QTypeName = keyof typeof QTYPES;
export const QTYPE_NAMES: QTypeName[] = ["choice", "score", "noul"];

/** laya `temp_bucket`: e.g. "choice:3-5", "noul:2", "score:11+". */
export function tempBucket(qtype: number, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qtype]}:${size}`;
}

/** laya `confidence_from_probs`: normalized Shannon entropy confidence 1 - H(p)/log(k). */
export function confidenceFromProbs(p: number[], k: number): number {
  if (k < 2) return 1.0;
  const pk = p.slice(0, k);
  let ent = 0;
  for (const v of pk) {
    const c = Math.min(Math.max(v, 1e-12), 1.0);
    ent -= v * Math.log(c);
  }
  const conf = 1.0 - ent / Math.log(k);
  return Math.min(Math.max(conf, 0.0), 1.0);
}

export interface Temperatures {
  /** per-qtype fallback temperature [choice, score, noul] */
  temperature: [number, number, number];
  /** per bucket overrides, e.g. { "choice:2": 1.19, "noul:2": 1.03 } */
  temperatureByOptions: Record<string, number>;
}

export function temperatureFor(t: Temperatures, qtype: number, k: number): number {
  return t.temperatureByOptions?.[tempBucket(qtype, k)] ?? t.temperature[qtype];
}

/** laya softmax over logits[:k] / max(1e-3, T), numerically stable. */
export function calibratedSoftmax(logits: number[], k: number, tScale: number): number[] {
  const denom = Math.max(1e-3, tScale);
  const z = logits.slice(0, k).map((v) => v / denom);
  const m = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** Python round(x, 4) approximation (half-away-from-zero; laya uses banker's — negligible drift). */
export function round4(x: number): number {
  return Math.round((x + Number.EPSILON) * 1e4) / 1e4;
}
