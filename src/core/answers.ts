/**
 * Answer assembly — exact port of laya `Agent.system_one` post-processing.
 * Takes option logits + act logits per question, applies bucketed temperature,
 * and builds the typed answer objects.
 */
import { calibratedSoftmax, confidenceFromProbs, QTYPES, round4, temperatureFor, type Temperatures } from "./calibration.js";
import type { Answer, InternalQuestion } from "./types.js";

export function softmax(v: number[]): number[] {
  const m = Math.max(...v);
  const exps = v.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/**
 * Build one answer from a row of option logits and a row of act logits.
 * Mirrors laya exactly: temperature bucket -> stable softmax -> entropy confidence -> typed shape.
 */
export function buildAnswer(q: InternalQuestion, optionLogits: number[], actLogits: number[], temps: Temperatures): Answer {
  const k = optionLogits.length;
  const qt = QTYPES[q.t];
  const tScale = temperatureFor(temps, qt, k);
  const p = calibratedSoftmax(optionLogits, k, tScale);

  const confScore = round4(confidenceFromProbs(p, k));
  const act = softmax(actLogits);
  const ext = { act_probability: round4(act[0]) };

  if (q.t === "choice") {
    const keys = Object.keys((q.crit ?? {}) as Record<string, unknown>);
    let best = 0;
    for (let i = 1; i < k; i++) if (p[i] > p[best]) best = i;
    const probabilities: Record<string, number> = {};
    keys.forEach((kk, i) => (probabilities[kk] = round4(p[i])));
    return { type: "choice", choice: keys[best], probabilities, confidence: confScore, action: ext };
  }

  if (q.t === "score") {
    let expScore = 0;
    for (let i = 0; i < k; i++) expScore += i * p[i];
    const crit = (q.crit ?? []) as unknown[];
    const legend: Record<string, unknown> = {};
    const probabilities: Record<string, number> = {};
    crit.forEach((c, i) => {
      legend[String(i)] = c;
      probabilities[String(i)] = round4(p[i]);
    });
    return { type: "score", score: round4(expScore), legend, probabilities, confidence: confScore, action: ext };
  }

  // noul
  return { type: "noul", noul: round4(p[1]), confidence: round4(Math.max(p[1], 1 - p[1])), action: ext };
}
