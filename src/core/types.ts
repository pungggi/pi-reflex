/**
 * Public question/answer types — mirrors laya's `system_one` API shape.
 */

export type QuestionInstructions = string | Record<string, unknown> | unknown[];

export interface ChoiceDef {
  type: "choice";
  instructions: QuestionInstructions;
  /** label -> description (or null/"" for bare label). A string[] is accepted as bare labels. */
  criteria: Record<string, string | Record<string, unknown> | null> | string[];
}

export interface ScoreDef {
  type: "score";
  instructions: QuestionInstructions;
  /** ordered rubric levels, index 0..K-1 */
  criteria: unknown[];
}

export interface NoulDef {
  type: "noul";
  instructions: QuestionInstructions;
  /** optional explicit framing of false/true criteria */
  criteria?: { false?: unknown; true?: unknown };
}

export type QuestionDef = ChoiceDef | ScoreDef | NoulDef;
export type Questions = Record<string, QuestionDef>;

/** Internal normalized question (laya `_to_internal`). */
export interface InternalQuestion {
  t: "choice" | "score" | "noul";
  ins: string;
  crit: unknown;
}

export interface ActionExt {
  act_probability: number;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  action: ActionExt;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, unknown>;
  probabilities: Record<string, number>;
  confidence: number;
  action: ActionExt;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number;
  action: ActionExt;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface SystemOneResult {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

/** Flatten a laya-style state (string | object | array) into an internal question. */
export function toInternal(qdef: QuestionDef): InternalQuestion {
  const t = qdef.type;
  let crit: unknown = "criteria" in qdef ? qdef.criteria : undefined;
  if (t === "choice" && Array.isArray(crit)) {
    crit = Object.fromEntries((crit as unknown[]).map((c) => [String(c), null]));
  }
  const ins = qdef.instructions;
  return { t, ins: typeof ins === "string" ? ins : JSON.stringify(ins), crit };
}
