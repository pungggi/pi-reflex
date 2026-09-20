/**
 * Exact port of laya's sequence construction (`common.py: build_sequence`).
 *
 * Format: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`
 * Markers are the positions of each option's leading [MASK]; the model pools exactly there.
 */
import type { InternalQuestion } from "./types.js";
import type { TokenizerLike } from "./tokenizer.js";

/** Python `json.dumps(x, ensure_ascii=False, separators=(", ", ": "))` equivalent. */
export function pyJson(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "number" || typeof v === "boolean") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(pyJson).join(", ") + "]";
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    return "{" + entries.map(([k, val]) => `${JSON.stringify(k)}: ${pyJson(val)}`).join(", ") + "}";
  }
  return JSON.stringify(String(v)); // python default=str
}

export function serializeState(state: unknown): string {
  if (typeof state === "string") return state;
  return pyJson(state);
}

/** laya `render_criterion`: strings pass through, structured values become spaced JSON. */
export function renderCriterion(value: unknown): string {
  if (typeof value === "string") return value;
  return pyJson(value);
}

/** laya `render_options`: option texts in label-index order. Noul is always [false, true]. */
export function renderOptions(q: InternalQuestion): string[] {
  if (q.t === "choice") {
    const crit = (q.crit ?? {}) as Record<string, unknown>;
    return Object.entries(crit).map(([k, v]) =>
      v === null || v === undefined || v === "" ? k : `${k}: ${renderCriterion(v)}`,
    );
  }
  if (q.t === "score") {
    const crit = (q.crit ?? []) as unknown[];
    return crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  }
  // noul
  const crit = (q.crit ?? {}) as { false?: unknown; true?: unknown };
  const fc = crit.false;
  const tc = crit.true;
  const f = fc === null || fc === undefined || fc === "" ? "no, the statement does not hold" : renderCriterion(fc);
  const t = tc === null || tc === undefined || tc === "" ? "yes, the statement holds" : renderCriterion(tc);
  return [`false: ${f}`, `true: ${t}`];
}

export interface SequenceOptions {
  maxLen?: number;
  headMaxLen?: number;
  optionOrder?: number[];
  truncateLeft?: boolean;
}

export interface BuiltSequence {
  ids: number[];
  markers: number[];
}

export function buildSequence(
  tok: TokenizerLike,
  state: unknown,
  q: InternalQuestion,
  opts: SequenceOptions = {},
): BuiltSequence {
  const maxLen = opts.maxLen ?? 512;
  const headMaxLen = opts.headMaxLen ?? 192;
  const maskTok = tok.maskToken;
  const scrub = (s: string) => s.split(maskTok).join(" ");

  const optsText = renderOptions(q);
  const order = opts.optionOrder ?? optsText.map((_, i) => i);

  let headIds = tok.encode(`${q.t} question: ${scrub(String(q.ins))}`);

  const optIds: number[][] = [];
  for (const i of order) {
    optIds.push([tok.maskTokenId, ...tok.encode(" " + scrub(optsText[i])).slice(0, 48)]);
  }

  let optBudget = headMaxLen - optIds.reduce((s, o) => s + o.length, 0);
  if (optBudget < 16) {
    const per = Math.max(4, Math.trunc((headMaxLen - 16) / Math.max(1, optIds.length)));
    for (let i = 0; i < optIds.length; i++) optIds[i] = optIds[i].slice(0, per);
    optBudget = headMaxLen - optIds.reduce((s, o) => s + o.length, 0);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));

  let ids: number[] = [tok.clsTokenId, ...headIds, tok.sepTokenId];
  const markers: number[] = [];
  for (const o of optIds) {
    markers.push(ids.length);
    ids = ids.concat(o);
  }
  ids.push(tok.sepTokenId);

  const room = Math.max(0, maxLen - ids.length - 1);
  let st = tok.encode(scrub(serializeState(state)));
  if (opts.truncateLeft) {
    // NOTE: replicates python `st[-room:]`, which for room == 0 returns the WHOLE string.
    st = room === 0 ? st : st.slice(-room);
  } else {
    st = st.slice(0, room);
  }
  ids = ids.concat(st, [tok.sepTokenId]);

  ids = ids.slice(0, maxLen);
  return { ids, markers: markers.filter((m) => m < maxLen) };
}

export interface CollatedItem {
  ids: number[];
  markers: number[];
  qtype: number;
}

export interface CollatedBatch {
  inputIds: number[][];
  attentionMask: number[][];
  markerPos: number[][];
  markerMask: boolean[][];
  qtype: number[];
  batch: number;
  seqLen: number;
  kmax: number;
  totalTokens: number;
}

/** laya `collate_items` for a single group of question items. */
export function collate(items: CollatedItem[], padId: number): CollatedBatch {
  const n = items.length;
  const L = Math.max(...items.map((it) => it.ids.length));
  const kmax = Math.max(...items.map((it) => it.markers.length));
  const inputIds: number[][] = [];
  const attentionMask: number[][] = [];
  const markerPos: number[][] = [];
  const markerMask: boolean[][] = [];
  let totalTokens = 0;

  for (const it of items) {
    const ids = it.ids.concat(Array(L - it.ids.length).fill(padId));
    inputIds.push(ids);
    const att = Array(it.ids.length).fill(1).concat(Array(L - it.ids.length).fill(0));
    attentionMask.push(att);
    totalTokens += it.ids.length;
    const k = it.markers.length;
    markerPos.push(it.markers.concat(Array(kmax - k).fill(0)));
    markerMask.push(Array(k).fill(true).concat(Array(kmax - k).fill(false)));
  }

  return { inputIds, attentionMask, markerPos, markerMask, qtype: items.map((it) => it.qtype), batch: n, seqLen: L, kmax, totalTokens };
}
