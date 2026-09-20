/**
 * Python-parity JSON serialization — the single home for laya's JSON semantics.
 *
 * pyJson mirrors `json.dumps(x, ensure_ascii=False, separators=(", ", ": "), default=str)`.
 *
 * Documented divergence (see README): states/criteria must be JSON-clean.
 * NaN/Infinity serialize as "null" (Python prints NaN/Infinity); -0 prints as "0"
 * (Python prints "-0.0").
 */

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

/** laya `serialize_state`: strings pass through, everything else becomes spaced JSON. */
export function serializeState(state: unknown): string {
  if (typeof state === "string") return state;
  return pyJson(state);
}

/** laya `render_criterion`: strings pass through, structured values become spaced JSON. */
export function renderCriterion(value: unknown): string {
  if (typeof value === "string") return value;
  return pyJson(value);
}

const INT_KEY = /^(0|[1-9]\d*)$/;

/**
 * H3 guard: JavaScript objects enumerate integer-like keys ("1", "2", "10") in
 * ascending numeric order FIRST, before string keys — Python dicts preserve
 * insertion order. When the two orders differ, choice-option column order diverges
 * from laya and answers silently misalign. Returns true when that would happen.
 */
export function hasKeyOrderDivergence(keys: readonly string[]): boolean {
  let prevInt = -1;
  let seenString = false;
  for (const k of keys) {
    if (INT_KEY.test(k)) {
      if (seenString) return true; // int key after a string key: JS moves it earlier
      const n = Number(k);
      if (n <= prevInt) return true; // not ascending: JS reorders
      prevInt = n;
    } else {
      seenString = true;
    }
  }
  return false;
}

/**
 * True when any key is an integer-like string. Unlike hasKeyOrderDivergence, this
 * works on JS-normalized key lists: ECMAScript reorders integer-like keys into
 * ascending order at the object level (object literals AND JSON.parse), so by the
 * time we can observe keys, original insertion order is unrecoverable. If any
 * choice label is integer-like, the option order may not match the user's intent
 * (python/laya preserves insertion order) — we warn and let them verify.
 */
export function hasIntegerLikeKeys(keys: readonly string[]): boolean {
  return keys.some((k) => INT_KEY.test(k));
}

let warnedKeyOrder = false;

/** Warn once per process when choice labels are integer-like (JS owns their order). */
export function warnIntegerKeyOrder(keys: readonly string[], context: string): void {
  if (warnedKeyOrder || !hasIntegerLikeKeys(keys)) return;
  warnedKeyOrder = true;
  console.warn(
    `[pi-jev] ${context}: integer-like labels detected (${keys.join(", ")}). ` +
      "JavaScript orders them numerically regardless of insertion order (python/laya preserves " +
      "insertion order) — verify this matches your intended option order.",
  );
}
