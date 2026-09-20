/**
 * Dependency-free language/script detection — port of laya `lang.py`.
 *
 * Routing only needs one decision: is this English Latin text, or something the
 * English checkpoint cannot read? Script detection is exact; the Latin-script
 * language guess is a stopword/diacritic heuristic and explicitly best-effort.
 */

// Unicode blocks the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
const SCRIPT_RANGES: Array<[string, Array<[number, number]>]> = [
  ["greek", [[0x0370, 0x03ff], [0x1f00, 0x1fff]]],
  ["cyrillic", [[0x0400, 0x052f], [0x2de0, 0x2dff], [0xa640, 0xa69f]]],
  ["hebrew", [[0x0590, 0x05ff]]],
  ["arabic", [[0x0600, 0x06ff], [0x0750, 0x077f], [0x08a0, 0x08ff], [0xfb50, 0xfdff], [0xfe70, 0xfeff]]],
  ["devanagari", [[0x0900, 0x097f], [0xa8e0, 0xa8ff]]],
  ["bengali", [[0x0980, 0x09ff]]],
  ["gurmukhi", [[0x0a00, 0x0a7f]]],
  ["gujarati", [[0x0a80, 0x0aff]]],
  ["oriya", [[0x0b00, 0x0b7f]]],
  ["tamil", [[0x0b80, 0x0bff]]],
  ["telugu", [[0x0c00, 0x0c7f]]],
  ["kannada", [[0x0c80, 0x0cff]]],
  ["malayalam", [[0x0d00, 0x0d7f]]],
  ["sinhala", [[0x0d80, 0x0dff]]],
  ["thai", [[0x0e00, 0x0e7f]]],
  ["lao", [[0x0e80, 0x0eff]]],
  ["tibetan", [[0x0f00, 0x0fff]]],
  ["myanmar", [[0x1000, 0x109f]]],
  ["georgian", [[0x10a0, 0x10ff]]],
  ["ethiopic", [[0x1200, 0x137f]]],
  ["khmer", [[0x1780, 0x17ff]]],
  ["hangul", [[0x1100, 0x11ff], [0x3130, 0x318f], [0xac00, 0xd7af]]],
  ["kana", [[0x3040, 0x309f], [0x30a0, 0x30ff], [0x31f0, 0x31ff]]],
  ["han", [[0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xf900, 0xfaff]]],
];

const STOP: Record<string, Set<string>> = {
  en: new Set("the and is are was were to of in for with that this it you have has not but on at be as from will can would there their what which please we i".split(" ")),
  fr: new Set("le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais sont ont aux ce".split(" ")),
  de: new Set("der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden wurde haben sind oder aber".split(" ")),
  es: new Set("el los las que por con para una es se del como pero son está este esta todo más muy hay sus".split(" ")),
  pt: new Set("os as que em um uma para com não é se do da dos das mas são está este esta muito pelo pela".split(" ")),
  it: new Set("il lo gli che di per con non è si del della sono questo questa anche come più nella alla".split(" ")),
  nl: new Set("het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt".split(" ")),
};

const NON_EN_DIACRITICS = new Set("àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö");
const IS_LETTER = /\p{L}/u;
const WORD_RE = /\p{L}+/gu;

export type StateInput = string | Record<string, unknown> | unknown[] | null | undefined;

function* iterText(state: StateInput, depth = 0): Generator<string> {
  if (depth > 6 || state === null || state === undefined) return;
  if (typeof state === "string") {
    yield state;
    return;
  }
  if (Array.isArray(state)) {
    for (const v of state as unknown[]) yield* iterText(v as StateInput, depth + 1);
    return;
  }
  if (typeof state === "object") {
    for (const v of Object.values(state as Record<string, unknown>)) yield* iterText(v as StateInput, depth + 1);
  }
}

/** Flatten a state into the text used for detection (keys ignored: usually English). */
export function stateText(state: StateInput, maxChars = 4000): string {
  let out = "";
  for (const s of iterText(state)) {
    out += (out ? " " : "") + s;
    if (out.length >= maxChars) break;
  }
  return out.slice(0, maxChars);
}

/** Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown'. */
export function detectScript(text: string): string {
  const counts: Record<string, number> = { latin: 0 };
  for (const ch of text) {
    if (!IS_LETTER.test(ch)) continue;
    const cp = ch.codePointAt(0)!;
    if (cp < 0x0250 || (0x1e00 <= cp && cp <= 0x1eff)) {
      counts.latin++;
      continue;
    }
    for (const [name, ranges] of SCRIPT_RANGES) {
      if (ranges.some(([lo, hi]) => lo <= cp && cp <= hi)) {
        counts[name] = (counts[name] ?? 0) + 1;
        break;
      }
    }
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) return "unknown";
  let bestName = "latin";
  let bestCount = -1;
  for (const [name, c] of Object.entries(counts)) {
    if (c > bestCount) {
      bestName = name;
      bestCount = c;
    }
  }
  return bestName;
}

/** Fraction of alphabetic characters belonging to each detected script. */
export function scriptProfile(text: string): Record<string, number> {
  const counts: Record<string, number> = { latin: 0 };
  for (const ch of text) {
    if (!IS_LETTER.test(ch)) continue;
    const cp = ch.codePointAt(0)!;
    if (cp < 0x0250 || (0x1e00 <= cp && cp <= 0x1eff)) {
      counts.latin++;
      continue;
    }
    for (const [name, ranges] of SCRIPT_RANGES) {
      if (ranges.some(([lo, hi]) => lo <= cp && cp <= hi)) {
        counts[name] = (counts[name] ?? 0) + 1;
        break;
      }
    }
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (!total) return {};
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(counts)) if (v) out[k] = v / total;
  return out;
}

/** Best-effort language code for Latin-script text, or null when undecided. */
export function guessLatinLanguage(text: string): string | null {
  const words = [...text.toLowerCase().matchAll(WORD_RE)].map((m) => m[0]);
  if (words.length < 4) return null;
  const scores: Record<string, number> = {};
  for (const [lg, sw] of Object.entries(STOP)) scores[lg] = words.filter((w) => sw.has(w)).length;
  const lowered = text.toLowerCase();
  let diac = 0;
  for (const ch of lowered) if (NON_EN_DIACRITICS.has(ch)) diac++;
  const diacRate = diac / Math.max(1, lowered.length);
  const en = scores.en ?? 0;
  let bestLg: string | null = null;
  let best = 0;
  for (const lg of Object.keys(STOP)) {
    if (lg === "en") continue;
    if (scores[lg] > best) {
      bestLg = lg;
      best = scores[lg];
    }
  }
  if (best === 0 && diacRate < 0.02) return en ? "en" : null;
  if (bestLg && best >= Math.max(2, en + 2)) return bestLg;
  if (diacRate >= 0.04 && bestLg && best >= en) return bestLg;
  return en ? "en" : null;
}

export interface Analysis {
  script: string;
  scriptProfile: Record<string, number>;
  language: string | null;
  isEnglish: boolean;
  nonLatinFraction: number;
}

export function analyse(state: StateInput): Analysis {
  const text = stateText(state);
  const prof = scriptProfile(text);
  const script = detectScript(text);
  const nonLatin = prof["latin"] !== undefined ? Math.round((1 - prof["latin"]) * 1e4) / 1e4 : 0;
  if (script === "unknown") {
    return { script: "unknown", scriptProfile: prof, language: null, isEnglish: true, nonLatinFraction: 0 };
  }
  if (script !== "latin") {
    return { script, scriptProfile: prof, language: null, isEnglish: false, nonLatinFraction: nonLatin };
  }
  const lang = guessLatinLanguage(text);
  return { script: "latin", scriptProfile: prof, language: lang, isEnglish: lang === null || lang === "en", nonLatinFraction: nonLatin };
}

export function isEnglish(state: StateInput): boolean {
  return analyse(state).isEnglish;
}
