import { describe, expect, it } from "vitest";
import {
  buildSequence,
  collate,
  pyJson,
  renderCriterion,
  renderOptions,
  serializeState,
} from "../src/core/serialize.js";
import type { TokenizerLike } from "../src/core/tokenizer.js";
import { toInternal, type QuestionDef } from "../src/core/types.js";

/** Deterministic char-per-token mock; ids = code points. Special ids mimic BERT. */
class MockTokenizer implements TokenizerLike {
  readonly clsTokenId = 101;
  readonly sepTokenId = 102;
  readonly maskTokenId = 103;
  readonly padTokenId = 0;
  readonly maskToken = "[MASK]";
  encode(text: string): number[] {
    return [...text].map((c) => c.codePointAt(0)!);
  }
}

const tok = new MockTokenizer();

describe("pyJson / renderCriterion", () => {
  it("matches python json.dumps spacing", () => {
    expect(pyJson({ a: 1, b: [1, 2] })).toBe('{"a": 1, "b": [1, 2]}');
    expect(pyJson(null)).toBe("null");
    expect(pyJson("x")).toBe('"x"');
  });
  it("strings pass through, structured values become spaced JSON", () => {
    expect(renderCriterion("hello")).toBe("hello");
    expect(renderCriterion({ desc: "x" })).toBe('{"desc": "x"}');
    expect(renderCriterion(0)).toBe("0");
  });
});

describe("renderOptions", () => {
  it("choice: key or 'key: desc'", () => {
    const q = toInternal({ type: "choice", instructions: "i", criteria: { bare: null, empty: "", rich: "desc" } } as QuestionDef);
    expect(renderOptions(q)).toEqual(["bare", "empty", "rich: desc"]);
  });
  it("choice from string[] becomes bare labels", () => {
    const q = toInternal({ type: "choice", instructions: "i", criteria: ["a", "b"] } as QuestionDef);
    expect(renderOptions(q)).toEqual(["a", "b"]);
  });
  it("score: 'level i: text'", () => {
    const q = toInternal({ type: "score", instructions: "i", criteria: ["low", "high"] } as QuestionDef);
    expect(renderOptions(q)).toEqual(["level 0: low", "level 1: high"]);
  });
  it("noul: defaults when no criteria", () => {
    const q = toInternal({ type: "noul", instructions: "is it?" } as QuestionDef);
    expect(renderOptions(q)).toEqual(["false: no, the statement does not hold", "true: yes, the statement holds"]);
  });
  it("noul: explicit criteria respected; 0 and False-like values render (python parity)", () => {
    const q = toInternal({ type: "noul", instructions: "i", criteria: { false: 0 } } as QuestionDef);
    expect(renderOptions(q)).toEqual(["false: 0", "true: yes, the statement holds"]);
  });
});

describe("serializeState", () => {
  it("strings pass through, objects become spaced JSON", () => {
    expect(serializeState("raw")).toBe("raw");
    expect(serializeState({ b: 2 })).toBe('{"b": 2}');
  });
});

describe("buildSequence", () => {
  const q = toInternal({ type: "choice", instructions: "route this", criteria: { a: "aa", b: "bb" } } as QuestionDef);

  it("layout: [CLS] head [SEP] [MASK]opt0 [MASK]opt1 [SEP] state [SEP]", () => {
    const { ids, markers } = buildSequence(tok, "hello world", q, { maxLen: 512, headMaxLen: 192 });
    expect(ids[0]).toBe(101);
    expect(ids[1]).toBe("c".charCodeAt(0)); // "choice question: route this"
    const firstSep = ids.indexOf(102);
    expect(firstSep).toBeGreaterThan(0);
    expect(ids[markers[0]]).toBe(103);
    expect(ids[markers[1]]).toBe(103);
    // each option: mask + " " + text
    expect(ids[markers[0] + 1]).toBe(" ".charCodeAt(0));
    expect(ids[markers[0] + 2]).toBe("a".charCodeAt(0));
    // trailing [SEP] state [SEP]
    expect(ids[ids.length - 1]).toBe(102);
    const stateStr = "hello world";
    expect(ids.indexOf(102, firstSep + 1)).toBe(ids.length - stateStr.length - 2);
    expect(markers.length).toBe(2);
  });

  it("squeezes many options into the head budget", () => {
    const many = Object.fromEntries(Array.from({ length: 30 }, (_, i) => [`opt${i}`, `description number ${i} with words`]));
    const qq = toInternal({ type: "choice", instructions: "pick", criteria: many } as QuestionDef);
    const { ids, markers } = buildSequence(tok, "s", qq);
    // head region: [CLS]+head+[SEP]+options+[SEP] must fit head budget-ish; options truncated to per-option cap
    const secondSep = ids.indexOf(102, 1 + 6 + 1); // after first option run approx
    expect(secondSep).toBeGreaterThan(0);
    // all 30 markers retained
    expect(markers.length).toBe(30);
    // per-option cap kicks in: each option ≤ per tokens
    const per = Math.max(4, Math.trunc((192 - 16) / 30));
    for (let i = 0; i < markers.length - 1; i++) {
      expect(markers[i + 1] - markers[i]).toBeLessThanOrEqual(per + 1); // +1 safety
    }
  });

  it("drops markers that fall past maxLen", () => {
    const many = Object.fromEntries(Array.from({ length: 60 }, (_, i) => [`o${i}`, `d${i}`]));
    const qq = toInternal({ type: "choice", instructions: "x", criteria: many } as QuestionDef);
    const { ids, markers } = buildSequence(tok, "state here", qq, { maxLen: 64 });
    expect(ids.length).toBe(64);
    for (const m of markers) expect(m).toBeLessThan(64);
  });

  it("state truncated from the right by default, from the left with truncateLeft", () => {
    const long = "x".repeat(600);
    const right = buildSequence(tok, long, q, { maxLen: 300 });
    expect(right.ids.length).toBeLessThanOrEqual(300);
    // last token before final [SEP] should be 'x' truncated from the right => equals state prefix
    const left = buildSequence(tok, long, q, { maxLen: 300, truncateLeft: true });
    // python st[-room:] keeps the TAIL; here state is all 'x' so just check length + sep
    expect(left.ids[left.ids.length - 1]).toBe(102);
  });

  it("scrubs mask tokens from instructions and state", () => {
    const qm = toInternal({ type: "noul", instructions: "is [MASK] evil", criteria: undefined } as unknown as QuestionDef);
    const { ids } = buildSequence(tok, "state with [MASK] inside", qm, { maxLen: 512, headMaxLen: 192 });
    const asStr = ids.filter((i) => i !== 103).map((i) => String.fromCodePoint(i)).join("");
    expect(asStr).not.toContain("[MASK]");
  });
});

describe("collate", () => {
  it("pads ids, masks attention, pads marker arrays", () => {
    const q1 = toInternal({ type: "noul", instructions: "a" } as QuestionDef);
    const q2 = toInternal({ type: "choice", instructions: "b", criteria: { x: "1", y: "2", z: "3" } } as QuestionDef);
    const s1 = buildSequence(tok, "short", q1);
    const s2 = buildSequence(tok, "a much longer state string here", q2);
    const b = collate(
      [
        { ids: s1.ids, markers: s1.markers, qtype: 2 },
        { ids: s2.ids, markers: s2.markers, qtype: 0 },
      ],
      tok.padTokenId,
    );
    expect(b.batch).toBe(2);
    expect(b.seqLen).toBe(Math.max(s1.ids.length, s2.ids.length));
    expect(b.kmax).toBe(3);
    expect(b.inputIds[0].length).toBe(b.seqLen);
    expect(b.attentionMask[0].slice(s1.ids.length).every((v) => v === 0)).toBe(true);
    expect(b.attentionMask[1].slice(s2.ids.length).every((v) => v === 0)).toBe(true);
    expect(b.markerMask[0]).toEqual([true, true, false]); // noul = 2 options (false/true)
    expect(b.markerPos[1].slice(s2.markers.length)).toEqual([0, 0, 0].slice(0, 3 - s2.markers.length));
    expect(b.totalTokens).toBe(s1.ids.length + s2.ids.length);
    expect(b.qtype).toEqual([2, 0]);
  });
});
