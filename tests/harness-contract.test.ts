import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { loadCorpus, LIFECYCLE_EVENTS, toJsonl, type DedupePairRecord, type LifecycleRecord } from "../src/harness/corpus.js";
import { toSimilarity, toSimilarityFromSet } from "../src/harness/similarity.js";
import { NoulConformal } from "../src/core/conformal.js";

/* ------------------------------------------------------------------ §2 seam */

describe("CONTRACT §2 — toSimilarity seam adapter", () => {
  it("singleton {true} → act with the calibrated score", () => {
    const r = toSimilarity(0.87, [false, true]);
    expect(r).toEqual({ score: 0.87 });
    expect(r.abstain).toBeUndefined();
  });
  it("singleton {false} → act with low score (harness threshold decides no-merge)", () => {
    const r = toSimilarity(0.12, [true, false]);
    expect(r).toEqual({ score: 0.12 });
    expect(r.abstain).toBeUndefined();
  });
  it("{true,false} → abstain (keep both)", () => {
    expect(toSimilarity(0.5, [true, true])).toEqual({ score: 0.5, abstain: true });
  });
  it("empty set → abstain (keep both)", () => {
    expect(toSimilarity(0.5, [false, false])).toEqual({ score: 0.5, abstain: true });
  });
  it("score is rounded to 4dp", () => {
    expect(toSimilarity(0.87123, [false, true]).score).toBe(0.8712);
  });
  it("rejects non-probabilities", () => {
    expect(() => toSimilarity(Number.NaN, [false, true])).toThrow(RangeError);
    expect(() => toSimilarity(1.2, [false, true])).toThrow(RangeError);
    expect(() => toSimilarity(-0.1, [false, true])).toThrow(RangeError);
  });
  it("end-to-end with a fitted NoulConformal (rule zero mapping)", () => {
    // sharp calibration data → tight q̂ → high-confidence posteriors act
    const calib = [0.98, 0.02, 0.97, 0.03, 0.99, 0.01, 0.96, 0.04, 0.98, 0.02, 0.97, 0.03];
    const labels = [true, false, true, false, true, false, true, false, true, false, true, false];
    const noul = new NoulConformal().fit(calib, labels, 0.1);
    const actTrue = toSimilarityFromSet(0.98, (p) => noul.set(p));
    expect(actTrue.abstain).toBeUndefined(); // singleton {true} → act
    expect(actTrue.score).toBe(0.98);
  });
});

/* ------------------------------------------------------------------ §4 corpus */

const VALID_DIR = join(__dirname, "fixtures", "corpus", "valid");

function tmpCorpus(files: Record<string, string>): string {
  const dir = mkdtempSync(join(tmpdir(), "pi-reflex-corpus-"));
  for (const [name, content] of Object.entries(files)) writeFileSync(join(dir, name), content, "utf8");
  return dir;
}

describe("CONTRACT §4 — loadCorpus valid fixture", () => {
  it("loads both files with typed records", () => {
    const c = loadCorpus(VALID_DIR);
    expect(c.pairs.length).toBe(4);
    expect(c.lifecycle.length).toBe(5);
    const p0 = c.pairs[0] as DedupePairRecord;
    expect(p0).toMatchObject({ v: 1, kind: "dedupe_pair", label: "dup", similarity: 0.71, source: "tokenOverlap" });
    expect(p0.needs_review).toBeUndefined();
    const l4 = c.lifecycle[4] as LifecycleRecord;
    expect(l4).toMatchObject({ kind: "lifecycle", event: "deleted" });
    expect(l4.importance).toBeUndefined();
  });

  it("stats() counts by label/event + needsReview", () => {
    const s = loadCorpus(VALID_DIR).stats();
    expect(s.pairs).toEqual({ total: 4, byLabel: { dup: 2, not_dup: 2 }, needsReview: 2 });
    expect(s.lifecycle.total).toBe(5);
    expect(s.lifecycle.byEvent).toEqual({ created: 1, cited: 1, pruned: 1, dropped: 0, kept: 1, deleted: 1 });
    expect(Object.keys(s.lifecycle.byEvent).sort()).toEqual([...LIFECYCLE_EVENTS].sort());
  });

  it("toJsonl round-trips through loadCorpus", () => {
    const c = loadCorpus(VALID_DIR);
    const dir = tmpCorpus({
      "dedupe-pairs.jsonl": toJsonl(c.pairs),
      "lifecycle.jsonl": toJsonl(c.lifecycle),
    });
    try {
      const c2 = loadCorpus(dir);
      expect(c2.pairs).toEqual(c.pairs);
      expect(c2.lifecycle).toEqual(c.lifecycle);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("CONTRACT §4 — strict schema validation (field-level errors)", () => {
  function expectCorpusError(files: Record<string, string>, pattern: RegExp): void {
    const dir = tmpCorpus(files);
    try {
      expect(() => loadCorpus(dir)).toThrow(pattern);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }

  it("bad JSON → file:line + parse detail", () => {
    expectCorpusError({ "dedupe-pairs.jsonl": '{"v": 1, "kind": "dedupe_pair"\n'}, /dedupe-pairs\.jsonl:1 — invalid JSON/);
  });
  it("wrong schema version", () => {
    expectCorpusError(
      { "lifecycle.jsonl": '{"v": 2, "kind": "lifecycle", "item_kind": "m", "content_hash": "sha256:ab", "event": "kept"}' },
      /lifecycle\.jsonl:1 — field 'v': expected 1/,
    );
  });
  it("wrong kind for the file", () => {
    expectCorpusError({ "dedupe-pairs.jsonl": '{"v": 1, "kind": "lifecycle"}' }, /field 'kind': expected "dedupe_pair"/);
  });
  it("bad label", () => {
    expectCorpusError(
      { "dedupe-pairs.jsonl": '{"v": 1, "kind": "dedupe_pair", "a": "x", "b": "y", "label": "maybe"}' },
      /field 'label': expected "dup" \| "not_dup", got "maybe"/,
    );
  });
  it("needs_review on a dup pair violates the contract (any presence, true or false)", () => {
    expectCorpusError(
      { "dedupe-pairs.jsonl": '{"v": 1, "kind": "dedupe_pair", "a": "x", "b": "y", "label": "dup", "needs_review": true}' },
      /field 'needs_review': may only be set on not_dup pairs/,
    );
    expectCorpusError(
      { "dedupe-pairs.jsonl": '{"v": 1, "kind": "dedupe_pair", "a": "x", "b": "y", "label": "dup", "needs_review": false}' },
      /field 'needs_review': may only be set on not_dup pairs/,
    );
  });
  it("similarity outside [0,1]", () => {
    expectCorpusError(
      { "dedupe-pairs.jsonl": '{"v": 1, "kind": "dedupe_pair", "a": "x", "b": "y", "label": "dup", "similarity": 1.4}' },
      /field 'similarity': expected number in \[0, 1\], got 1\.4/,
    );
  });
  it("content_hash without sha256: prefix", () => {
    expectCorpusError(
      { "lifecycle.jsonl": '{"v": 1, "kind": "lifecycle", "item_kind": "m", "content_hash": "md5:ab", "event": "kept"}' },
      /field 'content_hash': must start with 'sha256:'/,
    );
  });
  it("unknown lifecycle event (five-event exporters rejected)", () => {
    expectCorpusError(
      { "lifecycle.jsonl": '{"v": 1, "kind": "lifecycle", "item_kind": "m", "content_hash": "sha256:ab", "event": "merged"}' },
      /field 'event': expected one of/,
    );
  });
  it("importance outside [0,1]", () => {
    expectCorpusError(
      { "lifecycle.jsonl": '{"v": 1, "kind": "lifecycle", "item_kind": "m", "content_hash": "sha256:ab", "event": "kept", "importance": 2}' },
      /field 'importance'/,
    );
  });
  it("missing field names the field", () => {
    expectCorpusError(
      { "dedupe-pairs.jsonl": '{"v": 1, "kind": "dedupe_pair", "b": "y", "label": "dup"}' },
      /field 'a': expected string, got missing/,
    );
  });
  it("empty directory → clear error", () => {
    const dir = tmpCorpus({});
    try {
      expect(() => loadCorpus(dir)).toThrow(/neither dedupe-pairs\.jsonl nor lifecycle\.jsonl/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
  it("lifecycle-only corpus is valid (exports grow incrementally)", () => {
    const dir = tmpCorpus({ "lifecycle.jsonl": '{"v": 1, "kind": "lifecycle", "item_kind": "m", "content_hash": "sha256:ab", "event": "kept"}' });
    try {
      const c = loadCorpus(dir);
      expect(c.pairs).toEqual([]);
      expect(c.stats().lifecycle.total).toBe(1);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
