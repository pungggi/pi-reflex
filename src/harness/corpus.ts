/**
 * CONTRACT-harness.md §4 — calibration-corpus loader (pure Node, no native deps).
 *
 * Reads the JSONL emitted by pi-continual-harness ≥ 0.11.0
 * (`/harness export-corpus [path]` / `buildCorpus()`):
 *
 *   dedupe-pairs.jsonl  { v, kind:"dedupe_pair", a, b, label:"dup"|"not_dup",
 *                         similarity?, source?, needs_review? (not_dup only), meta? }
 *   lifecycle.jsonl     { v, kind:"lifecycle", item_kind, content_hash:"sha256:…",
 *                         event: created|cited|pruned|dropped|kept|deleted, importance? }
 *
 * Strict schema validation with field-level errors (file:line included) — the loader
 * must be ready before real exports arrive, and bad records must fail loudly, never
 * silently corrupt a calibration set.
 */
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

export type DedupeLabel = "dup" | "not_dup";
export type LifecycleEvent = "created" | "cited" | "pruned" | "dropped" | "kept" | "deleted";

export const LIFECYCLE_EVENTS: readonly LifecycleEvent[] = ["created", "cited", "pruned", "dropped", "kept", "deleted"];

export interface DedupePairMeta {
  kinds?: string[];
  owners?: string[];
  [key: string]: unknown;
}

export interface DedupePairRecord {
  v: 1;
  kind: "dedupe_pair";
  /** pair contents are carried (training needs them) */
  a: string;
  b: string;
  label: DedupeLabel;
  /** observed similarity in [0,1] (e.g. the Jaccard the planner saw) */
  similarity?: number;
  /** e.g. "tokenOverlap" */
  source?: string;
  /** set on not_dup only (planner declined a same-key-field pair) */
  needs_review?: boolean;
  meta?: DedupePairMeta;
}

export interface LifecycleRecord {
  v: 1;
  kind: "lifecycle";
  item_kind: string;
  /** "sha256:…" — lifecycle contents are hashed, never carried */
  content_hash: string;
  event: LifecycleEvent;
  /** importance in [0,1] at event time */
  importance?: number;
}

export interface CorpusStats {
  pairs: {
    total: number;
    byLabel: Record<DedupeLabel, number>;
    needsReview: number;
  };
  lifecycle: {
    total: number;
    byEvent: Record<LifecycleEvent, number>;
  };
}

export interface Corpus {
  dir: string;
  pairs: DedupePairRecord[];
  lifecycle: LifecycleRecord[];
  stats(): CorpusStats;
}

export const PAIRS_FILE = "dedupe-pairs.jsonl";
export const LIFECYCLE_FILE = "lifecycle.jsonl";

class SchemaError extends Error {}

function fail(file: string, lineNo: number, field: string, detail: string): never {
  throw new SchemaError(`${file}:${lineNo} — field '${field}': ${detail}`);
}

function expectObject(raw: string, file: string, lineNo: number): Record<string, unknown> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    throw new SchemaError(`${file}:${lineNo} — invalid JSON: ${(e as Error).message}`);
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    fail(file, lineNo, "(record)", "expected a JSON object");
  }
  return parsed as Record<string, unknown>;
}

function expectString(rec: Record<string, unknown>, file: string, lineNo: number, field: string, opts: { nonEmpty?: boolean; prefix?: string } = {}): string {
  const v = rec[field];
  if (typeof v !== "string") fail(file, lineNo, field, `expected string, got ${v === undefined ? "missing" : typeof v}`);
  if (opts.nonEmpty && v.length === 0) fail(file, lineNo, field, "must not be empty");
  if (opts.prefix && !v.startsWith(opts.prefix)) fail(file, lineNo, field, `must start with '${opts.prefix}'`);
  return v;
}

function expectOptionalNumber(rec: Record<string, unknown>, file: string, lineNo: number, field: string, min: number, max: number): number | undefined {
  const v = rec[field];
  if (v === undefined) return undefined;
  if (typeof v !== "number" || !Number.isFinite(v)) fail(file, lineNo, field, `expected finite number, got ${String(v)}`);
  if (v < min || v > max) fail(file, lineNo, field, `expected number in [${min}, ${max}], got ${v}`);
  return v;
}

function parseDedupePair(raw: string, file: string, lineNo: number): DedupePairRecord {
  const rec = expectObject(raw, file, lineNo);
  if (rec.v !== 1) fail(file, lineNo, "v", `expected 1, got ${String(rec.v)}`);
  if (rec.kind !== "dedupe_pair") fail(file, lineNo, "kind", `expected "dedupe_pair", got ${JSON.stringify(rec.kind) ?? "missing"}`);
  const a = expectString(rec, file, lineNo, "a");
  const b = expectString(rec, file, lineNo, "b");
  const label = rec.label;
  if (label !== "dup" && label !== "not_dup") {
    fail(file, lineNo, "label", `expected "dup" | "not_dup", got ${JSON.stringify(label) ?? "missing"}`);
  }
  const similarity = expectOptionalNumber(rec, file, lineNo, "similarity", 0, 1);
  let source: string | undefined;
  if (rec.source !== undefined) {
    if (typeof rec.source !== "string") fail(file, lineNo, "source", `expected string, got ${typeof rec.source}`);
    source = rec.source;
  }
  let needsReview: boolean | undefined;
  if (rec.needs_review !== undefined) {
    if (typeof rec.needs_review !== "boolean") fail(file, lineNo, "needs_review", `expected boolean, got ${typeof rec.needs_review}`);
    if (label !== "not_dup") {
      // contract §4: "set on not_dup only" — ANY presence on a dup record is a schema violation
      fail(file, lineNo, "needs_review", "may only be set on not_dup pairs (contract §4)");
    }
    needsReview = rec.needs_review;
  }
  let meta: DedupePairMeta | undefined;
  if (rec.meta !== undefined) {
    if (rec.meta === null || typeof rec.meta !== "object" || Array.isArray(rec.meta)) {
      fail(file, lineNo, "meta", "expected object");
    }
    meta = rec.meta as DedupePairMeta;
  }
  return { v: 1, kind: "dedupe_pair", a, b, label, similarity, source, needs_review: needsReview, meta };
}

function parseLifecycle(raw: string, file: string, lineNo: number): LifecycleRecord {
  const rec = expectObject(raw, file, lineNo);
  if (rec.v !== 1) fail(file, lineNo, "v", `expected 1, got ${String(rec.v)}`);
  if (rec.kind !== "lifecycle") fail(file, lineNo, "kind", `expected "lifecycle", got ${JSON.stringify(rec.kind) ?? "missing"}`);
  const itemKind = expectString(rec, file, lineNo, "item_kind", { nonEmpty: true });
  const contentHash = expectString(rec, file, lineNo, "content_hash", { prefix: "sha256:" });
  const event = rec.event;
  if (typeof event !== "string" || !LIFECYCLE_EVENTS.includes(event as LifecycleEvent)) {
    fail(file, lineNo, "event", `expected one of ${LIFECYCLE_EVENTS.map((e) => JSON.stringify(e)).join(" | ")}, got ${JSON.stringify(event) ?? "missing"}`);
  }
  const importance = expectOptionalNumber(rec, file, lineNo, "importance", 0, 1);
  return { v: 1, kind: "lifecycle", item_kind: itemKind, content_hash: contentHash, event: event as LifecycleEvent, importance };
}

/** Parse one JSONL kind from raw text. Blank lines are tolerated (trailing newlines). */
function parseJsonl<T>(text: string, file: string, parse: (raw: string, file: string, lineNo: number) => T): T[] {
  const out: T[] = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    if (!line) continue;
    out.push(parse(line, file, i + 1));
  }
  return out;
}

/**
 * Load a corpus directory. Either file may be absent (exports grow incrementally);
 * a missing directory or a directory with neither file throws.
 */
export function loadCorpus(dir: string): Corpus {
  if (!existsSync(dir) || !statSync(dir).isDirectory()) {
    throw new Error(`corpus directory not found: ${dir}`);
  }
  const pairsPath = join(dir, PAIRS_FILE);
  const lifecyclePath = join(dir, LIFECYCLE_FILE);
  const hasPairs = existsSync(pairsPath);
  const hasLifecycle = existsSync(lifecyclePath);
  if (!hasPairs && !hasLifecycle) {
    throw new Error(`corpus directory contains neither ${PAIRS_FILE} nor ${LIFECYCLE_FILE}: ${dir}`);
  }

  const pairs = hasPairs ? parseJsonl(readFileSync(pairsPath, "utf8"), PAIRS_FILE, parseDedupePair) : [];
  const lifecycle = hasLifecycle ? parseJsonl(readFileSync(lifecyclePath, "utf8"), LIFECYCLE_FILE, parseLifecycle) : [];

  return {
    dir,
    pairs,
    lifecycle,
    stats() {
      const byLabel: Record<DedupeLabel, number> = { dup: 0, not_dup: 0 };
      let needsReview = 0;
      for (const p of pairs) {
        byLabel[p.label]++;
        if (p.needs_review) needsReview++;
      }
      const byEvent = Object.fromEntries(LIFECYCLE_EVENTS.map((e) => [e, 0])) as Record<LifecycleEvent, number>;
      for (const l of lifecycle) byEvent[l.event]++;
      return {
        pairs: { total: pairs.length, byLabel, needsReview },
        lifecycle: { total: lifecycle.length, byEvent },
      };
    },
  };
}

/** Round-trip helper for exporters/tests: serialize records back to a JSONL string. */
export function toJsonl(records: readonly object[]): string {
  return records.map((r) => JSON.stringify(r)).join("\n") + (records.length ? "\n" : "");
}
