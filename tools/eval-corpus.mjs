// A4 eval harness — reflex (local) vs Cloudflare Clef (hosted) on harness dedupe pairs.
//
// Dev-only (tools/ is not shipped). Runs the D1 call-site shape from
// src/harness/companion.ts over CONTRACT-harness.md §4 corpus records and
// reports accuracy / ECE / AUC / latency, plus conformal coverage + abstain
// rate for the reflex arm (Clef has no conformal layer).
//
// Usage:
//   node tools/eval-corpus.mjs --synthetic 40                     # harness self-test (no corpus, no creds)
//   node tools/eval-corpus.mjs --corpus dedupe-pairs.jsonl         # reflex arm only
//   node tools/eval-corpus.mjs --corpus dedupe-pairs.jsonl --clef  # + @cf/cloudflare/clef (needs CLOUDFLARE_API_KEY + CLOUDFLARE_ACCOUNT_ID)
//   node tools/eval-corpus.mjs --corpus dedupe-pairs.jsonl --clef --clef-model @cf/cloudflare/clef-flash
//
// Clef arm uses the same REST transport pi's cloudflare-workers-ai provider
// uses (pi-ai api/cloudflare-workers-ai-system-one.ts): POST {base}/ai/run/
// with { model, input: { state, questions } }, answers as wire-level `noul`.
// Same state object + same D1 question for both arms — each system serializes
// its own way (reflex renders state; the Workers AI service sees raw JSON).
//
// Honesty: this is the A4 *protocol* runner. Numbers are smoke-grade until
// the corpus grows (see BENCHMARKS.md §A4) and the hosted arm needs paid creds.

import { Engine } from "../dist/engine/engine.js";
import { NoulConformal } from "../dist/core/conformal.js";
import fs from "node:fs";

// ── args ─────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
function arg(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i !== -1 && args[i + 1] ? args[i + 1] : fallback;
}
function has(name) {
  return args.includes(`--${name}`);
}

const CORPUS = arg("corpus");
const SYNTHETIC = Number(arg("synthetic", 0));
const ENGINE_NAME = arg("engine", "multilingual");
const CLEF_MODEL = arg("clef-model", "@cf/cloudflare/clef");
const CLEF = has("clef");
const LIMIT = Number(arg("limit", 200)); // hosted-arm cost guard (both arms)
const ALPHA = Number(arg("alpha", 0.1));
const CALIB_FRAC = Number(arg("calib-frac", 0.5));
const ARTIFACTS = arg("artifacts");

if (!CORPUS && !SYNTHETIC) {
  console.error("usage: node tools/eval-corpus.mjs --corpus <dedupe-pairs.jsonl> | --synthetic <n> [--clef] [--engine multilingual|english|typed-decisions]");
  process.exit(2);
}
// Cost guard must not be negative: slice(0, -N) would select everything EXCEPT the
// last N records and silently run thousands of paid hosted calls (review #3).
if (!Number.isFinite(LIMIT) || LIMIT < 1) {
  console.error(`--limit must be a positive integer (got ${arg("limit", "200")})`);
  process.exit(2);
}

// Deterministic RNG (contract invariant: same input ⇒ same result).
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

// ── records ──────────────────────────────────────────────────────────────────

/** CONTRACT-harness.md §4: { v, kind: "dedupe_pair", a, b, label: "dup"|"not_dup", meta? }.
 *  Records without non-empty string contents are counted as malformed — String(undefined)
 *  would otherwise create labeled "undefined"/"undefined" pairs that silently contaminate
 *  accuracy and conformal calibration (review #1). */
function loadCorpus(path) {
  const lines = fs.readFileSync(path, "utf8").split("\n").filter((l) => l.trim());
  const records = [];
  let skipped = 0;
  for (const line of lines) {
    try {
      const r = JSON.parse(line);
      const wellFormed =
        r.kind === "dedupe_pair" &&
        (r.label === "dup" || r.label === "not_dup") &&
        typeof r.a === "string" && r.a.trim().length > 0 &&
        typeof r.b === "string" && r.b.trim().length > 0;
      if (wellFormed) {
        records.push({ a: r.a, b: r.b, dup: r.label === "dup", source: r.source ?? "corpus" });
      } else skipped++;
    } catch {
      skipped++;
    }
  }
  return { records, skipped };
}

/** Synthetic pairs: harness self-test ONLY (obvious dups/non-dups; not a quality signal). */
function synthetic(n) {
  const pick = rng(42);
  const facts = [
    "pi-reflex needs ONNX artifacts downloaded before first use",
    "The release pipeline publishes via GitHub Actions OIDC",
    "Conformal prediction sets gate destructive dedupe decisions",
    "D4 injection relevance must fit a 100 ms turn-start budget",
    "The extension registers four decision tools under the reflex namespace",
    "int8 quantization is 1.3-1.6x faster than fp32 end-to-end",
    "Sessions resume with MCP tools rendered before the server connects",
    "The corpus exporter doubles as ongoing collection in the harness",
  ];
  const paraphrase = (s) => {
    const variants = [s, s.replace(/^The /, "A "), `${s}.`, s.toLowerCase(), s.replace(/ /g, "  ") /* spaced JSON style */];
    return variants[Math.floor(pick() * variants.length)];
  };
  const out = [];
  for (let i = 0; i < n; i++) {
    const dup = pick() < 0.5;
    const f = facts[Math.floor(pick() * facts.length)];
    if (dup) {
      out.push({ a: f, b: paraphrase(f), dup: true, source: "synthetic" });
    } else {
      let g = facts[Math.floor(pick() * facts.length)];
      while (g === f) g = facts[Math.floor(pick() * facts.length)];
      out.push({ a: f, b: g, dup: false, source: "synthetic" });
    }
  }
  return out;
}

// ── arms ─────────────────────────────────────────────────────────────────────

const D1_QUESTION = { same: { type: "noul", instructions: "Are these two items the same durable fact?" } };
/** The deployed D1 call site hardcodes kind: "memory" (src/harness/companion.ts) — the
 *  eval must feed the model the exact same state shape, NOT meta.kinds per record,
 *  or accuracy/conformal measure a different input than production (review #2). */
const D1_KIND = "memory";

async function reflexArm(records) {
  const url = ARTIFACTS ?? new URL(`../artifacts/${ENGINE_NAME}`, import.meta.url).pathname.replace(/^\/(\w:)/i, "$1");
  const engine = await Engine.fromArtifacts(url, { int8: true });
  const out = [];
  for (const r of records) {
    const t0 = performance.now();
    const res = await engine.systemOne({ a: r.a, b: r.b, kind: D1_KIND }, D1_QUESTION);
    out.push({ p: res.answers.same.noul, dup: r.dup, ms: performance.now() - t0, tokens: res.usage.input_tokens });
  }
  return out;
}

async function clefArm(records) {
  const key = process.env.CLOUDFLARE_API_KEY;
  const account = process.env.CLOUDFLARE_ACCOUNT_ID;
  if (!key || !account) throw new Error("--clef needs CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID");
  const base = `https://api.cloudflare.com/client/v4/accounts/${account}/ai/run/`;
  const out = [];
  for (const r of records) {
    const t0 = performance.now();
    const resp = await fetch(base, {
      method: "POST",
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: CLEF_MODEL,
        input: { state: { a: r.a, b: r.b, kind: D1_KIND }, questions: D1_QUESTION }, // noul is the wire type
      }),
    });
    const body = await resp.json();
    if (!resp.ok || body.success === false) throw new Error(`clef ${resp.status}: ${JSON.stringify(body.errors ?? body).slice(0, 300)}`);
    const result = body.result?.result ?? body.result; // third-party run record vs cloudflare-hosted direct
    const p = result?.answers?.same?.noul;
    if (typeof p !== "number") throw new Error(`clef returned no noul answer: ${JSON.stringify(body).slice(0, 300)}`);
    out.push({ p, dup: r.dup, ms: performance.now() - t0, tokens: result?.usage?.input_tokens ?? 0 });
  }
  return out;
}

// ── metrics ──────────────────────────────────────────────────────────────────

function quantile(sorted, q) {
  if (!sorted.length) return NaN; // zero-row inputs must not crash the report (review #4)
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

function metrics(rows) {
  const n = rows.length;
  if (n === 0) return { n: 0, accuracy: NaN, ece10: NaN, auc: NaN, p50ms: NaN, p95ms: NaN, tokens: 0, pos: 0, neg: 0 };
  const correct = rows.filter((r) => (r.p >= 0.5) === r.dup).length;
  // ECE, 10 equal-width bins on p(dup)
  const bins = Array.from({ length: 10 }, () => ({ n: 0, conf: 0, acc: 0 }));
  for (const r of rows) {
    const b = bins[Math.min(9, Math.floor(r.p * 10))];
    b.n++;
    b.conf += r.p;
    b.acc += r.dup ? 1 : 0;
  }
  const ece = bins.reduce((s, b) => (b.n ? s + (b.n / n) * Math.abs(b.acc / b.n - b.conf / b.n) : s), 0);
  // AUC via Mann-Whitney U (ties → 0.5)
  const pos = rows.filter((r) => r.dup).map((r) => r.p);
  const neg = rows.filter((r) => !r.dup).map((r) => r.p);
  let wins = 0;
  for (const p of pos) for (const q of neg) wins += p > q ? 1 : p === q ? 0.5 : 0;
  const auc = pos.length && neg.length ? wins / (pos.length * neg.length) : NaN;
  const ms = rows.map((r) => r.ms).sort((a, b) => a - b);
  return {
    n,
    accuracy: correct / n,
    ece10: ece,
    auc,
    p50ms: quantile(ms, 0.5),
    p95ms: quantile(ms, 0.95),
    tokens: rows.reduce((s, r) => s + r.tokens, 0),
    pos: pos.length,
    neg: neg.length,
  };
}

/** Conformal coverage + abstain on a held-out split (reflex arm only). */
function conformal(rows, alpha) {
  const shuffled = [...rows];
  const r = rng(7);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const cut = Math.max(1, Math.floor(shuffled.length * CALIB_FRAC));
  const calib = shuffled.slice(0, cut);
  const evalSet = shuffled.slice(cut);
  if (evalSet.length < 5) return { note: "too few pairs to split (need ≥10)" };
  const model = new NoulConformal().fit(calib.map((x) => x.p), calib.map((x) => x.dup), alpha);
  let covered = 0;
  let abstained = 0;
  let acted = 0;
  let actedCorrect = 0;
  for (const x of evalSet) {
    const [inF, inT] = model.set(x.p);
    const singleton = inF !== inT;
    if (singleton) {
      acted++;
      if ((inT && x.dup) || (inF && !x.dup)) actedCorrect++;
    } else abstained++;
    const truthInSet = x.dup ? inT : inF;
    if (truthInSet) covered++;
  }
  return {
    calib: calib.length,
    eval: evalSet.length,
    alpha,
    coverage: covered / evalSet.length,
    abstainRate: abstained / evalSet.length,
    actAccuracy: acted ? actedCorrect / acted : NaN,
    actedN: acted,
  };
}

// ── run ──────────────────────────────────────────────────────────────────────

const syntheticResult = synthetic(SYNTHETIC);
const records0 = SYNTHETIC ? { records: syntheticResult, skipped: 0 } : loadCorpus(CORPUS);
const records = records0.records.slice(0, LIMIT);
if (SYNTHETIC) console.log(`# synthetic self-test — ${records.length} generated pairs (harness validation ONLY, not a quality signal)`);
else console.log(`# corpus ${CORPUS} — ${records.length} usable dedupe pairs (${records0.skipped} malformed lines skipped)`);

// Empty export, all-skipped rows, or a limit below 1: report and exit cleanly
// instead of crashing on undefined quantiles (review #4).
if (records.length === 0) {
  console.log("! no usable records — export with /harness export-corpus or use --synthetic <n>");
  process.exit(0);
}

const fmtPct = (x) => (Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : "n/a");
const fmtNum = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : "n/a");
const report = { engine: ENGINE_NAME, model: null, clef: null };

{
  const rows = await reflexArm(records);
  const m = metrics(rows);
  const c = conformal(rows, ALPHA);
  report.model = { ...m, conformal: c };
  console.log(`\n## reflex/${ENGINE_NAME} (local, int8, CPU)`);
  console.log(`n=${m.n} (dup ${m.pos} / not-dup ${m.neg})`);
  console.log(`accuracy@0.5  ${fmtPct(m.accuracy)}`);
  console.log(`ECE-10        ${fmtNum(m.ece10)}`);
  console.log(`AUC           ${Number.isNaN(m.auc) ? "n/a (one class)" : m.auc.toFixed(3)}`);
  console.log(`latency       p50 ${fmtNum(m.p50ms, 0)} ms · p95 ${fmtNum(m.p95ms, 0)} ms · tokens ${m.tokens}`);
  if (c.coverage !== undefined) {
    console.log(`conformal     α=${c.alpha} · coverage ${(c.coverage * 100).toFixed(1)}% · abstain ${(c.abstainRate * 100).toFixed(1)}% · act-accuracy ${Number.isFinite(c.actAccuracy) ? (c.actAccuracy * 100).toFixed(1) + "%" : "n/a"} (${c.actedN} acted, calib ${c.calib}/eval ${c.eval})`);
  } else {
    console.log(`conformal     ${c.note}`);
  }
}

if (CLEF) {
  try {
    const rows = await clefArm(records);
    const m = metrics(rows);
    report.clef = { model: CLEF_MODEL, ...m };
    console.log(`\n## ${CLEF_MODEL} (hosted, network RTT included)`);
    console.log(`n=${m.n} (dup ${m.pos} / not-dup ${m.neg})`);
    console.log(`accuracy@0.5  ${fmtPct(m.accuracy)}`);
    console.log(`ECE-10        ${fmtNum(m.ece10)}`);
    console.log(`AUC           ${Number.isNaN(m.auc) ? "n/a (one class)" : m.auc.toFixed(3)}`);
    console.log(`latency       p50 ${fmtNum(m.p50ms, 0)} ms · p95 ${fmtNum(m.p95ms, 0)} ms · tokens ${m.tokens} (billed)`);
  } catch (e) {
    console.error(`\n! clef arm skipped: ${e.message}`);
    process.exitCode = 3;
  }
}

console.log("\n# honesty: p50 is the planning number; hosted latency includes network; conformal split is seeded (deterministic).");
if (report.model && report.clef) {
  console.log("# comparison is system-vs-system: same state object + question, each system serializes its own way.");
}
