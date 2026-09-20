// Latency evidence for CONTRACT-harness.md §3 (dev-only; run from repo root):
//   npm run build && node tools/bench.mjs
// Scenarios map to contract call-sites:
//   noul_single     → D1 dedupe / D2 create-gate shape (one pair / one delta)
//   score2_single   → D3 importance re-score shape
//   batch8_score2   → D4 injection relevance shape (N items, ONE call)
//   four_questions  → informational multi-question pass
// Uses int8 graphs (the deployment posture for CPU-only harness hosts).
import { join } from "node:path";
import { Engine } from "../dist/engine/engine.js";

const art = (name) => join(import.meta.dirname, "..", "artifacts", name);
const N = 20; // warm runs per scenario

const PAIR_STATE = (i) => ({
  a: `Item A${i}: retry the payment API with exponential backoff and jitter`,
  b: `Item B${i}: payment API calls must retry with backoff`,
  kind: "memory",
  evidence: "both persisted in the same session week",
});
const ITEM_STATE = (i) => ({
  item: `Item ${i}: prefer JSONL exports for calibration corpora; hash lifecycle contents with sha256`,
  task: "Current task: review the corpus loader schema for contract conformance",
});
const QUESTIONS = {
  department: {
    type: "choice",
    instructions: "Which department should handle this request?",
    criteria: {
      billing: "invoices, payments, refunds, subscription charges",
      technical: "bugs, outages, system errors, performance problems",
      sales: "pricing, new contracts, upgrades",
      other: "everything else",
    },
  },
  urgent: { type: "noul", instructions: "Does this require immediate intervention?" },
  angry: { type: "noul", instructions: "Is the user threatening to cancel?" },
  severity: {
    type: "score",
    instructions: "How severe is this issue?",
    criteria: ["minor annoyance", "degraded experience", "blocking work", "production down"],
  },
};
const NOUL_Q = { same: { type: "noul", instructions: "Are these two items the same durable fact?" } };
const SCORE2_Q = {
  relevance: {
    type: "score",
    instructions: "How relevant is this item to the current task?",
    criteria: ["irrelevant", "relevant"],
  },
};

function pct(sorted, p) {
  const idx = Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)));
  return sorted[idx];
}

async function time(fn, n = N) {
  await fn(); // warmup
  const samples = [];
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    await fn();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  return { p50: pct(samples, 0.5), p95: pct(samples, 0.95), mean, n: samples.length };
}

const fmt = (r) => `p50 ${r.p50.toFixed(1).padStart(7)} ms | p95 ${r.p95.toFixed(1).padStart(7)} ms | mean ${r.mean.toFixed(1).padStart(7)} ms (n=${r.n})`;

for (const name of ["multilingual", "english", "typed-decisions"]) {
  let engine;
  try {
    engine = await Engine.fromArtifacts(art(name), { int8: true });
  } catch (e) {
    console.log(`[${name}] SKIPPED: ${e.message.split("\n")[0]}`);
    continue;
  }
  console.log(`\n=== ${name} (int8, n=${N}) ===`);

  const d1 = await time(() => engine.systemOne(PAIR_STATE(1), NOUL_Q));
  console.log(`D1/D2  noul_single      ${fmt(d1)}`);

  const d3 = await time(() => engine.systemOne(ITEM_STATE(1), SCORE2_Q));
  console.log(`D3     score2_single    ${fmt(d3)}`);

  const d4 = await time(() => engine.batchQuestion([0, 1, 2, 3, 4, 5, 6, 7].map(ITEM_STATE), SCORE2_Q.relevance));
  console.log(`D4     batch8_score2    ${fmt(d4)}  (${(d4.p50 / 8).toFixed(1)} ms/item p50)`);

  const info = await time(() => engine.systemOne(PAIR_STATE(1), QUESTIONS));
  console.log(`info   four_questions   ${fmt(info)}`);
}
console.log("\nBENCH OK");
