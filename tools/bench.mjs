// Latency benchmark: single Engine, realistic harness-routing questions.
// Run: npm run build && node tools/bench.mjs
import { join } from "node:path";
import { Engine } from "../dist/engine/engine.js";

const art = (name) => join(import.meta.dirname, "..", "artifacts", name);

const state = {
  from: "user@acme.com",
  subject: "Duplicate charge on invoice #4411",
  body: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
};

const questions = {
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

for (const name of ["english", "multilingual"]) {
  const engine = await Engine.fromArtifacts(art(name));
  // warmup
  await engine.systemOne(state, questions);
  const N = 10;
  const t0 = performance.now();
  let last;
  for (let i = 0; i < N; i++) last = await engine.systemOne(state, questions);
  const perCall = (performance.now() - t0) / N;
  const nQ = Object.keys(questions).length;
  console.log(
    `${name.padEnd(14)} ${perCall.toFixed(1).padStart(7)} ms / call  (${(perCall / nQ).toFixed(1)} ms/question, ${nQ} questions)  ` +
      `dept=${last.answers.department.choice} urgent=${last.answers.urgent.noul} sev=${last.answers.severity.score}`,
  );
}
console.log("BENCH OK");
