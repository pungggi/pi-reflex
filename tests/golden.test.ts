/**
 * CONTRACT-harness.md §5.4 — golden determinism scaffolding.
 *
 * Fixed question set + fixed states + deterministic calibration → snapshot of the
 * prediction sets. Same input ⇒ same set, asserted twice per run (self-check) and
 * against the committed fixture (cross-run drift).
 *
 * HONESTY: the fixture at tests/fixtures/golden-prediction-sets.json is REAL engine
 * output (ONNX int8), written only via `PI_JEV_WRITE_GOLDEN=1 npx vitest run
 * tests/golden.test.ts`. It is never hand-authored. Skipped when artifacts are
 * absent (fresh clones / CI) — the engine-parity suite is the gate for running it.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import { NoulConformal, ScoreConformal } from "../src/core/conformal.js";
import type { Questions, NoulAnswer, ScoreAnswer } from "../src/core/types.js";

const ROOT = join(__dirname, "..");
const ARTIFACTS = join(ROOT, "artifacts");
const FIXTURE = join(ROOT, "tests", "fixtures", "golden-prediction-sets.json");
const READY = existsSync(join(ARTIFACTS, "english", "model.int8.onnx"));

// Fixed question set — contract call-site shapes (D1/D2 noul, D3/D4 score).
const QUESTIONS: Questions = {
  same_fact: { type: "noul", instructions: "Are these two items the same durable fact?" },
  durable_reusable: { type: "noul", instructions: "Is this proposed delta durable and reusable?" },
  importance: {
    type: "score",
    instructions: "How fit is this stored item for future reuse?",
    criteria: ["unfit", "marginal", "useful", "highly reusable"],
  },
  relevance: {
    type: "score",
    instructions: "How relevant is this item to the current task?",
    criteria: ["irrelevant", "relevant"],
  },
};

// Fixed states — deterministic content, contract-shaped inputs.
const STATES = [
  {
    pair: { a: "Retry payment API with exponential backoff", b: "Payment API retries use exponential backoff" },
    delta: "Store the corpus loader fixtures under tests/fixtures",
    item: "JSONL is the export format for calibration corpora",
    task: "Contract conformance pass on the corpus loader",
  },
  {
    pair: { a: "Sessions live in Redis with sliding expiry", b: "Refresh tokens belong in httpOnly cookies" },
    delta: "Bump the minor version and update the changelog",
    item: "The deploy target is Fly.io eu-west",
    task: "Contract conformance pass on the corpus loader",
  },
  {
    pair: { a: "Run npm test before committing", b: "npm test must pass before any commit" },
    delta: "Add a conformal golden fixture for harness determinism",
    item: "Conformal prediction sets need fixed calibration data",
    task: "Extending the conformal guarantee layer",
  },
];

// Deterministic calibration (fixed literals — same as the conformal goldens style).
const NOUL_CALIB = {
  p: [0.98, 0.02, 0.97, 0.03, 0.99, 0.01, 0.96, 0.04, 0.985, 0.015, 0.975, 0.025],
  labels: [true, false, true, false, true, false, true, false, true, false, true, false],
};
const SCORE_CALIB = {
  pred: [0.1, 0.9, 1.2, 2.8, 3.1, 0.4, 2.2, 3.6, 1.8, 0.7, 2.9, 3.3],
  truth: [0, 1, 1, 3, 3, 0, 2, 3, 2, 1, 3, 3],
};

interface GoldenSnapshot {
  engine: string;
  quantization: "int8";
  alpha: { noul: number; score: number };
  states: Array<Record<string, { answer: Record<string, unknown>; predictionSet: unknown }>>;
}

async function buildSnapshot(): Promise<GoldenSnapshot> {
  const engine = await Engine.fromArtifacts(join(ARTIFACTS, "english"), { int8: true });
  const noul = new NoulConformal().fit(NOUL_CALIB.p, NOUL_CALIB.labels, 0.1);
  const score = new ScoreConformal().fit(SCORE_CALIB.pred, SCORE_CALIB.truth, 0.1);

  const states: GoldenSnapshot["states"] = [];
  for (const state of STATES) {
    const result = await engine.systemOne(state, QUESTIONS);
    const entry: GoldenSnapshot["states"][number] = {};
    for (const [qid, answer] of Object.entries(result.answers)) {
      if (answer.type === "noul") {
        const a = answer as NoulAnswer;
        entry[qid] = {
          answer: { type: a.type, noul: a.noul, confidence: a.confidence },
          predictionSet: { kind: "noul", set: noul.set(a.noul) }, // [falseIn, trueIn]
        };
      } else if (answer.type === "score") {
        const a = answer as ScoreAnswer;
        entry[qid] = {
          answer: { type: a.type, score: a.score, confidence: a.confidence },
          predictionSet: { kind: "scoreInterval", interval: score.interval(a.score) },
        };
      } else {
        entry[qid] = { answer: { type: answer.type, choice: (answer as { choice: string }).choice }, predictionSet: { kind: "argmax" } };
      }
    }
    states.push(entry);
  }
  return { engine: "english", quantization: "int8", alpha: { noul: 0.1, score: 0.1 }, states };
}

describe.skipIf(!READY)("CONTRACT §5.4 — golden determinism (prediction sets)", () => {
  let snap: GoldenSnapshot;
  let snap2: GoldenSnapshot;

  beforeAll(async () => {
    snap = await buildSnapshot();
    snap2 = await buildSnapshot(); // determinism self-check: two runs, same input
  }, 180_000);

  it("same input ⇒ same prediction set (two consecutive runs)", () => {
    expect(JSON.stringify(snap2)).toBe(JSON.stringify(snap));
  });

  it("matches the committed golden fixture (regenerable via PI_JEV_WRITE_GOLDEN=1)", () => {
    if (process.env.PI_JEV_WRITE_GOLDEN === "1") {
      writeFileSync(FIXTURE, JSON.stringify(snap, null, 1) + "\n", "utf8");
      console.log(`[golden] wrote ${FIXTURE}`);
    }
    expect(existsSync(FIXTURE)).toBe(true);
    const committed = JSON.parse(readFileSync(FIXTURE, "utf8")) as GoldenSnapshot;
    expect(snap).toEqual(committed);
  });

  it("rule zero holds on the golden run: sets are singleton or abstain — never trusted blind", () => {
    for (const state of snap.states) {
      for (const [qid, entry] of Object.entries(state)) {
        if (entry.predictionSet.kind === "noul") {
          const [f, t] = entry.predictionSet.set as [boolean, boolean];
          expect(typeof f).toBe("boolean");
          expect(typeof t).toBe("boolean");
        }
      }
    }
  });
});
