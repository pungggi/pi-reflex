import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { Answer, Questions, ScoreAnswer, SystemOneResult } from "../src/core/types.js";

const ROOT = join(__dirname, "..");
const ARTIFACTS = join(ROOT, "artifacts");
const FIXTURES = join(ROOT, "tests", "fixtures");

interface FixtureCase {
  state: unknown;
  questions: Questions;
  expected: SystemOneResult;
}

const CHECKPOINTS = ["english", "multilingual", "typed-decisions"] as const;

// Full-engine parity vs the real laya torch runtime (M6: each checkpoint loads and
// runs only when BOTH its artifacts and fixture exist — no cross-dependency).
for (const name of CHECKPOINTS) {
  const artifactsDir = join(ARTIFACTS, name);
  const fixturePath = join(FIXTURES, `parity-${name}.json`);
  const ready = existsSync(join(artifactsDir, "model.onnx")) && existsSync(fixturePath);

  describe.skipIf(!ready)(`Engine ONNX parity vs laya — ${name}`, () => {
    let engine: Engine;

    beforeAll(async () => {
      engine = await Engine.fromArtifacts(artifactsDir);
    }, 120_000);

    const loadFixture = (): { cases: FixtureCase[] } =>
      JSON.parse(readFileSync(fixturePath, "utf8")) as { cases: FixtureCase[] };

    it("replays every case within tolerance", async () => {
      const { cases } = loadFixture();
      expect(cases.length).toBeGreaterThan(0);
      for (const c of cases) {
        const got = await engine.systemOne(c.state, c.questions);
        for (const [qid, expA] of Object.entries(c.expected.answers)) {
          const gotA = got.answers[qid];
          expect(gotA).toBeDefined();
          if (expA.type === "choice" && gotA.type === "choice") {
            // identical argmax unless the top-2 gap is razor thin
            const probs = Object.values(expA.probabilities).sort((a, b) => b - a);
            const razorThin = probs.length > 1 && probs[0] - probs[1] < 0.04;
            if (!razorThin) expect(gotA.choice).toBe(expA.choice);
            for (const [k, v] of Object.entries(expA.probabilities)) {
              expect(Math.abs(gotA.probabilities[k] - v)).toBeLessThan(0.02);
            }
            expect(Math.abs(gotA.confidence - expA.confidence)).toBeLessThan(0.02);
          } else if (expA.type === "score" && gotA.type === "score") {
            expect(Math.abs(gotA.score - expA.score)).toBeLessThan(0.05);
            expect(Math.abs(gotA.confidence - expA.confidence)).toBeLessThan(0.02);
          } else if (expA.type === "noul" && gotA.type === "noul") {
            expect(Math.abs(gotA.noul - expA.noul)).toBeLessThan(0.02);
            expect(Math.abs(gotA.confidence - expA.confidence)).toBeLessThan(0.02);
          }
          expect(Math.abs(gotA.action.act_probability - expA.action.act_probability)).toBeLessThan(0.02);
        }
        // same tokenization -> same token usage (byte-parity proof)
        expect(got.usage.input_tokens).toBe(c.expected.usage.input_tokens);
      }
    }, 240_000);

    it("scores a non-Latin state without error when present in the fixture", async () => {
      const { cases } = loadFixture();
      const hindi = cases.find((c) => JSON.stringify(c.state).includes("शुल्क"));
      if (!hindi) return; // fixtures without a non-Latin case skip quietly
      const got = await engine.systemOne(hindi.state, hindi.questions);
      expect(Object.keys(got.answers).length).toBeGreaterThan(0);
    });

    it("batchQuestion matches per-state systemOne (contract D4 batching path)", async () => {
      const { cases } = loadFixture();
      const anyScore = cases.flatMap((c) => Object.values(c.questions)).find((q) => q.type === "score");
      if (!anyScore) return;
      const states = cases.map((c) => c.state);
      const batched = (await engine.batchQuestion(states, anyScore)) as ScoreAnswer[];
      for (let i = 0; i < states.length; i++) {
        const single = await engine.systemOne(states[i], { q: anyScore });
        expect((single.answers.q as Answer).type).toBe("score");
        expect(Math.abs((single.answers.q as ScoreAnswer).score - batched[i].score)).toBeLessThan(1e-6);
      }
    }, 120_000);
  });
}
