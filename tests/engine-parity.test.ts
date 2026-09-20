import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { Questions, SystemOneResult } from "../src/core/types.js";

const ROOT = join(__dirname, "..");
const ARTIFACTS = join(ROOT, "artifacts");
const FIXTURES = join(ROOT, "tests", "fixtures");

interface FixtureCase {
  state: unknown;
  questions: Questions;
  expected: SystemOneResult;
}

const hasArtifacts = existsSync(join(ARTIFACTS, "english", "model.onnx"));

// Full-engine parity vs the real laya torch runtime. Skipped when artifacts are not
// present (e.g. CI) — `tools/export_onnx.py` + `tools/parity_reference.py` produce them.
describe.skipIf(!hasArtifacts)("Engine ONNX parity vs laya (torch)", () => {
  const engines: Record<string, Engine> = {};

  beforeAll(async () => {
    for (const name of ["english", "multilingual", "typed-decisions"]) {
      engines[name] = await Engine.fromArtifacts(join(ARTIFACTS, name));
    }
  }, 120_000);

  for (const name of ["english", "multilingual", "typed-decisions"]) {
    const fixturePath = join(FIXTURES, `parity-${name}.json`);
    const hasFixture = existsSync(fixturePath);

    describe.skipIf(!hasFixture)(name, () => {
      const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as { cases: FixtureCase[] };

      it("replays every case within tolerance", async () => {
        expect(fixture.cases.length).toBeGreaterThan(0);
        for (const c of fixture.cases) {
          const got = await engines[name].systemOne(c.state, c.questions);
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
          // same tokenization -> same token usage
          expect(got.usage.input_tokens).toBe(c.expected.usage.input_tokens);
        }
      }, 240_000);

      it("matches routing semantics on non-Latin states", async () => {
        const hindi = fixture.cases.find((c) => JSON.stringify(c.state).includes("शुल्क"));
        if (!hindi) return; // fixture without a non-Latin case (typed-decisions)
        const got = await engines[name].systemOne(hindi.state, hindi.questions);
        expect(got.answers).toBeDefined();
      });
    });
  }
});
