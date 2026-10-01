/**
 * Tool cores — the engine-backed logic behind the pi tools AND the MCP server.
 *
 * Kept free of pi/typebox imports so the standalone MCP server (bin/pi-reflex-mcp)
 * can reuse it without a pi installation. Each core returns both a human/model
 * readable `text` payload and a machine-readable `data` payload (pi structuredContent,
 * MCP structuredContent).
 */
import { Engine } from "../engine/engine.js";
import { INJECTION_GUARD, MODEL_ROUTER, tierFromRouter, type RouteRecommendation } from "../presets.js";
import type { NoulAnswer, SystemOneResult } from "../core/types.js";

export interface ToolCoreOutput {
  text: string;
  data: Record<string, unknown>;
}

function fmtResult(res: SystemOneResult, qid: string): string {
  const a = res.answers[qid];
  if (!a) return "no answer";
  const usage = ` [${res.usage.input_tokens} tok]`;
  if (a.type === "choice") return `${a.choice} (conf ${(a.confidence * 100).toFixed(1)}%) — ${JSON.stringify(a.probabilities)}${usage}`;
  if (a.type === "noul") return `P(true)=${a.noul} (conf ${(a.confidence * 100).toFixed(1)}%)${usage}`;
  return `score ${a.score}/${a.probabilities ? Object.keys(a.probabilities).length - 1 : "?"} (conf ${(a.confidence * 100).toFixed(1)}%)${usage}`;
}

/** Raw router recommendation for one message (shared by the reflex_route tool and the reflex/auto virtual model). */
export async function routeRaw(
  getEngine: () => Promise<Engine>,
  message: string,
): Promise<{ rec: RouteRecommendation; complexity?: string; needsCode?: number; longContext?: number; inputTokens: number }> {
  const engine = await getEngine();
  const res = await engine.systemOne(message, { ...MODEL_ROUTER, ...INJECTION_GUARD });
  const complexity = res.answers.complexity?.type === "choice" ? res.answers.complexity.choice : undefined;
  const needsCode = res.answers.needs_code?.type === "noul" ? (res.answers.needs_code as NoulAnswer).noul : undefined;
  const longContext = res.answers.long_context?.type === "noul" ? (res.answers.long_context as NoulAnswer).noul : undefined;
  const injection = res.answers.injection?.type === "noul" ? (res.answers.injection as NoulAnswer).noul : 0;
  const harmful = res.answers.harmful?.type === "noul" ? (res.answers.harmful as NoulAnswer).noul : 0;
  const rec = tierFromRouter({ complexity, needs_code: needsCode, long_context: longContext }, { injection, harmful });
  return { rec, complexity, needsCode, longContext, inputTokens: res.usage.input_tokens };
}

export function createToolCores(getEngine: () => Promise<Engine>, onLatency?: (ms: number) => void) {
  const timed = async <T>(fn: () => Promise<T>): Promise<T> => {
    const t0 = performance.now();
    try {
      return await fn();
    } finally {
      onLatency?.(performance.now() - t0);
    }
  };
  return {
    async decide(params: { state: string; instructions: string; options: Record<string, string> }): Promise<ToolCoreOutput> {
      const engine = await getEngine();
      const res = await timed(() =>
        engine.systemOne(params.state, {
          decision: { type: "choice", instructions: params.instructions, criteria: params.options },
        }),
      );
      const a = res.answers.decision;
      if (!a) return { text: "no answer", data: {} };
      const data =
        a.type === "choice"
          ? { type: "choice", choice: a.choice, probabilities: a.probabilities, confidence: a.confidence, inputTokens: res.usage.input_tokens }
          : {};
      return { text: fmtResult(res, "decision"), data };
    },
    async judge(params: { state: string; question: string }): Promise<ToolCoreOutput> {
      const engine = await getEngine();
      const res = await timed(() => engine.systemOne(params.state, { q: { type: "noul", instructions: params.question } }));
      const a = res.answers.q;
      if (!a) return { text: "no answer", data: {} };
      const data = a.type === "noul" ? { type: "bool", probability: a.noul, confidence: a.confidence, inputTokens: res.usage.input_tokens } : {};
      return { text: fmtResult(res, "q"), data };
    },
    async rate(params: { state: string; instructions: string; levels: string[] }): Promise<ToolCoreOutput> {
      const engine = await getEngine();
      if (params.levels.length < 2) throw new Error("rate needs at least 2 levels");
      const res = await timed(() =>
        engine.systemOne(params.state, {
          r: { type: "score", instructions: params.instructions, criteria: params.levels },
        }),
      );
      const a = res.answers.r;
      if (!a) return { text: "no answer", data: {} };
      const data =
        a.type === "score"
          ? {
              type: "score",
              score: a.score,
              levelCount: params.levels.length,
              probabilities: a.probabilities,
              confidence: a.confidence,
              inputTokens: res.usage.input_tokens,
            }
          : {};
      return { text: fmtResult(res, "r"), data };
    },
    async route(params: { message: string }): Promise<ToolCoreOutput> {
      const out = await timed(() => routeRaw(getEngine, params.message));
      const { rec, complexity, needsCode, longContext, inputTokens } = out;
      const detail = `complexity=${complexity} needs_code=${needsCode?.toFixed(2)} long_context=${longContext?.toFixed(2)} injection=${rec.guards.injection.toFixed(2)} harmful=${rec.guards.harmful.toFixed(2)}`;
      return {
        text: `tier: ${rec.tier} — ${rec.reason} (${detail})`,
        data: {
          type: "route",
          tier: rec.tier,
          reason: rec.reason,
          complexity,
          needsCode,
          longContext,
          guards: rec.guards,
          inputTokens,
        },
      };
    },
  };
}

export type ToolCores = ReturnType<typeof createToolCores>;
