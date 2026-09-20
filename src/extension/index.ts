/**
 * The pi extension surface — System 1 decision tools for pi coding-agent sessions.
 *
 * Loaded via the pi-package manifest (`pi.extensions: ["./extensions"]`, which
 * re-exports this module from dist/). Engine loads lazily on first tool use;
 * artifacts resolve from $PI_REFLEX_ARTIFACTS → cache → HF download (soft-fail
 * with actionable instructions when nothing is available).
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Engine } from "../engine/engine.js";
import { ensureEngine, findLocalEngine, type EngineName, type Quant } from "../engine/download.js";
import { INJECTION_GUARD, MODEL_ROUTER, tierFromRouter } from "../presets.js";
import type { Answer, NoulAnswer, SystemOneResult } from "../core/types.js";

export interface ExtensionDeps {
  loadEngine?: () => Promise<Engine>;
  env?: Record<string, string | undefined>;
}

interface EngineSlot {
  engine: Engine | null;
  error: string | null;
  lastLatencyMs: number | null;
  source: string | null;
}

async function resolveEngine(env: Record<string, string | undefined>): Promise<{ engine: Engine; source: string }> {
  const name = (env.PI_REFLEX_ENGINE as EngineName | undefined) ?? "multilingual";
  const quant: Quant = env.PI_REFLEX_QUANT === "fp32" ? "fp32" : "int8";
  const local = findLocalEngine(name, quant);
  if (local) return { engine: await Engine.fromArtifacts(local, { int8: quant === "int8" }), source: `local:${local}` };
  const dir = await ensureEngine(name, { quant });
  return { engine: await Engine.fromArtifacts(dir, { int8: quant === "int8" }), source: `downloaded:${dir}` };
}

function makeSlot(deps?: ExtensionDeps) {
  const env = deps?.env ?? process.env;
  const slot: EngineSlot = { engine: null, error: null, lastLatencyMs: null, source: null };
  return {
    async get(): Promise<Engine> {
      if (slot.engine) return slot.engine;
      if (slot.error) throw new Error(slot.error);
      try {
        if (deps?.loadEngine) {
          slot.engine = await deps.loadEngine();
          slot.source = "injected";
        } else {
          const { engine, source } = await resolveEngine(env);
          slot.engine = engine;
          slot.source = source;
        }
      } catch (e) {
        slot.error = `pi-reflex engine unavailable: ${(e as Error).message}. Generate artifacts with tools/export_onnx.py or set PI_REFLEX_ARTIFACTS.`;
        throw new Error(slot.error);
      }
      return slot.engine;
    },
    slot,
  };
}

function fmtResult(res: SystemOneResult, qid: string): string {
  const a = res.answers[qid] as Answer | undefined;
  if (!a) return "no answer";
  const usage = ` [${res.usage.input_tokens} tok]`;
  if (a.type === "choice") return `${a.choice} (conf ${(a.confidence * 100).toFixed(1)}%) — ${JSON.stringify(a.probabilities)}${usage}`;
  if (a.type === "noul") return `P(true)=${a.noul} (conf ${(a.confidence * 100).toFixed(1)}%)${usage}`;
  return `score ${a.score}/${a.probabilities ? Object.keys(a.probabilities).length - 1 : "?"} (conf ${(a.confidence * 100).toFixed(1)}%)${usage}`;
}

/** Tool-core (testable without pi): returns the text payload for each tool. */
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
    async decide(params: { state: string; instructions: string; options: Record<string, string> }): Promise<string> {
      const engine = await getEngine();
      const res = await timed(() => engine.systemOne(params.state, {
        decision: { type: "choice", instructions: params.instructions, criteria: params.options },
      }));
      return fmtResult(res, "decision");
    },
    async judge(params: { state: string; question: string }): Promise<string> {
      const engine = await getEngine();
      const res = await timed(() => engine.systemOne(params.state, { q: { type: "noul", instructions: params.question } }));
      return fmtResult(res, "q");
    },
    async rate(params: { state: string; instructions: string; levels: string[] }): Promise<string> {
      const engine = await getEngine();
      if (params.levels.length < 2) throw new Error("rate needs at least 2 levels");
      const res = await timed(() => engine.systemOne(params.state, {
        r: { type: "score", instructions: params.instructions, criteria: params.levels },
      }));
      return fmtResult(res, "r");
    },
    async route(params: { message: string }): Promise<string> {
      const engine = await getEngine();
      const res = await timed(() => engine.systemOne(params.message, { ...MODEL_ROUTER, ...INJECTION_GUARD }));
      const complexity = res.answers.complexity?.type === "choice" ? res.answers.complexity.choice : undefined;
      const needsCode = res.answers.needs_code?.type === "noul" ? (res.answers.needs_code as NoulAnswer).noul : undefined;
      const longCtx = res.answers.long_context?.type === "noul" ? (res.answers.long_context as NoulAnswer).noul : undefined;
      const injection = res.answers.injection?.type === "noul" ? (res.answers.injection as NoulAnswer).noul : 0;
      const harmful = res.answers.harmful?.type === "noul" ? (res.answers.harmful as NoulAnswer).noul : 0;
      const rec = tierFromRouter({ complexity, needs_code: needsCode, long_context: longCtx }, { injection, harmful });
      const detail = `complexity=${complexity} needs_code=${needsCode?.toFixed(2)} long_context=${longCtx?.toFixed(2)} injection=${injection.toFixed(2)} harmful=${harmful.toFixed(2)}`;
      return `tier: ${rec.tier} — ${rec.reason} (${detail})`;
    },
  };
}

export default function activate(pi: ExtensionAPI): void {
  const { get, slot } = makeSlot();
  const cores = createToolCores(get, (ms) => (slot.lastLatencyMs = ms));

  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify("pi-reflex ready (engine loads on first decision tool)", "info");
  });

  pi.registerTool({
    name: "reflex_decide",
    label: "Decide",
    description: "Fast calibrated single-choice decision over a state (System 1: no text generation). Use for routing, triage, categorization.",
    promptSnippet: "Make a fast calibrated choice among options (reflex_decide) instead of asking the LLM to classify",
    parameters: Type.Object({
      state: Type.String({ description: "The input text/JSON to decide over" }),
      instructions: Type.String({ description: "The question to answer about the state" }),
      options: Type.Record(Type.String(), Type.String({ description: "option description" }), { description: "label → description mapping" }),
    }),
    async execute(_id, params) {
      return { content: [{ type: "text", text: await cores.decide(params) }], details: {} };
    },
  });

  pi.registerTool({
    name: "reflex_judge",
    label: "Judge",
    description: "Calibrated probability that a condition holds for a state (binary, no text generation).",
    promptSnippet: "Get a calibrated P(true) for a yes/no question (reflex_judge) instead of asking the LLM to guess",
    parameters: Type.Object({
      state: Type.String(),
      question: Type.String({ description: "Yes/no question about the state" }),
    }),
    async execute(_id, params) {
      return { content: [{ type: "text", text: await cores.judge(params) }], details: {} };
    },
  });

  pi.registerTool({
    name: "reflex_rate",
    label: "Rate",
    description: "Calibrated rating on an ordinal rubric (expected level + distribution).",
    parameters: Type.Object({
      state: Type.String(),
      instructions: Type.String(),
      levels: Type.Array(Type.String(), { minItems: 2, description: "ordered rubric levels, low → high" }),
    }),
    async execute(_id, params) {
      return { content: [{ type: "text", text: await cores.rate(params) }], details: {} };
    },
  });

  pi.registerTool({
    name: "reflex_route",
    label: "Route model",
    description: "Recommend a model tier (small/mid/frontier) + guardrail flags for an incoming message, in one ~50ms pass.",
    promptSnippet: "Use reflex_route on new user messages to pick the model tier cheaply before starting work",
    parameters: Type.Object({
      message: Type.String({ description: "Incoming user message" }),
    }),
    async execute(_id, params) {
      return { content: [{ type: "text", text: await cores.route(params) }], details: {} };
    },
  });

  pi.registerCommand("reflex", {
    description: "pi-reflex engine status",
    handler: async (_args, ctx) => {
      const status = slot.engine ? `loaded (${slot.source})` : slot.error ? `error: ${slot.error}` : "idle (loads on first tool call)";
      ctx.ui.notify(`pi-reflex: ${status}${slot.lastLatencyMs ? ` · last call ${slot.lastLatencyMs.toFixed(0)}ms` : ""}`, "info");
    },
  });
}
