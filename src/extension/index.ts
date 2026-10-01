/**
 * The pi extension surface — System 1 decision tools for pi coding-agent sessions.
 *
 * Loaded via the pi-package manifest (`pi.extensions: ["./extensions"]`, which
 * re-exports this module from dist/). Engine loads lazily on first use;
 * artifacts resolve from $PI_REFLEX_ARTIFACTS → cache → HF download (soft-fail
 * with actionable instructions when nothing is available).
 *
 * Registers (pi ≥ 0.99):
 * - 4 decision tools with structured output (namespace `reflex`, read-only)
 * - the `reflex` classifier provider: local classifier models next to Jev
 * - the `reflex/auto` virtual model: per-turn tier routing (env-mapped models)
 * - optional MCP server registration (PI_REFLEX_MCP=1)
 * - optional prompt-injection guard on context_with_system (PI_REFLEX_GUARD=1)
 */
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { JsonValue } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { Engine } from "../engine/engine.js";
import { ensureEngine, findLocalEngine, type EngineName, type Quant } from "../engine/download.js";
import { createToolCores } from "./cores.js";
import { registerReflexProvider, REFLEX_PROVIDER_ID } from "./provider.js";
import { latestUserText, routeUserTurn, type ReflexAutoState } from "./vmodel.js";
import { createInjectionGuard } from "./guard.js";

export interface ExtensionDeps {
  loadEngine?: () => Promise<Engine>;
  env?: Record<string, string | undefined>;
}

interface EngineSlot {
  engines: Map<string, Engine>;
  error: string | null;
  lastLatencyMs: number | null;
  source: string | null;
  engineName: EngineName;
  quant: Quant;
}

function makeSlot(deps?: ExtensionDeps) {
  const env = deps?.env ?? process.env;
  const slot: EngineSlot = {
    engines: new Map(),
    error: null,
    lastLatencyMs: null,
    source: null,
    engineName: (env.PI_REFLEX_ENGINE as EngineName | undefined) ?? "multilingual",
    quant: env.PI_REFLEX_QUANT === "fp32" ? "fp32" : "int8",
  };

  const loadNamed = async (name: EngineName): Promise<Engine> => {
    if (deps?.loadEngine) return deps.loadEngine();
    const local = findLocalEngine(name, slot.quant);
    if (local) {
      slot.source = `local:${local}`;
      return Engine.fromArtifacts(local, { int8: slot.quant === "int8" });
    }
    const dir = await ensureEngine(name, { quant: slot.quant });
    slot.source = `downloaded:${dir}`;
    return Engine.fromArtifacts(dir, { int8: slot.quant === "int8" });
  };

  return {
    slot,
    /** The default engine (tools, guard, router). */
    async get(): Promise<Engine> {
      return this.getNamed(slot.engineName);
    },
    /** Engine per classifier model id (multilingual | english | typed-decisions). */
    async getNamed(name: string): Promise<Engine> {
      const cached = slot.engines.get(name);
      if (cached) return cached;
      if (slot.error) throw new Error(slot.error);
      try {
        const engine = await loadNamed(name as EngineName);
        slot.engines.set(name, engine);
      } catch (e) {
        slot.error = `pi-reflex engine unavailable: ${(e as Error).message}. Generate artifacts with tools/export_onnx.py or set PI_REFLEX_ARTIFACTS.`;
        throw new Error(slot.error);
      }
      return slot.engines.get(name)!;
    },
  };
}

export { createToolCores } from "./cores.js";

export default function activate(pi: ExtensionAPI, deps?: ExtensionDeps): void {
  const env = deps?.env ?? process.env;
  const { get, getNamed, slot } = makeSlot(deps);
  const cores = createToolCores(get, (ms) => (slot.lastLatencyMs = ms));

  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify(
      "pi-reflex ready: classifier models (reflex/*), reflex/auto router, decision tools (engine loads on first use)",
      "info",
    );
  });

  // ── ② classifier provider ────────────────────────────────────────────────
  registerReflexProvider(pi, (modelId) => getNamed(modelId));

  // ── ③ virtual model reflex/auto ──────────────────────────────────────────
  pi.registerVirtualModel<ReflexAutoState>({
    provider: REFLEX_PROVIDER_ID,
    id: "auto",
    name: "Reflex Auto",
    thinkingLevels: ["low", "high"],
    async route(request, ctx) {
      const sticky = request.reason !== "user" ? (request.failed ?? request.previous) : undefined;
      if (sticky) {
        return { model: sticky.model, thinkingLevel: sticky.thinkingLevel ?? "medium" };
      }
      const message = latestUserText(request.messages);
      const out = await routeUserTurn(get, message, request.thinkingLevel, {
        env,
        find: (provider, id) => ctx.modelRegistry.find(provider, id),
      }, request.previous?.model);
      return { model: out.model, thinkingLevel: out.thinkingLevel, state: out.state };
    },
  });

  // ── ④ decision tools (namespace, annotations, structured output) ─────────
  const exposure = env.PI_REFLEX_EXPOSURE === "codemode" ? ("codemode" as const) : undefined;
  const namespace = {
    name: "reflex",
    description: "pi-reflex System 1 decisions: local calibrated choice/bool/score in one forward pass",
    instructions:
      "Use these instead of asking the model to classify: reflex_decide (routing/triage), reflex_judge (yes/no probability), reflex_rate (ordinal rubric), reflex_route (model tier + guardrails).",
  };
  const annotations = { readOnlyHint: true, idempotentHint: true, openWorldHint: false };

  pi.registerTool({
    name: "reflex_decide",
    label: "Decide",
    description: "Fast calibrated single-choice decision over a state (System 1: no text generation). Use for routing, triage, categorization.",
    promptSnippet: "Make a fast calibrated choice among options (reflex_decide) instead of asking the LLM to classify",
    namespace,
    annotations,
    exposure,
    parameters: Type.Object({
      state: Type.String({ description: "The input text/JSON to decide over" }),
      instructions: Type.String({ description: "The question to answer about the state" }),
      options: Type.Record(Type.String(), Type.String({ description: "option description" }), { description: "label → description mapping" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("choice"),
      choice: Type.String(),
      probabilities: Type.Record(Type.String(), Type.Number()),
      confidence: Type.Number(),
      inputTokens: Type.Integer(),
    }),
    async execute(_id, params) {
      const out = await cores.decide(params);
      return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
    },
  });

  pi.registerTool({
    name: "reflex_judge",
    label: "Judge",
    description: "Calibrated probability that a condition holds for a state (binary, no text generation).",
    promptSnippet: "Get a calibrated P(true) for a yes/no question (reflex_judge) instead of asking the LLM to guess",
    namespace,
    annotations,
    exposure,
    parameters: Type.Object({
      state: Type.String(),
      question: Type.String({ description: "Yes/no question about the state" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("bool"),
      probability: Type.Number(),
      confidence: Type.Number(),
      inputTokens: Type.Integer(),
    }),
    async execute(_id, params) {
      const out = await cores.judge(params);
      return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
    },
  });

  pi.registerTool({
    name: "reflex_rate",
    label: "Rate",
    description: "Calibrated rating on an ordinal rubric (expected level + distribution).",
    namespace,
    annotations,
    exposure,
    parameters: Type.Object({
      state: Type.String(),
      instructions: Type.String(),
      levels: Type.Array(Type.String(), { minItems: 2, description: "ordered rubric levels, low → high" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("score"),
      score: Type.Number(),
      levelCount: Type.Integer(),
      probabilities: Type.Record(Type.String(), Type.Number()),
      confidence: Type.Number(),
      inputTokens: Type.Integer(),
    }),
    async execute(_id, params) {
      const out = await cores.rate(params);
      return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
    },
  });

  pi.registerTool({
    name: "reflex_route",
    label: "Route model",
    description: "Recommend a model tier (small/mid/frontier) + guardrail flags for an incoming message, in one ~50ms pass.",
    promptSnippet: "Use reflex_route on new user messages to pick the model tier cheaply before starting work",
    namespace,
    annotations,
    exposure,
    parameters: Type.Object({
      message: Type.String({ description: "Incoming user message" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("route"),
      tier: Type.Union([Type.Literal("small"), Type.Literal("mid"), Type.Literal("frontier")]),
      reason: Type.String(),
      complexity: Type.Optional(Type.String()),
      needsCode: Type.Optional(Type.Number()),
      longContext: Type.Optional(Type.Number()),
      guards: Type.Object({ injection: Type.Number(), harmful: Type.Number() }),
      inputTokens: Type.Integer(),
    }),
    async execute(_id, params) {
      const out = await cores.route(params);
      return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
    },
  });

  // ── ⑤ optional MCP registration (default off: spawns a node process) ─────
  if (env.PI_REFLEX_MCP === "1") {
    const serverJs = fileURLToPath(new URL("../mcp/server.js", import.meta.url));
    pi.registerMcpServer("reflex", {
      command: process.execPath,
      args: [serverJs],
      exposure: "codemode",
      description: "pi-reflex System 1 decisions (local ONNX engine)",
    });
  }

  // ── ⑥ optional prompt-injection guard (default off; PI_REFLEX_GUARD=1) ───
  const guard = env.PI_REFLEX_GUARD === "1" ? createInjectionGuard(get, { threshold: Number(env.PI_REFLEX_GUARD_THRESHOLD) || 0.75 }) : null;
  if (guard) {
    pi.on("context_with_system", async (event) => {
      const messages = await guard.process(event.messages);
      return messages ? { messages } : undefined;
    });
  }

  pi.registerCommand("reflex", {
    description: "pi-reflex engine + surfaces status",
    handler: async (_args, ctx) => {
      const engine = slot.engines.get(slot.engineName);
      const status = engine
        ? `loaded (${slot.source})`
        : slot.error
          ? `error: ${slot.error}`
          : "idle (loads on first use)";
      const tiers = ["SMALL", "MID", "FRONTIER"]
        .map((t) => env[`PI_REFLEX_TIER_${t}`] ?? "-")
        .join(" | ");
      const parts = [
        `engine: ${status}${slot.lastLatencyMs ? ` · last call ${slot.lastLatencyMs.toFixed(0)}ms` : ""}`,
        `classifier: ${REFLEX_PROVIDER_ID}/multilingual|english|typed-decisions`,
        `reflex/auto tiers: ${tiers}`,
        `guard: ${guard ? `${guard.stats.checked} checked, ${guard.stats.flagged} flagged${guard.stats.tripped ? " (tripped)" : ""}` : "off (PI_REFLEX_GUARD=1)"}`,
        `mcp: ${env.PI_REFLEX_MCP === "1" ? "registered" : "off (PI_REFLEX_MCP=1)"}`,
      ];
      ctx.ui.notify(`pi-reflex:\n${parts.join("\n")}`, "info");
    },
  });
}
