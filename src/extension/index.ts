/**
 * The pi extension surface — System 1 decision tools for pi coding-agent sessions.
 *
 * Loaded via the pi-package manifest (`pi.extensions: ["./extensions"]`, which
 * re-exports this module from dist/). Engine loads lazily on first use;
 * artifacts resolve from $PI_REFLEX_ARTIFACTS → cache → HF download (soft-fail
 * with actionable instructions when nothing is available).
 *
 * Registers (pi ≥ 0.99; aligned with pi 1.0's leaner codemode):
 * - 4 decision tools with structured output (namespace `reflex`, read-only;
 *   PI_REFLEX_EXPOSURE=codemode|deferred keeps them out of the model's tool list)
 * - compact tool renderers via pi.registerToolRenderer (pi ≥ 1.0.1): one-line
 *   results for the registered tools, the MCP-served copies
 *   (mcp__reflex__reflex_*), and reflex calls in resumed sessions / HTML exports
 * - the `reflex` classifier provider: local classifier models next to Jev
 * - the `reflex/auto` virtual model: per-turn tier routing (env-mapped models)
 * - optional MCP server registration (PI_REFLEX_MCP=1)
 * - optional prompt-injection guard on context_with_system (PI_REFLEX_GUARD=1)
 *
 * pi 1.0 alignment: tool failures return `isError: true` + a structured recovery
 * payload (codemode scripts read structuredContent and can degrade) instead of
 * throwing; the startup banner is opt-in (PI_REFLEX_QUIET=0), matching pi's
 * quieter startup (quietStartup: "header").
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
import { reflexToolRendererResolver } from "./renderers.js";

export interface ExtensionDeps {
  loadEngine?: (name: EngineName) => Promise<Engine>;
  env?: Record<string, string | undefined>;
}

interface EngineSlot {
  engines: Map<string, Engine>;
  /** In-flight loads, keyed by engine name — concurrent callers share one load (PR#2 review #6). */
  loading: Map<string, Promise<Engine>>;
  /** Per-engine load failures — one bad engine must not poison the others (PR#2 review #1). */
  errors: Map<string, string>;
  lastLatencyMs: number | null;
  source: string | null;
  engineName: EngineName;
  quant: Quant;
}

function makeSlot(deps?: ExtensionDeps) {
  const env = deps?.env ?? process.env;
  const slot: EngineSlot = {
    engines: new Map(),
    loading: new Map(),
    errors: new Map(),
    lastLatencyMs: null,
    source: null,
    engineName: (env.PI_REFLEX_ENGINE as EngineName | undefined) ?? "multilingual",
    quant: env.PI_REFLEX_QUANT === "fp32" ? "fp32" : "int8",
  };

  const loadNamed = async (name: EngineName): Promise<Engine> => {
    if (deps?.loadEngine) return deps.loadEngine(name);
    const local = findLocalEngine(name, slot.quant);
    if (local) {
      slot.source = `local:${local}`;
      return Engine.fromArtifacts(local, { int8: slot.quant === "int8" });
    }
    const dir = await ensureEngine(name, { quant: slot.quant });
    slot.source = `downloaded:${dir}`;
    return Engine.fromArtifacts(dir, { int8: slot.quant === "int8" });
  };

  const getNamed = async (name: string): Promise<Engine> => {
    const cached = slot.engines.get(name);
    if (cached) return cached;
    const inFlight = slot.loading.get(name);
    if (inFlight) return inFlight;
    const load = (async () => {
      try {
        const engine = await loadNamed(name as EngineName);
        slot.engines.set(name, engine);
        slot.errors.delete(name);
        return engine;
      } catch (e) {
        const message = `pi-reflex engine '${name}' unavailable: ${(e as Error).message}. Generate artifacts with tools/export_onnx.py or set PI_REFLEX_ARTIFACTS.`;
        slot.errors.set(name, message);
        throw new Error(message);
      } finally {
        slot.loading.delete(name);
      }
    })();
    slot.loading.set(name, load);
    return load;
  };

  // Closures, not object-literal methods: `activate()` destructures `get` off the
  // returned object, which would strip `this` and break every engine call through
  // the extension (tools, guard, router) — a latent bug in ≤ 0.1.4 caught by the
  // pi 1.0 alignment tests.
  return {
    slot,
    /** The default engine (tools, guard, router). */
    get: () => getNamed(slot.engineName),
    /** Engine per classifier model id (multilingual | english | typed-decisions). */
    getNamed,
  };
}

export { createToolCores } from "./cores.js";

/**
 * pi ≥ 1.0 failure convention (agent tool contract): report failures with
 * `isError: true` instead of throwing — the model still sees `content` as an
 * error result, but `structuredContent` survives for codemode scripts and
 * programmatic callers, so they can degrade instead of aborting. Engine errors
 * already carry the artifacts hint in their message.
 */
function errorResult(e: unknown) {
  const message = e instanceof Error ? e.message : String(e);
  return {
    content: [{ type: "text" as const, text: `pi-reflex error: ${message}` }],
    structuredContent: {
      type: "error",
      error: message,
      recovery:
        "/reflex shows engine status; engine errors usually mean missing artifacts — generate with tools/export_onnx.py or set PI_REFLEX_ARTIFACTS to a local artifacts dir",
    } as unknown as JsonValue,
    isError: true,
    details: {},
  };
}

export default function activate(pi: ExtensionAPI, deps?: ExtensionDeps): void {
  const env = deps?.env ?? process.env;
  const { get, getNamed, slot } = makeSlot(deps);
  const cores = createToolCores(get, (ms) => (slot.lastLatencyMs = ms));

  // Startup banner is opt-in (PI_REFLEX_QUIET=0), aligned with pi ≥ 1.0's
  // quieter startup (quietStartup: "header"); status lives in /reflex.
  pi.on("session_start", async (_event, ctx) => {
    if (env.PI_REFLEX_QUIET !== "0") return;
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
  // `codemode`: listed one line each in the codemode tool, callable from scripts.
  // `deferred`: not listed anywhere; tool_search finds and activates it (pi ≥ 1.0
  // keeps deferred tools across resume//reload). Both keep the tools out of the
  // model's tool list.
  const exposure =
    env.PI_REFLEX_EXPOSURE === "codemode" || env.PI_REFLEX_EXPOSURE === "deferred"
      ? (env.PI_REFLEX_EXPOSURE as "codemode" | "deferred")
      : undefined;
  // pi ≥ 1.0 codemode lists each tool as ONE line (its `description`) and keeps
  // namespace `instructions` out of every prompt — scripts read them with
  // describeNamespace("reflex"). So descriptions stay crisp one-liners and the
  // decision guide lives in instructions, prompt-free.
  const namespace = {
    name: "reflex",
    description: "Local System 1 decisions: calibrated choice/bool/score in one forward pass, no text generation",
    instructions:
      "Use these instead of generating an answer for classification-shaped work (routing, triage, rubric scoring, yes/no checks): one local forward pass each (~50–200 ms, no API cost, no text generation). " +
      "Pick by shape: reflex_decide = one label from a set; reflex_judge = P(true) for a yes/no question; reflex_rate = ordinal rubric with expected level + distribution; reflex_route = model tier + guardrail flags for an incoming user message. " +
      "Every result carries calibrated probabilities and confidence; treat confidence < 0.5 as abstain and fall back to the model. " +
      'Probe availability with "reflex_decide" in tools (typeof probes do not work in codemode).',
  };
  const annotations = { readOnlyHint: true, idempotentHint: true, openWorldHint: false };

  pi.registerTool({
    name: "reflex_decide",
    label: "Decide",
    description: "Calibrated single-choice decision (routing, triage, categorization) in one local forward pass — no text generation.",
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
      try {
        const out = await cores.decide(params);
        return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
      } catch (e) {
        return errorResult(e);
      }
    },
  });

  pi.registerTool({
    name: "reflex_judge",
    label: "Judge",
    description: "Calibrated P(true) for a yes/no question about a state — no text generation.",
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
      try {
        const out = await cores.judge(params);
        return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
      } catch (e) {
        return errorResult(e);
      }
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
      try {
        const out = await cores.rate(params);
        return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
      } catch (e) {
        return errorResult(e);
      }
    },
  });

  pi.registerTool({
    name: "reflex_route",
    label: "Route model",
    description: "Recommend a model tier (small/mid/frontier) + guardrail flags for an incoming message in one ~50–200 ms pass.",
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
      try {
        const out = await cores.route(params);
        return { content: [{ type: "text", text: out.text }], structuredContent: out.data as unknown as JsonValue, details: {} };
      } catch (e) {
        return errorResult(e);
      }
    },
  });

  // ── ⑤ compact tool renderers (pi ≥ 1.0.1 registerToolRenderer) ─────────────
  // Resolves by NAME, so the same one-line rendering covers the four registered
  // tools above, their MCP-served twins (mcp__reflex__reflex_*), and reflex calls
  // in resumed sessions / HTML exports drawn before any tool or server existed.
  // Cosmetic only — guarded so pi ≥ 0.99 hosts without the API keep working.
  if (typeof pi.registerToolRenderer === "function") pi.registerToolRenderer(reflexToolRendererResolver);

  // ── ⑥ optional MCP registration (default off: spawns a node process) ─────
  if (env.PI_REFLEX_MCP === "1") {
    const serverJs = fileURLToPath(new URL("../mcp/server.js", import.meta.url));
    pi.registerMcpServer("reflex", {
      command: process.execPath,
      args: [serverJs],
      exposure: "codemode",
      description: "pi-reflex System 1 decisions (local ONNX engine)",
    });
  }

  // ── ⑦ optional prompt-injection guard (default off; PI_REFLEX_GUARD=1) ───
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
        : slot.errors.get(slot.engineName)
          ? `error: ${slot.errors.get(slot.engineName)}`
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
        `startup banner: ${env.PI_REFLEX_QUIET === "0" ? "on" : "off (PI_REFLEX_QUIET=0)"}`,
      ];
      ctx.ui.notify(`pi-reflex:\n${parts.join("\n")}`, "info");
    },
  });
}
