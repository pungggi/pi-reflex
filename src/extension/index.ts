/**
 * The pi extension surface — System 1 decision tools for pi coding-agent sessions.
 *
 * Loaded via the pi-package manifest (`pi.extensions: ["./extensions"]`, which
 * re-exports this module from dist/). Engine loads lazily on first use;
 * artifacts resolve from $PI_REFLEX_ARTIFACTS → cache → HF download with
 * byte-resume across attempts, footer progress while downloading, and soft-fail
 * with end-user recovery instructions when nothing is available.
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
 * throwing. The startup banner is on by default (PI_REFLEX_QUIET=1 silences it).
 * `deferred` exposure self-checks on session_start: without an active tool_search
 * the deferred tools are unreachable, so the extension warns (and activates
 * tool_search when nothing else could reach them).
 */
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Api, ClassifierResult, JsonValue, Model } from "@earendil-works/pi-ai";
import { Engine } from "../engine/engine.js";
import { ensureEngine, findLocalEngine, type EngineName, type Quant } from "../engine/download.js";
import { createToolCores } from "./cores.js";
import { REFLEX_TOOL_SCHEMAS } from "./schemas.js";
import { registerReflexProvider, REFLEX_PROVIDER_ID } from "./provider.js";
import { latestUserText, routeUserTurn, type ReflexAutoState } from "./vmodel.js";
import { createInjectionGuard } from "./guard.js";
import { NATIVE_PROVIDER_ID, routerBackendFromEnv } from "./native.js";
import { reflexToolRendererResolver, reflexToolFromName } from "./renderers.js";
import {
  harnessInUse,
  loadNudgeState,
  nudgeMessage,
  parseNudgeDays,
  saveNudgeState,
  shouldNudge,
} from "./nudge.js";
import { homedir } from "node:os";
import { join } from "node:path";

export interface ExtensionDeps {
  loadEngine?: ((name: EngineName) => Promise<Engine>) | undefined;
  env?: Record<string, string | undefined> | undefined;
  /** Test seam for the dedupe-nudge clock (ms epoch). */
  now?: (() => number) | undefined;
  /** Test seam for the dedupe-nudge state file path. */
  nudgeStatePath?: string | undefined;
  /** Test seam for the home dir used by the harness-in-use check. */
  homeDir?: string | undefined;
}

/**
 * Minimal UI surface for download progress, structurally typed so hosts/tests
 * that only offer `notify` (or nothing) still load engines. Every use is
 * best-effort and optional-called.
 */
interface ProgressUI {
  setStatus?(key: string, text: string | undefined): void;
  notify?(message: string, type?: "info" | "warning" | "error"): void;
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
  /** Latest UI surface seen from a context that triggered engine work; opportunistic, may go stale after /reload. */
  ui: ProgressUI | null;
  /** Human-readable in-flight download state for /reflex (null when idle). */
  downloadStatus: string | null;
}

const MB = 1024 * 1024;
function fmtBytes(n: number): string {
  return n >= MB ? `${(n / MB).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`;
}

/**
 * End-user recovery first: the overwhelmingly common cause is a network hiccup
 * during the lazy HF download, and a retry resumes from the partial on disk.
 * The dev escape hatches (offline export, local artifacts dir) come last.
 */
function engineUnavailableMessage(name: string, cause: string): string {
  return (
    `pi-reflex engine '${name}' unavailable: ${cause}. ` +
    "Check the connection and try again — interrupted artifact downloads resume from where they stopped; /reflex shows status. " +
    "Fully offline instead: generate artifacts with tools/export_onnx.py or set PI_REFLEX_ARTIFACTS."
  );
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
    ui: null,
    downloadStatus: null,
  };

  let lastProgressAt = 0;
  const showDownloadStatus = (text: string) => {
    slot.downloadStatus = text;
    try {
      slot.ui?.setStatus?.("pi-reflex", text);
    } catch {
      // Best-effort: /reflex always reports slot.downloadStatus regardless.
    }
  };
  const clearDownloadStatus = () => {
    slot.downloadStatus = null;
    try {
      slot.ui?.setStatus?.("pi-reflex", undefined);
    } catch {
      // ditto
    }
  };

  const loadNamed = async (name: EngineName): Promise<Engine> => {
    if (deps?.loadEngine) return deps.loadEngine(name);
    const local = findLocalEngine(name, slot.quant);
    if (local) {
      slot.source = `local:${local}`;
      return Engine.fromArtifacts(local, { int8: slot.quant === "int8" });
    }
    const label = `${name} (${slot.quant})`;
    try {
      const dir = await ensureEngine(name, {
        quant: slot.quant,
        onProgress: (info) => {
          const pct = info.total ? ` · ${Math.round((info.bytes / info.total) * 100)}%` : "";
          const text = `downloading ${label} — ${fmtBytes(info.bytes)}${info.total ? ` / ${fmtBytes(info.total)}` : ""}${pct}`;
          const now = Date.now();
          if (now - lastProgressAt < 1000) {
            slot.downloadStatus = text; // /reflex stays current without spamming the footer
            return;
          }
          lastProgressAt = now;
          showDownloadStatus(text);
        },
        onRetry: (info) =>
          showDownloadStatus(`downloading ${label} — retry ${info.attempt + 1} from ${fmtBytes(info.byteOffset)} (${info.error})`),
      });
      clearDownloadStatus();
      slot.source = `downloaded:${dir}`;
      slot.ui?.notify?.(`pi-reflex: downloaded ${label} artifacts from Hugging Face — loading engine`, "info");
      return Engine.fromArtifacts(dir, { int8: slot.quant === "int8" });
    } catch (e) {
      clearDownloadStatus();
      throw e;
    }
  };

  const getNamed = async (name: string): Promise<Engine> => {
    const cached = slot.engines.get(name);
    if (cached) return cached;
    const inFlight = slot.loading.get(name);
    if (inFlight) return inFlight;

    slot.errors.delete(name);
    const load = (async () => {
      try {
        const engine = await loadNamed(name as EngineName);
        slot.engines.set(name, engine);
        return engine;
      } catch (e) {
        const message = engineUnavailableMessage(name, (e as Error).message);
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
        "Check the connection and retry — interrupted artifact downloads resume from where they stopped; /reflex shows engine status. For offline use, generate artifacts with tools/export_onnx.py or set PI_REFLEX_ARTIFACTS to a local artifacts dir",
    } as unknown as JsonValue,
    isError: true,
    details: {},
  };
}

/**
 * Message for PI_REFLEX_EXPOSURE=deferred sessions without an active tool_search.
 * `codemodeActive` distinguishes "reachable from scripts that know the names but
 * undiscoverable" from "unreachable, period".
 */
function deferredWithoutToolSearch(codemodeActive: boolean): string {
  const fix = 'enable tool_search ("defaultTools": ["+codemode", "+tool_search"]) or set PI_REFLEX_EXPOSURE=codemode';
  if (codemodeActive) {
    return `pi-reflex: PI_REFLEX_EXPOSURE=deferred but tool_search is not active — the reflex tools are only callable from codemode scripts that already know their names. ${fix}.`;
  }
  return `pi-reflex: PI_REFLEX_EXPOSURE=deferred but neither tool_search nor codemode is active — the reflex tools are unreachable this session. ${fix}.`;
}

export default function activate(pi: ExtensionAPI, deps?: ExtensionDeps): void {
  const env = deps?.env ?? process.env;
  const { get, getNamed, slot } = makeSlot(deps);
  const cores = createToolCores(get, (ms) => (slot.lastLatencyMs = ms));

  // `codemode`: listed one line each in the codemode tool, callable from scripts.
  // `deferred`: not listed anywhere; tool_search finds and activates it (pi ≥ 1.0
  // keeps deferred tools across resume//reload). Both keep the tools out of the
  // model's tool list.
  const exposure =
    env.PI_REFLEX_EXPOSURE === "codemode" || env.PI_REFLEX_EXPOSURE === "deferred"
      ? (env.PI_REFLEX_EXPOSURE as "codemode" | "deferred")
      : undefined;
  // Unrecognized values used to fall back to `direct` silently — e.g. a user
  // setting `hidden` (a valid pi exposure this env does not support) would get the
  // tools declared to the model against their intent. Warn loudly instead;
  // `direct` is the documented explicit no-op and stays silent.
  if (
    env.PI_REFLEX_EXPOSURE &&
    env.PI_REFLEX_EXPOSURE !== "codemode" &&
    env.PI_REFLEX_EXPOSURE !== "deferred" &&
    env.PI_REFLEX_EXPOSURE !== "direct"
  ) {
    console.error(
      `pi-reflex: ignoring PI_REFLEX_EXPOSURE='${env.PI_REFLEX_EXPOSURE}' — expected codemode | deferred | direct; using direct`,
    );
  }

  // Startup banner on by default (PI_REFLEX_QUIET=1 silences); status lives in /reflex.
  // Same handler enforces the deferred contract: deferred tools are only reachable
  // via an active tool_search (or codemode scripts that already know the names), so
  // warn when tool_search is missing — and when nothing could reach the tools at
  // all, activate tool_search if the host registered it. Introspection APIs are
  // guarded so pi ≥ 0.99 hosts without them skip the check.
  pi.on("session_start", async (_event, ctx) => {
    slot.ui = ctx.ui;
    if (env.PI_REFLEX_QUIET !== "1") {
      ctx.ui.notify(
        "pi-reflex ready: classifier models (reflex/*), reflex/auto router, decision tools (engine loads on first use)",
        "info",
      );
    }
    // ── dedupe-corpus nudge (BENCHMARKS.md §A4): organic labels only exist when
    // /refine --proposer dedupe actually runs (the harness registers refinement
    // as its own /refine command — /harness has no refine case); remind on a
    // throttled interval when the continual harness is in use. Pure-throttle
    // decision in nudge.ts; every IO here is best-effort (a reminder must never
    // break startup).
    if (env.PI_REFLEX_QUIET !== "1") {
      const nudgeDays = parseNudgeDays(env);
      const nudgePath = deps?.nudgeStatePath ?? join(homedir(), ".pi-reflex", "dedupe-nudge.json");
      const now = deps?.now ?? Date.now;
      const state = await loadNudgeState(nudgePath);
      const home = deps?.homeDir ?? homedir();
      // pi print/JSON modes have no UI (ctx.hasUI === false) and their notify is
      // a no-op: stamping there would consume the 10-day reminder unseen and
      // suppress the next interactive one (review #4382004221).
      const headless = (ctx as { hasUI?: boolean }).hasUI === false;
      if (!headless && nudgeDays > 0 && shouldNudge(state?.lastShown ?? null, now(), nudgeDays) && harnessInUse(home)) {
        ctx.ui.notify(nudgeMessage(nudgeDays), "warning");
        await saveNudgeState(nudgePath, { lastShown: new Date(now()).toISOString() });
      }
    }
    if (exposure !== "deferred" || typeof pi.getActiveTools !== "function") return;
    const active = pi.getActiveTools();
    if (active.includes("tool_search")) return;
    if (active.includes("codemode")) {
      ctx.ui.notify(deferredWithoutToolSearch(true), "warning");
      return;
    }
    const hostHasToolSearch =
      typeof pi.getAllTools === "function" && pi.getAllTools().some((tool) => tool.name === "tool_search");
    if (hostHasToolSearch && typeof pi.setActiveTools === "function") {
      pi.setActiveTools([...active, "tool_search"]);
      ctx.ui.notify(
        `${deferredWithoutToolSearch(false)} Activated tool_search for this session so the deferred reflex tools stay reachable.`,
        "warning",
      );
      return;
    }
    ctx.ui.notify(deferredWithoutToolSearch(false), "warning");
  });

  // ── ② classifier provider ────────────────────────────────────────────────
  registerReflexProvider(pi, (modelId) => getNamed(modelId));

  // ── ③ virtual model reflex/auto ──────────────────────────────────────────
  // pi 1.1 interop: when a native llama.cpp decision model (api
  // "typesafe-system-one", provider "llama.cpp") is in the catalog, the router
  // classifies through the running llama-server; otherwise (or on failure) it
  // uses the local ONNX engine. PI_REFLEX_ROUTER_BACKEND=local pins local.
  let lastRouterBackend: { backend: "native" | "local" | "degraded"; nativeModel?: string; at: number } | null = null;
  pi.registerVirtualModel<ReflexAutoState>({
    provider: REFLEX_PROVIDER_ID,
    id: "auto",
    name: "Reflex Auto",
    thinkingLevels: ["low", "high"],
    async route(request, ctx) {
      slot.ui = (ctx as { ui?: ProgressUI }).ui ?? slot.ui;
      const sticky = request.reason !== "user" ? (request.failed ?? request.previous) : undefined;
      if (sticky) {
        return { model: sticky.model, thinkingLevel: sticky.thinkingLevel ?? "medium" };
      }
      const message = latestUserText(request.messages);
      const registry = ctx.modelRegistry as
        | {
            find: (provider: string, id: string) => unknown;
            getModelsOfType?: (type: "classifier") => readonly { provider: string; id: string; api: string }[];
            classify?: (model: never, context: never, options?: never) => Promise<unknown>;
          }
        | undefined;
      const out = await routeUserTurn(
        get,
        message,
        request.thinkingLevel,
        {
          env,
          find: (provider, id) => (registry?.find(provider, id) as Model<Api> | undefined) ?? undefined,
          native:
            registry?.getModelsOfType && registry.classify
              ? {
                  list: () => registry.getModelsOfType!("classifier"),
                  classify: (model, context, options) =>
                    registry.classify!(model as never, context as never, options as never) as Promise<ClassifierResult>,
                }
              : undefined,
        },
        request.previous?.model,
        (request as { signal?: AbortSignal }).signal,
      );
      lastRouterBackend = { backend: out.backend, ...(out.state.nativeModel ? { nativeModel: out.state.nativeModel } : {}), at: out.state.at };
      return { model: out.model, thinkingLevel: out.thinkingLevel, state: out.state };
    },
  });

  // ── ④ decision tools (namespace, annotations, structured output) ─────────
  // (exposure is resolved above so session_start can enforce the deferred contract)
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
    ...(exposure ? { exposure } : {}),
    parameters: REFLEX_TOOL_SCHEMAS.decide.parameters,
    outputSchema: REFLEX_TOOL_SCHEMAS.decide.outputSchema,
    async execute(_id, params, _signal?, _onUpdate?, ctx?) {
      if (ctx) slot.ui = ctx.ui;
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
    ...(exposure ? { exposure } : {}),
    parameters: REFLEX_TOOL_SCHEMAS.judge.parameters,
    outputSchema: REFLEX_TOOL_SCHEMAS.judge.outputSchema,
    async execute(_id, params, _signal?, _onUpdate?, ctx?) {
      if (ctx) slot.ui = ctx.ui;
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
    ...(exposure ? { exposure } : {}),
    parameters: REFLEX_TOOL_SCHEMAS.rate.parameters,
    outputSchema: REFLEX_TOOL_SCHEMAS.rate.outputSchema,
    async execute(_id, params, _signal?, _onUpdate?, ctx?) {
      if (ctx) slot.ui = ctx.ui;
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
    ...(exposure ? { exposure } : {}),
    parameters: REFLEX_TOOL_SCHEMAS.route.parameters,
    outputSchema: REFLEX_TOOL_SCHEMAS.route.outputSchema,
    async execute(_id, params, _signal?, _onUpdate?, ctx?) {
      if (ctx) slot.ui = ctx.ui;
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

  // ── ⑤b latency telemetry (pi ≥ 1.1 tool_execution_end.durationMs) ──────────
  // pi 1.1 records each execute() duration on the event; we aggregate per tool
  // (count, last, p50/p95 over a bounded ring) and surface it in /reflex. This is
  // the real-world counterpart of BENCHMARKS.md numbers — measured in actual
  // sessions, monotonic-clock based, and free (no engine instrumentation needed).
  // durationMs is absent for results that did not run (cached / nested) and on
  // hosts older than 1.1 — those events are simply skipped.
  const TELEMETRY_RING = 256;
  interface ToolLatencyStats {
    count: number;
    ring: number[];
  }
  const latency = new Map<string, ToolLatencyStats>();
  const recordLatency = (tool: string, ms: number) => {
    const s = latency.get(tool) ?? { count: 0, ring: [] };
    s.count += 1;
    s.ring.push(ms);
    if (s.ring.length > TELEMETRY_RING) s.ring.shift();
    latency.set(tool, s);
  };
  const percentile = (sorted: number[], p: number): number | undefined =>
    sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))] : undefined;
  const latencyLine = (): string => {
    if (!latency.size) return "no calls yet";
    return [...latency.entries()]
      .map(([tool, s]) => {
        const sorted = [...s.ring].sort((a, b) => a - b);
        const p50 = percentile(sorted, 50);
        const p95 = percentile(sorted, 95);
        return `${tool}: ${s.count} call${s.count === 1 ? "" : "s"} · p50 ${p50!.toFixed(0)}ms · p95 ${p95!.toFixed(0)}ms (last ${s.ring[s.ring.length - 1]!.toFixed(0)}ms)`;
      })
      .join("\n  ");
  };
  pi.on("tool_execution_end", (event) => {
    if (typeof event.durationMs !== "number") return;
    const tool = reflexToolFromName(event.toolName);
    if (tool) recordLatency(tool, event.durationMs);
  });

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
    pi.on("context_with_system", async (event, ctx) => {
      slot.ui = ctx.ui;
      const messages = await guard.process(event.messages);
      return messages ? { messages } : undefined;
    });
  }

  pi.registerCommand("reflex", {
    description: "pi-reflex engine + surfaces status",
    handler: async (_args, ctx) => {
      slot.ui = ctx.ui;
      const engine = slot.engines.get(slot.engineName);
      const status = engine
        ? `loaded (${slot.source})`
        : slot.errors.get(slot.engineName)
          ? `error: ${slot.errors.get(slot.engineName)}`
          : (slot.downloadStatus ?? "idle (loads on first use)");
      const tiers = ["SMALL", "MID", "FRONTIER"]
        .map((t) => env[`PI_REFLEX_TIER_${t}`] ?? "-")
        .join(" | ");
      const activeTools = typeof pi.getActiveTools === "function" ? pi.getActiveTools() : null;
      const exposureLine =
        exposure === "deferred" && activeTools
          ? ` · tool_search ${activeTools.includes("tool_search") ? "active" : "NOT active — deferred tools unreachable"}`
          : "";
      const backendMode = routerBackendFromEnv(env);
      const nativeSeen = lastRouterBackend?.nativeModel ? ` (${NATIVE_PROVIDER_ID}/${lastRouterBackend.nativeModel})` : "";
      const backendLine =
        `router backend: ${backendMode}` +
        (lastRouterBackend ? ` · last turn: ${lastRouterBackend.backend}${nativeSeen}` : " · no turns routed yet");
      const parts = [
        `engine: ${status}${slot.lastLatencyMs ? ` · last call ${slot.lastLatencyMs.toFixed(0)}ms` : ""}`,
        `latency: ${latencyLine()}`,
        `classifier: ${REFLEX_PROVIDER_ID}/multilingual|english|typed-decisions`,
        `reflex/auto tiers: ${tiers}`,
        backendLine,
        `exposure: ${exposure ?? "direct (declared to the model)"}${exposureLine}`,
        `guard: ${guard ? `${guard.stats.checked} checked, ${guard.stats.flagged} flagged${guard.stats.tripped ? " (tripped)" : ""}` : "off (PI_REFLEX_GUARD=1)"}`,
        `mcp: ${env.PI_REFLEX_MCP === "1" ? "registered" : "off (PI_REFLEX_MCP=1)"}`,
        `startup banner: ${env.PI_REFLEX_QUIET === "1" ? "off (PI_REFLEX_QUIET=1)" : "on"}`,
      ];
      ctx.ui.notify(`pi-reflex:\n${parts.join("\n")}`, "info");
    },
  });
}
