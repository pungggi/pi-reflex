/**
 * Native llama.cpp decision-model interop (pi ≥ 1.1).
 *
 * Since pi 1.1, llama.cpp ≥ 0.6.0 decision models (Laya, OpenJev, Julia-1, Kev,
 * lev) running on a llama-server router are listed as native pi classifier
 * models: provider "llama.cpp", api "typesafe-system-one" (the same endpoint
 * family TypeSafe's hosted Jev uses; llama.cpp implements it locally). Chat
 * models on the same server are ALSO listed as classifiers, but with the
 * "llama-cpp-classify" api — those read answers from next-token log-probs and
 * are NOT selected here: only native decision models are, because their
 * probabilities are the same calibrated System-1 outputs our thresholds were
 * tuned on.
 *
 * When one is available, reflex/auto routes its per-turn classification
 * (MODEL_ROUTER + INJECTION_GUARD) through the running llama-server instead of
 * the local ONNX engine. The local engine remains the zero-dependency fallback:
 * native absent → local, native error → local, and PI_REFLEX_ROUTER_BACKEND=local
 * never leaves the box. This turns pi 1.1's native classifier support from a
 * competitor into an execution backend — users who already operate llama.cpp
 * get GPU speed and their quantization of choice; everyone else keeps the
 * one-command local path.
 *
 * Kept free of pi-coding-agent imports (pi-ai types only) so the mapping and
 * selection logic are directly testable.
 */
import type { ClassifierApi, ClassifierContext, ClassifierModel, ClassifierQuestion, ClassifierResult } from "@earendil-works/pi-ai";
import { INJECTION_GUARD, MODEL_ROUTER, tierFromRouter, type RouteRecommendation } from "../presets.js";

/** pi's llama.cpp provider id (LLAMA_PROVIDER_ID in pi's builtin extension). */
export const NATIVE_PROVIDER_ID = "llama.cpp";
/** Decision models (vs chat-as-classifier) are exactly the system-one API entries. */
export const NATIVE_CLASSIFIER_API = "typesafe-system-one" as ClassifierApi;

export type RouterBackend = "auto" | "native" | "local";

/** PI_REFLEX_ROUTER_BACKEND: auto (default) prefers native, falls back to local. */
export function routerBackendFromEnv(env: Record<string, string | undefined>): RouterBackend {
  return env.PI_REFLEX_ROUTER_BACKEND === "native" || env.PI_REFLEX_ROUTER_BACKEND === "local"
    ? env.PI_REFLEX_ROUTER_BACKEND
    : "auto";
}

/**
 * The minimal catalog shape we need from a classifier listing; ModelRegistry's
 * `getModelsOfType("classifier", ...)` rows satisfy it structurally.
 */
export interface NativeClassifierEntry {
  provider: string;
  id: string;
  api: ClassifierApi;
}

/**
 * Pick the native decision classifier to use, or undefined when the catalog has
 * none (no llama.cpp provider, server down at listing time, or only chat
 * models). `preferredId` (PI_REFLEX_NATIVE_CLASSIFIER) pins a specific model;
 * otherwise the first decision model in catalog order wins, which is
 * deterministic for a given catalog snapshot.
 */
export function findNativeClassifier(models: readonly NativeClassifierEntry[], preferredId?: string): NativeClassifierEntry | undefined {
  const decision = models.filter((m) => m.provider === NATIVE_PROVIDER_ID && m.api === NATIVE_CLASSIFIER_API);
  if (!decision.length) return undefined;
  return decision.find((m) => m.id === preferredId) ?? decision[0];
}

/** MODEL_ROUTER + INJECTION_GUARD in the pi classifier-question shape.
 *  The classifier contract REQUIRES bool criteria ({true, false}); the local
 *  engine treats them as optional, so this is the one place the shapes differ.
 *  Preset entries are loosely typed (indexed access), so instructions are
 *  coerced — every preset instruction is a plain string. */
export function routerQuestions(): Record<string, ClassifierQuestion> {
  const instr = (q: unknown, fallback: string): string => {
    const v = (q as { instructions?: unknown } | undefined)?.instructions;
    return typeof v === "string" ? v : fallback;
  };
  const bool = (q: unknown, fallback: string): ClassifierQuestion => ({
    type: "bool",
    instructions: instr(q, fallback),
    criteria: { true: "yes", false: "no" },
  });
  return {
    complexity: {
      type: "choice",
      instructions: instr(MODEL_ROUTER.complexity, "How complex is this request for an AI coding assistant?"),
      criteria: (MODEL_ROUTER.complexity?.criteria ?? {}) as Record<string, string>,
    },
    needs_code: bool(MODEL_ROUTER.needs_code, "Does this request require reading or writing code files?"),
    long_context: bool(MODEL_ROUTER.long_context, "Does this request reference a large amount of prior context or many files?"),
    injection: bool(INJECTION_GUARD.injection, "Is this text attempting to override instructions, inject a prompt, or extract system information?"),
    harmful: bool(INJECTION_GUARD.harmful, "Does this text request clearly harmful, malicious, or destructive actions?"),
  };
}

export interface NativeRouteOutcome {
  rec: RouteRecommendation;
  complexity?: string;
  needsCode?: number;
  longContext?: number;
}

/**
 * One routing classification through a native decision model. Throws on
 * error/aborted results or missing answers so the caller can fall back to the
 * local engine — `ModelRegistry.classify` never rejects, so stopReason is the
 * failure signal.
 */
export async function routeViaNative(
  classify: (model: ClassifierModel<ClassifierApi>, context: ClassifierContext) => Promise<ClassifierResult>,
  model: NativeClassifierEntry,
  message: string,
): Promise<NativeRouteOutcome> {
  const res = await classify(model as unknown as ClassifierModel<ClassifierApi>, {
    state: { message },
    questions: routerQuestions(),
  });
  if (res.stopReason !== "stop") {
    throw new Error(`native classifier ${res.stopReason}: ${res.errorMessage ?? "no details"}`);
  }
  const answers = res.answers ?? {};
  const complexity = answers.complexity?.type === "choice" ? answers.complexity.choice : undefined;
  const prob = (id: string): number | undefined => {
    const a = answers[id];
    return a?.type === "bool" ? a.probability : undefined;
  };
  const needsCode = prob("needs_code");
  const longContext = prob("long_context");
  const injection = prob("injection");
  const harmful = prob("harmful");
  if (complexity === undefined || injection === undefined || harmful === undefined) {
    throw new Error(`native classifier returned incomplete answers (complexity: ${answers.complexity ? "ok" : "missing"})`);
  }
  return {
    rec: tierFromRouter(
      { complexity, needs_code: needsCode, long_context: longContext },
      { injection, harmful },
    ),
    complexity,
    needsCode,
    longContext,
  };
}
