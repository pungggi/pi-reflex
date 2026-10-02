/**
 * pi classifier provider (pi ≥ 0.99): registers pi-reflex engines as native
 * `type: "classifier"` catalog models under the `reflex` provider, next to
 * TypeSafe's hosted Jev models — but local, offline, and free.
 *
 * The pi classifier contract (`models.classify(model, { state, questions })`)
 * maps 1:1 onto `Engine.systemOne`: choice↔choice, score↔score, bool↔noul.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type {
  ClassifierAnswer,
  ClassifierApi,
  ClassifierContext,
  ClassifierQuestion,
  ClassifierResult,
  JsonObject,
  ProviderClassifier,
  Usage,
} from "@earendil-works/pi-ai";
import { Engine } from "../engine/engine.js";
import type { Questions, QuestionDef } from "../core/types.js";

export const REFLEX_PROVIDER_ID = "reflex";
export const REFLEX_CLASSIFIER_API = "reflex-onnx" as ClassifierApi;

/**
 * Placeholder endpoint. pi's provider composer ("baseUrl" is required when
 * defining custom models) demands a truthy baseUrl; the classifier path runs
 * the local ONNX engine and never performs HTTP against it.
 */
export const REFLEX_BASE_URL = "http://localhost/pi-reflex";

/** pi-reflex engines that appear as classifier models. */
export const CLASSIFIER_MODEL_IDS = ["multilingual", "english", "typed-decisions"] as const;
export type ClassifierModelId = (typeof CLASSIFIER_MODEL_IDS)[number];

/**
 * Declared context window. The engine truncates each question's sequence to its
 * `max_len` (default 512 tokens, from the artifacts' rl_agent_config.json) — this
 * is the honest catalog value for that budget, not ModernBERT's 8k positions.
 */
export const REFLEX_CONTEXT_WINDOW = 512;

/** bool → noul (laya's name for a calibrated P(true) question). */
export function mapQuestion(q: ClassifierQuestion): QuestionDef {
  switch (q.type) {
    case "choice":
      return { type: "choice", instructions: q.instructions, criteria: q.criteria };
    case "score":
      return { type: "score", instructions: q.instructions, criteria: q.criteria };
    case "bool":
      return { type: "noul", instructions: q.instructions, criteria: { true: q.criteria?.true, false: q.criteria?.false } };
  }
}

export function mapQuestions(questions: Record<string, ClassifierQuestion>): Questions {
  return Object.fromEntries(Object.entries(questions).map(([id, q]) => [id, mapQuestion(q)]));
}

/** noul → bool; choice/score pass through (pi drops the extra laya fields). */
export function mapAnswer(a: { type: string; choice?: string; probabilities?: Record<string, number>; confidence?: number; score?: number; noul?: number }): ClassifierAnswer {
  switch (a.type) {
    case "choice":
      return { type: "choice", choice: a.choice!, probabilities: a.probabilities!, confidence: a.confidence! };
    case "score":
      return { type: "score", score: a.score!, confidence: a.confidence! };
    default:
      return { type: "bool", probability: a.noul! };
  }
}

function usageOf(res: { usage: { input_tokens: number; output_tokens: number } }): Usage {
  const input = res.usage.input_tokens;
  const output = res.usage.output_tokens;
  return {
    input,
    output,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: input + output,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }, // local engine: no per-token price
  };
}

/**
 * The `classify` implementation pi calls for models with `api: "reflex-onnx"`.
 * Throws on engine failure; pi's ModelRuntime.classify converts that into an
 * error result (its contract never rejects).
 */
export function createReflexClassifier(engineFor: (modelId: string) => Promise<Engine>): ProviderClassifier {
  return {
    async classify(model, context: ClassifierContext): Promise<ClassifierResult> {
      const engine = await engineFor(model.id);
      const res = await engine.systemOne(context.state as JsonObject, mapQuestions(context.questions));
      const answers = Object.fromEntries(
        Object.entries(res.answers).map(([id, a]) => [id, mapAnswer(a)]),
      ) as Record<string, ClassifierAnswer>;
      return {
        api: REFLEX_CLASSIFIER_API,
        provider: REFLEX_PROVIDER_ID,
        model: model.id,
        answers,
        usage: usageOf(res),
        stopReason: "stop",
        timestamp: Date.now(),
      };
    },
  };
}

export function reflexClassifierModels() {
  return CLASSIFIER_MODEL_IDS.map((id) => ({
    type: "classifier" as const,
    id,
    name: `pi-reflex ${id}`,
    api: REFLEX_CLASSIFIER_API,
    input: ["text" as const],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: REFLEX_CONTEXT_WINDOW,
  }));
}

/**
 * Register the `reflex` provider with its three local classifier models.
 * `apiKey: "local"` + placeholder `baseUrl` satisfy pi's provider composition
 * (baseUrl is mandatory for custom models); classify dispatch never performs
 * HTTP — the implementation runs the ONNX engine.
 */
export function registerReflexProvider(pi: ExtensionAPI, engineFor: (modelId: string) => Promise<Engine>): void {
  pi.registerProvider(REFLEX_PROVIDER_ID, {
    name: "pi-reflex (local)",
    apiKey: "local",
    baseUrl: REFLEX_BASE_URL,
    models: reflexClassifierModels(),
    classifiers: { [REFLEX_CLASSIFIER_API]: createReflexClassifier(engineFor) },
  });
}
