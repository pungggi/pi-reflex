/**
 * Router — port of laya's checkpoint routing decision (the model-loading part is
 * delegated to the inference session; this module is pure).
 *
 * Precedence: explicit model > explicit task > detected workflow (opt-in) >
 * explicit lang > detected script/language > default.
 */
import { analyse, type StateInput } from "../lang/analyze.js";
import type { Questions } from "../core/types.js";

export type ModelKey = "english" | "multilingual" | "typed-decisions";

export const BUNDLE_REPO = "convaiinnovations/laya";
export const DEFAULT_MODELS: Record<ModelKey, { repo: string; subfolder: string | null }> = {
  english: { repo: BUNDLE_REPO, subfolder: null },
  multilingual: { repo: BUNDLE_REPO, subfolder: "multilingual" },
  "typed-decisions": { repo: BUNDLE_REPO, subfolder: "typed-decisions" },
};

export const STANDALONE_MODELS: Record<ModelKey, string> = {
  english: "convaiinnovations/laya",
  multilingual: "convaiinnovations/laya-multilingual",
  "typed-decisions": "convaiinnovations/laya-typed-decisions",
};

const ALIASES: Record<string, ModelKey> = {
  en: "english",
  laya: "english",
  default: "english",
  multi: "multilingual",
  ml: "multilingual",
  "laya-multilingual": "multilingual",
  typed: "typed-decisions",
  typed_decisions: "typed-decisions",
  "laya-typed-decisions": "typed-decisions",
  decisions: "typed-decisions",
};

// Question-id signatures of the four typed-decisions workflows (opt-in only).
const TYPED_DECISION_WORKFLOWS: Record<string, Set<string>> = {
  agent_trace_observability: new Set(["action", "needs_review", "outcome", "risk", "urgency"]),
  customer_service: new Set(["action", "category", "churn_risk", "needs_human", "urgency"]),
  invoice_processing: new Set(["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
  security_incidents: new Set(["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
};

export function normaliseName(name: string): ModelKey {
  const key = String(name).trim().toLowerCase();
  const alias = ALIASES[key] ?? key;
  if (!(alias in DEFAULT_MODELS)) {
    throw new Error(`unknown model '${name}'; choose one of ${Object.keys(DEFAULT_MODELS).sort()} (or an alias: ${Object.keys(ALIASES).sort()})`);
  }
  return alias as ModelKey;
}

export function matchTypedDecisionsWorkflow(questions: Questions | null | undefined): string | null {
  if (!questions) return null;
  const ids = new Set(Object.keys(questions));
  for (const [wf, sig] of Object.entries(TYPED_DECISION_WORKFLOWS)) {
    if (ids.size === sig.size && [...ids].every((i) => sig.has(i))) return wf;
  }
  return null;
}

export interface RouteDecision {
  model: ModelKey;
  repo: string;
  reason: string;
  detection?: ReturnType<typeof analyse>;
  workflow?: string | null;
}

export interface RouteOptions {
  model?: string;
  task?: string;
  lang?: string;
  default?: string;
  autoTaskDetection?: boolean;
}

function repoStr(m: ModelKey): string {
  const spec = DEFAULT_MODELS[m];
  return spec.subfolder ? `${spec.repo}/${spec.subfolder}` : spec.repo;
}

export function route(state: StateInput, questions?: Questions | null, opts: RouteOptions = {}): RouteDecision {
  if (opts.model != null) {
    const key = normaliseName(opts.model);
    return { model: key, repo: repoStr(key), reason: `explicit model='${opts.model}'` };
  }

  if (opts.task != null) {
    const t = String(opts.task).toLowerCase().replace(/-/g, "_");
    const key = normaliseName(t === "typed_decisions" ? "typed-decisions" : t);
    return { model: key, repo: repoStr(key), reason: `explicit task='${opts.task}'` };
  }

  const workflow = matchTypedDecisionsWorkflow(questions);
  if (workflow && opts.autoTaskDetection) {
    return { model: "typed-decisions", repo: repoStr("typed-decisions"), reason: `question ids match the '${workflow}' typed-decisions workflow`, workflow };
  }

  if (opts.lang != null) {
    const base = String(opts.lang).toLowerCase().split("-")[0];
    const key: ModelKey = base === "en" || base === "eng" || base === "english" ? "english" : "multilingual";
    return { model: key, repo: repoStr(key), reason: `explicit lang='${opts.lang}'`, workflow };
  }

  const det = analyse(state);
  const defKey = normaliseName(opts.default ?? "english");
  if (det.script === "unknown") {
    return { model: defKey, repo: repoStr(defKey), reason: `no letters detected in state; using default (${defKey})`, detection: det, workflow };
  }
  if (det.script !== "latin") {
    return {
      model: "multilingual",
      repo: repoStr("multilingual"),
      reason: `non-Latin script (${det.script}, ${Math.round(100 * det.nonLatinFraction)}% of letters); the English checkpoint cannot read it`,
      detection: det,
      workflow,
    };
  }
  if (!det.isEnglish) {
    return { model: "multilingual", repo: repoStr("multilingual"), reason: `Latin script but language looks like '${det.language}', not English`, detection: det, workflow };
  }
  return { model: "english", repo: repoStr("english"), reason: "English Latin text", detection: det, workflow };
}
