/**
 * reflex/auto — a pi virtual model (pi ≥ 0.99) that routes each user turn to a
 * physical model tier chosen by the local System 1 engine in one ~50–200 ms pass.
 *
 * Tier → model mapping comes from the environment:
 *   PI_REFLEX_TIER_SMALL=provider/model-id
 *   PI_REFLEX_TIER_MID=provider/model-id
 *   PI_REFLEX_TIER_FRONTIER=provider/model-id
 * Missing tiers fall back to the previous physical model (or the router errors
 * with a setup hint). Continuations and retries stay sticky on the turn's model
 * so prompt caches and thinking signatures stay valid.
 */
import type { Api, Message, Model, ModelThinkingLevel } from "@earendil-works/pi-ai";
import { Engine } from "../engine/engine.js";
import type { RouteRecommendation } from "../presets.js";
import { routeRaw } from "./cores.js";

export type Tier = "small" | "mid" | "frontier";

export interface ReflexAutoState {
  tier: Tier;
  reason: string;
  guards: { injection: number; harmful: number };
  at: number;
}

const TIER_ENV: Record<Tier, string> = {
  small: "PI_REFLEX_TIER_SMALL",
  mid: "PI_REFLEX_TIER_MID",
  frontier: "PI_REFLEX_TIER_FRONTIER",
};

/** "anthropic/claude-sonnet-4-5" → { provider: "anthropic", id: "claude-sonnet-4-5" }. */
export function parseModelRef(ref: string): { provider: string; id: string } | undefined {
  const i = ref.indexOf("/");
  if (i <= 0 || i >= ref.length - 1) return undefined;
  return { provider: ref.slice(0, i), id: ref.slice(i + 1) };
}

export function tierFromEnv(env: Record<string, string | undefined>, tier: Tier): string | undefined {
  return env[TIER_ENV[tier]];
}

/** Effort bump: a "high" virtual thinking level escalates one tier. */
export function bumpTier(tier: Tier, highEffort: boolean): Tier {
  if (!highEffort) return tier;
  return tier === "small" ? "mid" : "frontier";
}

export function tierThinkingLevel(tier: Tier): ModelThinkingLevel {
  return tier === "small" ? "low" : tier === "mid" ? "medium" : "high";
}

/** Latest user-message text (string or text blocks), skipping system/assistant/tool traffic. */
export function latestUserText(messages: readonly Message[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === "user") {
      const c = m.content;
      if (typeof c === "string") return c;
      const text = c.filter((b): b is { type: "text"; text: string } => b.type === "text").map((b) => b.text).join("\n");
      return text || undefined;
    }
  }
  return undefined;
}

export interface RouteUserTurnDeps {
  env: Record<string, string | undefined>;
  find: (provider: string, id: string) => Model<Api> | undefined;
}

export interface RouteUserTurnResult {
  model: Model<Api>;
  thinkingLevel: ModelThinkingLevel;
  state: ReflexAutoState;
}

/**
 * The routing decision for a user turn: classify → tier → physical model.
 * Pure (no pi imports) so it is directly testable. Fallback semantics (PR#2 reviews #2/#3):
 * - degraded routes (engine failure / nothing to classify) stay `mid` — the effort bump
 *   never applies, so a dead router cannot select the most expensive model;
 * - an unmapped (possibly bumped) tier falls back to the PREVIOUS physical model before
 *   the original recommendation's tier, so an active frontier/mid conversation is never
 *   silently downgraded to a small model;
 * - with no previous model, the original tier's mapping is the last resort, then a
 *   descriptive error pi surfaces as a failed route.
 */
export async function routeUserTurn(
  getEngine: () => Promise<Engine>,
  message: string | undefined,
  effort: ModelThinkingLevel,
  deps: RouteUserTurnDeps,
  previous?: Model<Api>,
): Promise<RouteUserTurnResult> {
  let rec: RouteRecommendation;
  let degraded = false;
  if (message) {
    try {
      const out = await routeRaw(getEngine, message);
      rec = out.rec;
    } catch {
      degraded = true;
      rec = { tier: "mid", reason: "engine unavailable — defaulting to mid tier", guards: { injection: 0, harmful: 0 } };
    }
  } else {
    degraded = true;
    rec = { tier: "mid", reason: "no user message to classify — defaulting to mid tier", guards: { injection: 0, harmful: 0 } };
  }
  const highEffort = effort === "high" || effort === "xhigh" || effort === "max";
  const tier = degraded ? rec.tier : bumpTier(rec.tier, highEffort);

  const tryFind = (t: Tier): Model<Api> | undefined => {
    const ref = tierFromEnv(deps.env, t);
    const parsed = ref ? parseModelRef(ref) : undefined;
    return parsed ? deps.find(parsed.provider, parsed.id) : undefined;
  };

  const now = Date.now();
  const routed = tryFind(tier);
  if (routed) {
    return { model: routed, thinkingLevel: tierThinkingLevel(tier), state: { tier, reason: rec.reason, guards: rec.guards, at: now } };
  }
  if (previous) {
    return {
      model: previous,
      thinkingLevel: effort,
      state: { tier, reason: `${rec.reason}; tier "${tier}" unmapped — staying on the previous model`, guards: rec.guards, at: now },
    };
  }
  const original = tryFind(rec.tier);
  if (original) {
    return {
      model: original,
      thinkingLevel: tierThinkingLevel(rec.tier),
      state: { tier: rec.tier, reason: `${rec.reason}; tier "${tier}" unmapped — using the ${rec.tier} mapping`, guards: rec.guards, at: now },
    };
  }
  throw new Error(
    `pi-reflex reflex/auto: tier "${tier}" has no model. Set ${TIER_ENV[tier]}=provider/model-id` +
      ` (e.g. anthropic/claude-haiku-4-5).`,
  );
}
