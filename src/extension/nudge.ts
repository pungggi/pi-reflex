// Periodic nudge to grow the organic dedupe corpus (CONTRACT-harness §4).
//
// Why: the A4 evals (BENCHMARKS.md §A4) showed the curated corpus is a
// stopgap — the real D1 calibration data comes from `harness-refinement`
// entries with proposer "dedupe", and zero such runs existed in the maintainer's
// entire session history. The corpus only grows when someone actually runs
// `/harness refine --proposer dedupe`. This module reminds them, throttled.
//
// Policy (documented, tested):
// - interval: PI_REFLEX_DEDUPE_NUDGE_DAYS (default 10, 0 disables, invalid → default)
// - suppression: PI_REFLEX_QUIET=1 silences the startup nudge like the banner
// - relevance: only nudges when the continual harness is in use on this machine
//   (its durable state files exist); nagging harness-less users is spam
// - state: last-shown timestamp in ~/.pi-reflex/dedupe-nudge.json; every write
//   is best-effort — a read/write failure must never break session_start

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export const NUDGE_DEFAULT_INTERVAL_DAYS = 10;
const STATE_FILE = "dedupe-nudge.json";
const DAY_MS = 24 * 60 * 60 * 1000;

/** Resolve the interval from the env; 0 disables, malformed values fall back. */
export function parseNudgeDays(env: Record<string, string | undefined>): number {
  const raw = env.PI_REFLEX_DEDUPE_NUDGE_DAYS;
  if (raw === undefined || raw === "") return NUDGE_DEFAULT_INTERVAL_DAYS;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) {
    console.error(
      `pi-reflex: ignoring PI_REFLEX_DEDUPE_NUDGE_DAYS='${raw}' — expected a non-negative integer (days); using ${NUDGE_DEFAULT_INTERVAL_DAYS}`,
    );
    return NUDGE_DEFAULT_INTERVAL_DAYS;
  }
  return Math.floor(n);
}

/** Pure throttle decision: nudge when never shown or the interval has elapsed. */
export function shouldNudge(lastShownIso: string | null, nowMs: number, intervalDays: number): boolean {
  if (intervalDays <= 0) return false;
  if (!lastShownIso) return true;
  const last = Date.parse(lastShownIso);
  if (!Number.isFinite(last)) return true; // unreadable state: nudge rather than never-nudge
  return nowMs - last >= intervalDays * DAY_MS;
}

/** Harness in use on this machine? Its durable layer lives in ~/.pi/agent/. */
export function harnessInUse(homeDir: string = homedir()): boolean {
  const base = join(homeDir, ".pi", "agent");
  return existsSync(join(base, "harness-state.md")) || existsSync(join(base, "harness-state"));
}

export interface NudgeState {
  lastShown: string;
}

/** Best-effort read; any failure (missing, malformed, unreadable) → null. */
export async function loadNudgeState(path: string): Promise<NudgeState | null> {
  try {
    const raw = await readFile(path, "utf8");
    const parsed = JSON.parse(raw) as Partial<NudgeState>;
    return typeof parsed.lastShown === "string" ? { lastShown: parsed.lastShown } : null;
  } catch {
    return null;
  }
}

/** Best-effort write; failures are swallowed so session_start never breaks. */
export async function saveNudgeState(path: string, state: NudgeState): Promise<void> {
  try {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(state), "utf8");
  } catch {
    /* never break startup over a reminder stamp */
  }
}

/** The message a due nudge should show. */
export function nudgeMessage(intervalDays: number): string {
  return (
    "pi-reflex: organic dedupe labels improve D1 calibration and the A4 benchmarks (BENCHMARKS.md §A4). " +
    "Consider running `/harness refine --proposer dedupe` in this project, then `/harness export-corpus`. " +
    `Next reminder in ${intervalDays} days (PI_REFLEX_DEDUPE_NUDGE_DAYS=0 to disable).`
  );
}
