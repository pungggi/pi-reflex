import { describe, expect, it } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  harnessInUse,
  loadNudgeState,
  nudgeMessage,
  NUDGE_DEFAULT_INTERVAL_DAYS,
  parseNudgeDays,
  saveNudgeState,
  shouldNudge,
} from "../src/extension/nudge.js";

const DAY_MS = 24 * 60 * 60 * 1000;
const T0 = Date.parse("2026-10-26T12:00:00Z");

describe("parseNudgeDays", () => {
  it("defaults to 10 days when unset", () => {
    expect(parseNudgeDays({})).toBe(NUDGE_DEFAULT_INTERVAL_DAYS);
    expect(parseNudgeDays({ PI_REFLEX_DEDUPE_NUDGE_DAYS: "" })).toBe(NUDGE_DEFAULT_INTERVAL_DAYS);
  });

  it("parses positive integers", () => {
    expect(parseNudgeDays({ PI_REFLEX_DEDUPE_NUDGE_DAYS: "1" })).toBe(1);
    expect(parseNudgeDays({ PI_REFLEX_DEDUPE_NUDGE_DAYS: "30" })).toBe(30);
  });

  it("0 disables the nudge", () => {
    expect(parseNudgeDays({ PI_REFLEX_DEDUPE_NUDGE_DAYS: "0" })).toBe(0);
  });

  it("rejects malformed and negative values with a fallback", () => {
    const err = console.error;
    console.error = () => {};
    try {
      expect(parseNudgeDays({ PI_REFLEX_DEDUPE_NUDGE_DAYS: "abc" })).toBe(NUDGE_DEFAULT_INTERVAL_DAYS);
      expect(parseNudgeDays({ PI_REFLEX_DEDUPE_NUDGE_DAYS: "-3" })).toBe(NUDGE_DEFAULT_INTERVAL_DAYS);
      expect(parseNudgeDays({ PI_REFLEX_DEDUPE_NUDGE_DAYS: "2.5" })).toBe(2); // floors, still a valid interval
    } finally {
      console.error = err;
    }
  });
});

describe("shouldNudge (pure throttle)", () => {
  it("nudges when never shown", () => {
    expect(shouldNudge(null, T0, 10)).toBe(true);
  });

  it("suppresses within the interval", () => {
    const recent = new Date(T0 - 9 * DAY_MS).toISOString();
    expect(shouldNudge(recent, T0, 10)).toBe(false);
  });

  it("fires at exactly the interval boundary", () => {
    const boundary = new Date(T0 - 10 * DAY_MS).toISOString();
    expect(shouldNudge(boundary, T0, 10)).toBe(true);
  });

  it("never fires when disabled (0 days)", () => {
    expect(shouldNudge(null, T0, 0)).toBe(false);
  });

  it("nudges on unreadable state rather than silencing forever", () => {
    expect(shouldNudge("not-a-date", T0, 10)).toBe(true);
  });
});

describe("nudge state IO (best-effort)", () => {
  it("round-trips through a file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "nudge-"));
    const path = join(dir, "nested", "dedupe-nudge.json");
    expect(await loadNudgeState(path)).toBeNull(); // missing → null, no throw
    await saveNudgeState(path, { lastShown: "2026-10-26T00:00:00Z" });
    expect(await loadNudgeState(path)).toEqual({ lastShown: "2026-10-26T00:00:00Z" });
    await rm(dir, { recursive: true, force: true });
  });

  it("returns null for malformed content instead of crashing", async () => {
    const dir = await mkdtemp(join(tmpdir(), "nudge-"));
    const path = join(dir, "dedupe-nudge.json");
    await writeFile(path, "{not json", "utf8");
    expect(await loadNudgeState(path)).toBeNull();
    await writeFile(path, JSON.stringify({ other: 1 }), "utf8");
    expect(await loadNudgeState(path)).toBeNull();
    await rm(dir, { recursive: true, force: true });
  });

  it("save swallows failures (parent is a file) without throwing", async () => {
    const alt = await mkdtemp(join(tmpdir(), "nudge-"));
    await writeFile(join(alt, "not-a-dir"), "x", "utf8");
    await expect(
      saveNudgeState(join(alt, "not-a-dir", "state.json"), { lastShown: "x" }),
    ).resolves.toBeUndefined();
    await rm(alt, { recursive: true, force: true });
  });
});

describe("harnessInUse", () => {
  it("detects global and per-project durable state", async () => {
    const dir = await mkdtemp(join(tmpdir(), "nudge-"));
    expect(harnessInUse(dir)).toBe(false);
    const { mkdir } = await import("node:fs/promises");
    await mkdir(join(dir, ".pi", "agent", "harness-state"), { recursive: true });
    expect(harnessInUse(dir)).toBe(true);
    await rm(join(dir, ".pi", "agent", "harness-state"), { recursive: true });
    await writeFile(join(dir, ".pi", "agent", "harness-state.md"), "# state", "utf8");
    expect(harnessInUse(dir)).toBe(true);
    await rm(dir, { recursive: true, force: true });
  });
});

describe("nudgeMessage", () => {
  it("names the command, the benefit, and the off switch", () => {
    const m = nudgeMessage(10);
    expect(m).toContain("/refine --proposer dedupe");
    expect(m).not.toContain("/harness refine"); // /harness has no refine case — refinement is its own /refine command
    expect(m).toContain("/harness export-corpus");
    expect(m).toContain("PI_REFLEX_DEDUPE_NUDGE_DAYS=0");
    expect(m).toContain("10 days");
  });
});
