import { mkdtempSync, rmSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { contentKey, engineDir, engineFiles, ensureEngine, findLocalEngine, hfFileUrl, isEngineReady } from "../src/engine/download.js";

describe("download — pure path/URL logic", () => {
  it("cache dir layout + engine file sets", () => {
    expect(engineDir("multilingual", "int8", "/cache")).toBe(join("/cache", "multilingual-int8"));
    expect(engineFiles("int8")).toContain("model.int8.onnx");
    expect(engineFiles("int8")).not.toContain("model.onnx.data");
    expect(engineFiles("fp32")).toContain("model.onnx.data");
  });
  it("hf URL building", () => {
    expect(hfFileUrl("pungggi/pi-reflex-artifacts", "english", "tokenizer/tokenizer.json")).toBe(
      "https://huggingface.co/pungggi/pi-reflex-artifacts/resolve/main/english/tokenizer/tokenizer.json",
    );
  });
  it("contentKey is stable and order-sensitive", () => {
    expect(contentKey("a", "b")).toBe(contentKey("a", "b"));
    expect(contentKey("a", "b")).not.toBe(contentKey("b", "a"));
  });
  it("findLocalEngine honors PI_REFLEX_ARTIFACTS when complete", () => {
    process.env.PI_REFLEX_ARTIFACTS = join(__dirname, "fixtures");
    try {
      expect(findLocalEngine("english", "int8")).toBeNull(); // fixture dir has no engine files
    } finally {
      delete process.env.PI_REFLEX_ARTIFACTS;
    }
  });
});

describe("download — ensureEngine with mocked fetch", () => {
  const root = mkdtempSync(join(tmpdir(), "pi-reflex-dl-"));
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  const BODIES: Record<string, string> = {
    "english/model.int8.onnx": "FAKEQUANTIZEDWEIGHTS",
    "english/rl_agent_config.json": '{"temperature": [1.1, 1.2, 1.05], "max_len": 512}',
    "english/tokenizer/tokenizer.json": '{"added_tokens": []}',
    "english/tokenizer/tokenizer_config.json": '{"cls_token": "[CLS]"}',
  }; // engine-keyed: only english exists on the fake hub
  let calls = 0;
  const fetchImpl = (async (url: string | URL | Request) => {
    calls++;
    const u = String(url);
    const file = u.split("/").slice(7).join("/"); // after .../resolve/main/ → "<engine>/<file>"
    const body = BODIES[file];
    if (body === undefined) return new Response("not found", { status: 404 });
    return new Response(body, { headers: { "content-length": String(body.length) } });
  }) as typeof fetch;

  it("downloads all files, verifies completeness, skips on second call", async () => {
    const dir = await ensureEngine("english", { root, quant: "int8", fetchImpl, repo: "pungggi/pi-reflex-artifacts" });
    expect(isEngineReady(dir, "int8")).toBe(true);
    expect(readFileSync(join(dir, "rl_agent_config.json"), "utf8")).toContain("temperature");
    const firstCalls = calls;
    const again = await ensureEngine("english", { root, quant: "int8", fetchImpl });
    expect(again).toBe(dir);
    expect(calls).toBe(firstCalls); // nothing re-downloaded
  });

  it("404 surfaces a clear error", async () => {
    await expect(
      ensureEngine("typed-decisions", { root, quant: "int8", fetchImpl, repo: "pungggi/pi-reflex-artifacts" }),
    ).rejects.toThrow(/download failed \(404\)/);
  });

  it("missing env artifacts leave no partial local hit", () => {
    expect(existsSync(join(root, "typed-decisions-int8", "rl_agent_config.json"))).toBe(false);
  });
});
