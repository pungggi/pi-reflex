import { mkdtempSync, mkdirSync, rmSync, readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { contentKey, engineDir, engineFiles, ensureEngine, findLocalEngine, hfFileUrl, isEngineReady } from "../src/engine/download.js";

describe("download — pure path/URL logic", () => {
  it("cache dir layout + engine file sets", () => {
    expect(engineDir("multilingual", "int8", "/cache")).toBe(join("/cache", "multilingual-int8"));
    expect(engineFiles("int8")).toContain("model.int8.onnx");
    expect(engineFiles("int8")).not.toContain("model.onnx.data");
    expect(engineFiles("fp32")).toContain("model.onnx.data");
  });
  it("hf URL building", () => {
    expect(hfFileUrl("ngSoftware/pi-reflex-artifacts", "english", "tokenizer/tokenizer.json")).toBe(
      "https://huggingface.co/ngSoftware/pi-reflex-artifacts/resolve/main/english/tokenizer/tokenizer.json",
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
    const dir = await ensureEngine("english", { root, quant: "int8", fetchImpl, repo: "ngSoftware/pi-reflex-artifacts" });
    expect(isEngineReady(dir, "int8")).toBe(true);
    expect(readFileSync(join(dir, "rl_agent_config.json"), "utf8")).toContain("temperature");
    const firstCalls = calls;
    const again = await ensureEngine("english", { root, quant: "int8", fetchImpl });
    expect(again).toBe(dir);
    expect(calls).toBe(firstCalls); // nothing re-downloaded
  });

  it("404 surfaces a clear error", async () => {
    await expect(
      ensureEngine("typed-decisions", { root, quant: "int8", fetchImpl, repo: "ngSoftware/pi-reflex-artifacts" }),
    ).rejects.toThrow(/download failed \(404\)/);
  });

  it("missing env artifacts leave no partial local hit", () => {
    expect(existsSync(join(root, "typed-decisions-int8", "rl_agent_config.json"))).toBe(false);
  });
});

describe("download — Range resume across interrupted attempts", () => {
  const root = mkdtempSync(join(tmpdir(), "pi-reflex-resume-"));
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  const FULL = "FAKEQUANTIZEDWEIGHTS";
  const ETAG = '"w1"';
  const PART_BYTES = 10;
  let dir: string;
  let part: string;
  let meta: string;
  let modelCalls = 0;
  let secondRequestRange: string | undefined;

  /** engine files other than the model download trivially (shared by the mocks) */
  const smallBody = (url: string) => new Response("{}", { headers: { "content-length": "2", etag: ETAG } });

  beforeEach(() => {
    // ensureEngine always works inside engineDir(name, quant, root) — seed there.
    dir = engineDir("english", "int8", root);
    part = join(dir, "model.int8.onnx.part");
    meta = `${part}.meta`;
    modelCalls = 0;
    secondRequestRange = undefined;
    rmSync(dir, { recursive: true, force: true });
  });

  function seedPartial(withMeta: boolean, etag = ETAG) {
    mkdirSync(dir, { recursive: true });
    writeFileSync(part, FULL.slice(0, PART_BYTES));
    if (withMeta) writeFileSync(meta, JSON.stringify({ etag, total: FULL.length }));
  }

  it("resumes a partial from its byte offset when the server honors Range", async () => {
    // start empty; attempt 1 fails mid-stream to test pipeline error propagation
    const progress: { bytes: number; total?: number }[] = [];
    const retries: { byteOffset: number; error: string }[] = [];
    const fetchImpl = (async (url: string | URL | Request, init?: RequestInit) => {
      const u = String(url);
      if (!u.includes("model.int8.onnx")) return smallBody(u);
      modelCalls++;
      if (modelCalls === 1) {
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(Buffer.from(FULL.slice(0, PART_BYTES)));
              setTimeout(() => controller.error(new TypeError("fetch failed: ECONNRESET")), 10);
            },
          }),
          { status: 200, headers: { "content-length": String(FULL.length), etag: ETAG } }
        );
      }
      if (secondRequestRange === undefined) secondRequestRange = (init?.headers as Record<string, string> | undefined)?.range;
      const offset = existsSync(part) ? statSync(part).size : 0;
      return new Response(FULL.slice(offset), {
        status: 206,
        headers: {
          "content-length": String(FULL.length - offset),
          "content-range": `bytes ${offset}-${FULL.length - 1}/${FULL.length}`,
          etag: ETAG,
        },
      });
    }) as typeof fetch;

    const result = await ensureEngine("english", {
      root,
      quant: "int8",
      fetchImpl,
      retryDelayMs: 1,
      onProgress: (i) => {
        if (i.file.includes("model.int8.onnx")) progress.push({ bytes: i.bytes, total: i.total });
      },
      onRetry: (i) => retries.push({ byteOffset: i.byteOffset, error: i.error }),
    });

    expect(isEngineReady(result, "int8")).toBe(true);
    expect(readFileSync(join(result, "model.int8.onnx"), "utf8")).toBe(FULL);
    expect(secondRequestRange).toBe(`bytes=${PART_BYTES}-`);
    expect(retries).toHaveLength(1);
    expect(retries[0].byteOffset).toBe(PART_BYTES);
    expect(retries[0].error).toContain("ECONNRESET");
    expect(progress.at(-1)).toMatchObject({ bytes: FULL.length, total: FULL.length });
    // success cleans up the transient files and brings the rest of the engine
    expect(existsSync(part)).toBe(false);
    expect(existsSync(meta)).toBe(false);
    expect(existsSync(join(result, "tokenizer/tokenizer.json"))).toBe(true);
  });

  it("a 200 response after a partial restarts from scratch instead of appending", async () => {
    seedPartial(false); // pre-sidecar partial (older pi-reflex) — resume is offset/etag-unverified
    const fetchImpl = (async (url: string | URL | Request) => {
      const u = String(url);
      if (!u.includes("model.int8.onnx")) return smallBody(u);
      return new Response(FULL, { headers: { "content-length": String(FULL.length), etag: ETAG } });
    }) as typeof fetch;

    const result = await ensureEngine("english", { root, quant: "int8", fetchImpl, retryDelayMs: 1 });
    expect(readFileSync(join(result, "model.int8.onnx"), "utf8")).toBe(FULL);
  });

  it("an etag mismatch restarts rather than concatenating file generations", async () => {
    seedPartial(true, '"old"');
    const fetchImpl = (async (url: string | URL | Request) => {
      const u = String(url);
      if (!u.includes("model.int8.onnx")) return smallBody(u);
      const offset = existsSync(part) ? statSync(part).size : 0;
      return new Response(FULL.slice(offset), {
        status: 206,
        headers: {
          "content-length": String(FULL.length - offset),
          "content-range": `bytes ${offset}-${FULL.length - 1}/${FULL.length}`,
          etag: ETAG,
        },
      });
    }) as typeof fetch;

    const result = await ensureEngine("english", { root, quant: "int8", fetchImpl, retryDelayMs: 1 });
    expect(readFileSync(join(result, "model.int8.onnx"), "utf8")).toBe(FULL);
    expect(existsSync(join(result, "rl_agent_config.json"))).toBe(true);
  });

  it("a stale partial past EOF (416) is discarded and the download restarts", async () => {
    seedPartial(true);
    let modelCalls = 0;
    const fetchImpl = (async (url: string | URL | Request) => {
      const u = String(url);
      if (!u.includes("model.int8.onnx")) return smallBody(u);
      modelCalls++;
      if (modelCalls === 1) {
        // pretend the partial exceeds the current file size
        return new Response("range not satisfiable", { status: 416 });
      }
      return new Response(FULL, { headers: { "content-length": String(FULL.length), etag: ETAG } });
    }) as typeof fetch;

    const result = await ensureEngine("english", { root, quant: "int8", fetchImpl, retryDelayMs: 1 });
    expect(readFileSync(join(result, "model.int8.onnx"), "utf8")).toBe(FULL);
    expect(modelCalls).toBe(2);
  });
});
