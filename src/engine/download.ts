/**
 * Lazy engine-artifact downloader — HF repo + local cache (zero Python, zero install-time cost).
 *
 * Layout on the hub (one repo, one folder per checkpoint):
 *   <repo>/english/{model.int8.onnx|model.onnx(+.data), rl_agent_config.json, tokenizer/*}
 *   <repo>/multilingual/...   <repo>/typed-decisions/...
 *
 * Cache layout: $PI_REFLEX_ENGINES_DIR (default ~/.pi-reflex/engines)/<name>-<quant>/
 * Resolution order everywhere: $PI_REFLEX_ARTIFACTS (local export dir) → cache → download.
 */
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createHash } from "node:crypto";

export type EngineName = "english" | "multilingual" | "typed-decisions";
export type Quant = "int8" | "fp32";

export const DEFAULT_HF_REPO = "pungggi/pi-reflex-artifacts";

export const ENGINE_NAMES: readonly EngineName[] = ["english", "multilingual", "typed-decisions"];

export function enginesRoot(): string {
  return process.env.PI_REFLEX_ENGINES_DIR ?? join(homedir(), ".pi-reflex", "engines");
}

export function engineDir(name: EngineName, quant: Quant, root = enginesRoot()): string {
  return join(root, `${name}-${quant}`);
}

export function hfFileUrl(repo: string, name: EngineName, file: string): string {
  return `https://huggingface.co/${repo}/resolve/main/${name}/${file}`;
}

export function engineFiles(quant: Quant): string[] {
  const model = quant === "int8" ? ["model.int8.onnx"] : ["model.onnx", "model.onnx.data"];
  return [...model, "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"];
}

export function isEngineReady(dir: string, quant: Quant): boolean {
  try {
    return engineFiles(quant).every((f) => statSync(join(dir, f)).isFile());
  } catch {
    return false;
  }
}

/**
 * Resolve a ready engine directory without downloading: local artifacts override
 * ($PI_REFLEX_ARTIFACTS, e.g. a dev checkout with tools/export_onnx.py output),
 * then the shared cache. Returns null when only a download would satisfy it.
 */
export function findLocalEngine(name: EngineName, quant: Quant): string | null {
  const local = process.env.PI_REFLEX_ARTIFACTS;
  if (local && isEngineReady(join(local, name), quant)) return join(local, name);
  const cached = engineDir(name, quant);
  if (isEngineReady(cached, quant)) return cached;
  return null;
}

export interface DownloadOptions {
  repo?: string;
  quant?: Quant;
  root?: string;
  onProgress?: (info: { file: string; bytes: number; total?: number }) => void;
  fetchImpl?: typeof fetch;
}

async function downloadTo(url: string, dest: string, onProgress: DownloadOptions["onProgress"], fetchImpl: typeof fetch): Promise<void> {
  const res = await fetchImpl(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`download failed (${res.status}) for ${url}`);
  const total = res.headers.get("content-length");
  mkdirSync(dirname(dest), { recursive: true });
  const tmp = `${dest}.part`;
  let bytes = 0;
  const body = Readable.fromWeb(res.body as import("node:stream/web").ReadableStream);
  body.on("data", (chunk: Buffer) => {
    bytes += chunk.length;
    onProgress?.({ file: dest, bytes, total: total ? Number(total) : undefined });
  });
  await pipeline(body, createWriteStream(tmp));
  renameSync(tmp, dest);
}

/**
 * Ensure an engine's artifacts exist locally; download missing files from the HF repo.
 * Returns the directory ready for `Engine.fromArtifacts(dir, { quant })`.
 */
export async function ensureEngine(name: EngineName, opts: DownloadOptions = {}): Promise<string> {
  const quant = opts.quant ?? "int8";
  const repo = opts.repo ?? DEFAULT_HF_REPO;
  const local = findLocalEngine(name, quant);
  if (local) return local;

  const dir = engineDir(name, quant, opts.root);
  const fetchImpl = opts.fetchImpl ?? fetch;
  for (const file of engineFiles(quant)) {
    const dest = join(dir, file);
    if (existsSync(dest) && statSync(dest).isFile() && statSync(dest).size > 0) continue;
    await downloadTo(hfFileUrl(repo, name, file), dest, opts.onProgress, fetchImpl);
  }
  if (!isEngineReady(dir, quant)) throw new Error(`engine '${name}' (${quant}) incomplete after download in ${dir}`);
  return dir;
}

/** Content-addressed cache keys (companion pair caching). */
export function contentKey(...parts: string[]): string {
  const h = createHash("sha256");
  for (const p of parts) h.update(p, "utf8").update("\u0000");
  return h.digest("hex");
}
