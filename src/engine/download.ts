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
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createHash } from "node:crypto";

export type EngineName = "english" | "multilingual" | "typed-decisions";
export type Quant = "int8" | "fp32";

export const DEFAULT_HF_REPO = "ngSoftware/pi-reflex-artifacts";

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

export interface DownloadProgressInfo {
  file: string;
  bytes: number;
  total?: number;
}

export interface DownloadRetryInfo {
  file: string;
  /** 1-based number of the attempt that just failed. */
  attempt: number;
  /** Byte offset the next attempt resumes from (0 = the partial was discarded). */
  byteOffset: number;
  error: string;
}

export interface DownloadOptions {
  repo?: string;
  quant?: Quant;
  root?: string;
  onProgress?: (info: DownloadProgressInfo) => void;
  onRetry?: (info: DownloadRetryInfo) => void;
  fetchImpl?: typeof fetch;
  /** Backoff between in-call retries; tests pass 1 to keep suites fast. */
  retryDelayMs?: number;
}

const DOWNLOAD_ATTEMPTS = 3;
const DEFAULT_RETRY_DELAY_MS = 1500;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function partSize(tmp: string): number {
  try {
    return statSync(tmp).size;
  } catch {
    return 0;
  }
}

/**
 * Sidecar for an in-flight download: the etag/total of the response that produced
 * the current `.part` bytes. A resume must verify the upstream file is unchanged
 * (matching etag) before appending, or it would concatenate mismatched generations.
 */
function readPartMeta(metaPath: string): { etag?: string; total?: number } | null {
  try {
    return JSON.parse(readFileSync(metaPath, "utf8")) as { etag?: string; total?: number };
  } catch {
    // Partials from before the sidecar existed resume unverified; the
    // content-range offset check still guards against a stale offset.
    return null;
  }
}

/**
 * Stream one file to `dest`, resuming from `dest.part` across attempts.
 * Resume appends only when the server honored the Range (206) and the etag still
 * matches the partial's sidecar meta; a 200 (or an incompatible 206) restarts
 * from zero rather than risk mixing bytes. Transient failures (network resets,
 * 5xx/429) retry within the call; permanent 4xx fail fast. Either way the
 * `.part` stays on disk, so a later call picks up where this one stopped.
 */
async function downloadTo(
  url: string,
  dest: string,
  opts: { onProgress?: DownloadOptions["onProgress"]; onRetry?: DownloadOptions["onRetry"]; fetchImpl: typeof fetch; retryDelayMs: number },
): Promise<void> {
  const { onProgress, onRetry, fetchImpl, retryDelayMs } = opts;
  mkdirSync(dirname(dest), { recursive: true });
  const tmp = `${dest}.part`;
  const metaPath = `${tmp}.meta`;

  for (let attempt = 1; ; attempt++) {
    const existing = partSize(tmp);
    const meta = existing > 0 ? readPartMeta(metaPath) : null;
    const headers: Record<string, string> = existing > 0 ? { range: `bytes=${existing}-` } : {};
    let res: Response;
    try {
      res = await fetchImpl(url, { redirect: "follow", headers });
    } catch (e) {
      const cause = (e as Error).message ?? String(e);
      if (attempt >= DOWNLOAD_ATTEMPTS) throw new Error(`download failed (${cause}) for ${url}`);
      onRetry?.({ file: dest, attempt, byteOffset: existing, error: cause });
      await sleep(retryDelayMs * attempt);
      continue;
    }
    // The partial is past the file's end (upstream changed, or the previous attempt
    // finished writing but died before the rename): it cannot be resumed.
    if (res.status === 416 && existing > 0) {
      rmSync(tmp, { force: true });
      rmSync(metaPath, { force: true });
      if (attempt >= DOWNLOAD_ATTEMPTS) throw new Error(`download failed (416 for a ${existing}-byte partial) for ${url}`);
      onRetry?.({ file: dest, attempt, byteOffset: 0, error: "stale partial (416) — discarding" });
      await sleep(retryDelayMs);
      continue;
    }
    if (!res.ok || !res.body) {
      const transient = res.status >= 500 || res.status === 429;
      if (!transient || attempt >= DOWNLOAD_ATTEMPTS) throw new Error(`download failed (${res.status}) for ${url}`);
      onRetry?.({ file: dest, attempt, byteOffset: existing, error: `HTTP ${res.status}` });
      await sleep(retryDelayMs * attempt);
      continue;
    }

    const etag = res.headers.get("etag") ?? undefined;
    const contentLength = res.headers.get("content-length");
    const contentRange = res.headers.get("content-range");
    let start = 0;
    let total = contentLength ? Number(contentLength) : undefined;
    if (existing > 0 && res.status === 206) {
      const match = /^bytes (\d+)-(\d+)\/(\d+|\*)$/.exec(contentRange ?? "");
      const sameFile = !meta?.etag || !etag || meta.etag === etag;
      if (sameFile && (match ? Number(match[1]) === existing : true)) {
        start = existing;
        if (match && match[3] !== "*") total = Number(match[3]);
        else if (contentLength) total = existing + Number(contentLength);
      } else {
        // The server honored the Range but the partial does not match the current
        // file (etag change or offset mismatch). Appending would mix generations,
        // and a fresh write of a 206 body would miss its first bytes — discard and
        // re-request from scratch.
        rmSync(tmp, { force: true });
        rmSync(metaPath, { force: true });
        if (attempt >= DOWNLOAD_ATTEMPTS) throw new Error(`download failed (incompatible partial after ${attempt} attempts) for ${url}`);
        onRetry?.({ file: dest, attempt, byteOffset: 0, error: "partial does not match the current file — restarting" });
        continue;
      }
    }
    if (total !== undefined && (!Number.isFinite(total) || total <= 0)) total = undefined;

    // Persist BEFORE streaming so a mid-stream death leaves the next attempt
    // enough to verify its resume against.
    try {
      writeFileSync(metaPath, JSON.stringify({ etag, total: total ?? (contentLength ? start + Number(contentLength) : undefined) }));
    } catch {
      // Best-effort: without the sidecar the next attempt resumes unverified.
    }

    let bytes = start;
    const body = Readable.fromWeb(res.body as import("node:stream/web").ReadableStream);
    body.on("data", (chunk: Buffer) => {
      bytes += chunk.length;
      onProgress?.({ file: dest, bytes, total });
    });
    try {
      await pipeline(body, createWriteStream(tmp, { flags: start > 0 ? "a" : "w" }));
    } catch (e) {
      const cause = (e as Error).message ?? String(e);
      if (attempt >= DOWNLOAD_ATTEMPTS) throw new Error(`download failed (${cause}) for ${url}`);
      onRetry?.({ file: dest, attempt, byteOffset: partSize(tmp), error: cause });
      await sleep(retryDelayMs * attempt);
      continue;
    }
    renameSync(tmp, dest);
    rmSync(metaPath, { force: true });
    return;
  }
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
    await downloadTo(hfFileUrl(repo, name, file), dest, {
      onProgress: opts.onProgress,
      onRetry: opts.onRetry,
      fetchImpl,
      retryDelayMs: opts.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS,
    });
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
