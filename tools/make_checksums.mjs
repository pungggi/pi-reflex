#!/usr/bin/env node
/**
 * Regenerate checksums.json — the pinned artifact digests shipped in the npm
 * package and enforced by src/engine/download.ts.
 *
 * Sources, per file on the HF repo:
 *   - LFS files: the sha256 is the LFS object id (api .../tree/main reports it);
 *     nothing is downloaded.
 *   - plain git files (configs, small tokenizers): downloaded once and hashed
 *     locally (~8 MB total).
 *
 * Run after any artifacts re-upload, then release — a mismatch between pins and
 * the hub makes downloads fail loudly instead of loading changed weights.
 *
 *   node tools/make_checksums.mjs [--repo ngSoftware/pi-reflex-artifacts]
 */
import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

const args = process.argv.slice(2);
const repoFlag = args.indexOf("--repo");
const repo = repoFlag >= 0 ? args[repoFlag + 1] : "ngSoftware/pi-reflex-artifacts";

const sha256OfBytes = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function fetchOk(url) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res;
}

const tree = await (await fetchOk(`https://huggingface.co/api/models/${repo}/tree/main?recursive=true`)).json();

const manifest = {};
for (const entry of tree) {
  if (entry.type !== "file" || entry.path === ".gitattributes") continue;
  if (entry.lfs?.oid) {
    manifest[entry.path] = { sha256: entry.lfs.oid, size: entry.size };
  } else {
    const bytes = Buffer.from(await (await fetchOk(`https://huggingface.co/${repo}/resolve/main/${entry.path}`)).arrayBuffer());
    if (bytes.length !== entry.size) throw new Error(`size drift for ${entry.path}: downloaded ${bytes.length}, tree says ${entry.size}`);
    manifest[entry.path] = { sha256: sha256OfBytes(bytes), size: bytes.length };
  }
}

const sorted = Object.fromEntries(Object.keys(manifest).sort().map((k) => [k, manifest[k]]));
const out = join(ROOT, "checksums.json");
writeFileSync(out, `${JSON.stringify(sorted, null, 2)}\n`);
console.log(`wrote ${out} (${Object.keys(sorted).length} files)`);
