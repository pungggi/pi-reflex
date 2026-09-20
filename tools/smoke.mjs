// Scratch smoke test: real tokenizer + real ONNX graph, no test framework.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-node";

const art = join(dirname(fileURLToPath(import.meta.url)), "..", "artifacts", "english");

// 1. tokenizer
const tokJSON = JSON.parse(readFileSync(`${art}/tokenizer/tokenizer.json`, "utf8"));
const added = Object.fromEntries(tokJSON.added_tokens.map((t) => [t.content, t.id]));
console.log("special tokens:", {
  cls: added["[CLS]"],
  sep: added["[SEP]"],
  mask: added["[MASK]"],
  pad: added["[PAD]"],
});

const tok = new Tokenizer(tokJSON, JSON.parse(readFileSync(`${art}/tokenizer/tokenizer_config.json`, "utf8")));
const enc = tok.encode("choice question: route this");
const ids = enc.ids.map(Number);
console.log("encode (no specials):", ids.slice(0, 8), "...", "tokens:", enc.tokens.slice(0, 4));

// 2. ONNX session
const session = await ort.InferenceSession.create(`${art}/model.onnx`);
console.log("inputs:", session.inputNames, "outputs:", session.outputNames);

const B = 1, L = 48, K = 4;
const inputIds = new BigInt64Array(B * L).fill(0n);
const att = new BigInt64Array(B * L).fill(0n);
const seq = [0, 100, 102, 1, 2, 3, 4, 102]; // cls head sep mask opts sep
for (let i = 0; i < seq.length; i++) { inputIds[i] = BigInt(seq[i]); att[i] = 1n; }
const markerPos = new BigInt64Array(B * K).fill(0n);
const markerMask = new Uint8Array(B * K).fill(0);
markerPos[0] = 4n; markerMask[0] = 1; markerPos[1] = 5n; markerMask[1] = 1; markerPos[2] = 6n; markerMask[2] = 1;
const qtype = new BigInt64Array(B).fill(0n);

const feeds = {
  input_ids: new ort.Tensor("int64", inputIds, [B, L]),
  attention_mask: new ort.Tensor("int64", att, [B, L]),
  marker_pos: new ort.Tensor("int64", markerPos, [B, K]),
  marker_mask: new ort.Tensor("bool", markerMask, [B, K]),
  qtype: new ort.Tensor("int64", qtype, [B]),
};
const t0 = performance.now();
const out = await session.run(feeds);
const dt = performance.now() - t0;
console.log("option_logits shape:", out.option_logits.dims, "act_logits shape:", out.act_logits.dims);
console.log("first run:", dt.toFixed(1), "ms");
const t1 = performance.now();
for (let i = 0; i < 5; i++) await session.run(feeds);
console.log("warm avg:", ((performance.now() - t1) / 5).toFixed(1), "ms");

// int8 variant
const s8 = await ort.InferenceSession.create(`${art}/model.int8.onnx`);
const o8 = await s8.run(feeds);
const a = out.option_logits.data, b = o8.option_logits.data;
let maxd = 0;
for (let i = 0; i < a.length; i++) maxd = Math.max(maxd, Math.abs(a[i] - b[i]));
console.log("int8 vs fp32 max |Δlogit|:", maxd.toFixed(4));
console.log("SMOKE OK");
