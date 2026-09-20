# Third-party notices

pi-jev is Apache-2.0, but it builds on and derives from the work of others.
This file credits them; their license texts live in the locations noted.

## Laya (code + model weights) — Apache-2.0

- Code: portions of `reference/laya/` are vendored from
  https://github.com/NandhaKishorM/laya (license at `reference/laya/LICENSE`).
  The TypeScript runtime in `src/` is a clean-room port of that reference
  implementation's documented semantics, kept byte-compatible with it.
- Weights: the `artifacts/` ONNX graphs are converted from the Apache-2.0
  checkpoints published at https://huggingface.co/convaiinnovations/laya
  (English ModernBERT-large, multilingual mmBERT-base, typed-decisions).
- Research lineage: arXiv:2503.23303 and arXiv:2510.01237 by the laya author.

## ModernBERT — Apache-2.0 (Answer.AI / Warner et al., arXiv:2412.13663)

Backbone of the `english` and `typed-decisions` checkpoints.

## mmBERT — MIT (JHU-CLSP, https://github.com/JHU-CLSP/mmBERT)

Backbone of the `multilingual` checkpoint (256k Gemma-style tokenizer).

## onnxruntime-node — MIT (Microsoft)

Native ONNX inference for Node.js (`node_modules/onnxruntime-node/LICENSE`).

## @huggingface/tokenizers (tokenizers.js) — MIT (Hugging Face)

Pure-JS tokenizer engine used at runtime; also the `tokenizers` Rust crate
(MIT, Hugging Face) used in the dev-only export path.

---

Apache-2.0 requires preserving these notices in redistributions. Do not remove
this file, `LICENSE`, or `reference/laya/LICENSE` from derived distributions.
