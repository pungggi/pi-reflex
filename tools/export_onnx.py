#!/usr/bin/env python3
"""
DEV-ONLY: export laya checkpoints to fused ONNX graphs for pi-jev (TypeScript runtime).

This script never ships to npm users. We run it once per checkpoint; users download
the .onnx artifacts (fp32 + int8) from our artifact host. After this step, no Python
exists anywhere in pi-jev's dependency tree.

What it exports
---------------
laya's DecisionModel (encoder + type_emb + 2-layer head + scorer + act_head) as ONE
fused ONNX graph with dynamic axes:
    input_ids      int64  [batch, seq]
    attention_mask int64  [batch, seq]
    marker_pos     int64  [batch, kmax]
    marker_mask    bool   [batch, kmax]
    qtype          int64  [batch]
    -> option_logits float32 [batch, kmax]   (softmax + temperature happens in TS)
    -> act_logits     float32 [batch, n_act]

Usage
-----
    pip install torch transformers safetensors onnx onnxruntime numpy
    python tools/export_onnx.py --checkpoint convaiinnovations/laya --out artifacts/english
    python tools/export_onnx.py --checkpoint convaiinnovations/laya --subfolder multilingual --out artifacts/multilingual
    python tools/export_onnx.py --checkpoint convaiinnovations/laya --subfolder typed-decisions --out artifacts/typed-decisions

Also copies rl_agent_config.json (temperatures, max_len, head_max_len) next to the
graph so the TS runtime reads the same calibration constants laya uses.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

# Reuse laya's own architecture code (vendored under reference/laya, Apache 2.0).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference", "laya"))
from common import build_model, collate_items, build_sequence, render_options, QTYPES  # noqa: E402

MAX_LEN_DEFAULT = 512
HEAD_MAX_LEN_DEFAULT = 192


class ExportWrapper(torch.nn.Module):
    """Flatten DecisionModel outputs to plain tensors for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act_logits = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits, act_logits


def _fix_tokenizer_config(path: str):
    """Port of laya's patch: tokenizer_config must load across transformers versions."""
    import json as _json

    cfg_file = os.path.join(path, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tcfg = _json.load(f)
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
            changed = True
        if changed:
            with open(cfg_file, "w") as f:
                _json.dump(tcfg, f, indent=2)
    except Exception as e:
        print(f"[warn] tokenizer config fix skipped: {e}")


def load_checkpoint(checkpoint: str, subfolder: str | None):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    kw = {}
    if subfolder:
        kw["allow_patterns"] = [f"{subfolder}/*"]
    else:
        # English checkpoint lives at the repo ROOT: fetch only root-level files,
        # not the bundled multilingual/typed-decisions subfolders.
        kw["allow_patterns"] = ["*.json", "*.safetensors", "tokenizer/*", "encoder/*"]
    model_dir = snapshot_download(checkpoint, **kw)
    if subfolder:
        model_dir = os.path.join(model_dir, subfolder)

    _fix_tokenizer_config(model_dir)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    from transformers import AutoTokenizer

    _fix_tokenizer_config(model_dir)
    tok_dir = os.path.join(model_dir, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else cfg["encoder"])

    enc_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, cfg, tok, model_dir


def make_dummy_batch(tok, cfg):
    """A realistic batch: one choice (3 options), one noul, one score (4 levels)."""
    max_len = cfg.get("max_len", MAX_LEN_DEFAULT)
    head_max_len = cfg.get("head_max_len", HEAD_MAX_LEN_DEFAULT)

    def item(t, ins, crit, state):
        q = {"t": t, "ins": ins, "crit": crit}
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
        return {"ids": seq, "markers": markers, "qtype": QTYPES[t]}

    items = [
        item("choice", "Which department?", {"billing": "invoices", "tech": "bugs", "sales": "pricing"}, "please refund my invoice"),
        item("noul", "Is it urgent?", None, "production is down"),
        item("score", "How severe?", ["low", "medium", "high", "critical"], "minor ui glitch"),
    ]
    b = collate_items([items], tok.pad_token_id)
    return (
        b["input_ids"],
        b["attention_mask"],
        b["marker_pos"],
        b["marker_mask"],
        b["qtype"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--int8", action="store_true", help="also emit a dynamic-int8 quantized graph")
    args = ap.parse_args()

    model, cfg, tok, model_dir = load_checkpoint(args.checkpoint, args.subfolder)
    wrapper = ExportWrapper(model).to("cpu").eval()

    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "model.onnx")

    dummy = make_dummy_batch(tok, cfg)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["option_logits", "act_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "marker_pos": {0: "batch", 1: "kmax"},
                "marker_mask": {0: "batch", 1: "kmax"},
                "qtype": {0: "batch"},
                "option_logits": {0: "batch", 1: "kmax"},
                "act_logits": {0: "batch"},
            },
            opset_version=18,
            do_constant_folding=True,
        )

    # Copy calibration + layout constants for the TS runtime.
    shutil.copy(os.path.join(model_dir, "rl_agent_config.json"), os.path.join(args.out, "rl_agent_config.json"))

    # Sanity: parity between torch and onnx outputs on the dummy batch.
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {
        "input_ids": dummy[0].numpy(),
        "attention_mask": dummy[1].numpy(),
        "marker_pos": dummy[2].numpy(),
        "marker_mask": dummy[3].numpy().astype(bool),
        "qtype": dummy[4].numpy(),
    }
    ort_logits, ort_act = sess.run(None, feeds)
    with torch.no_grad():
        ref_logits, ref_act = wrapper(*dummy)
    lgap = float(np.abs(ort_logits - ref_logits.numpy()).max())
    agap = float(np.abs(ort_act - ref_act.numpy()).max())
    print(f"[ok] {onnx_path}  max |Δlogits|={lgap:.2e}  max |Δact|={agap:.2e}")
    # logits must be near-exact; act passes through softmax (exp amplifies fp drift), so looser
    if lgap > 1e-4 or agap > 5e-3:
        raise SystemExit("parity check failed")

    if args.int8:
        from onnx import load as onnx_load, save as onnx_save
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # The dynamo exporter leaves a stale intermediate shape annotation that contradicts
        # inference (1028 vs 256). Strip value_info — inputs/outputs keep their symbolic
        # dynamic axes — so the quantizer's shape-inference pre-pass passes cleanly.
        import onnx as _onnx

        clean_path = onnx_path + ".clean.onnx"
        m = _onnx.load(onnx_path)
        del m.graph.value_info[:]
        _onnx.save(m, clean_path)
        quantize_dynamic(clean_path, os.path.join(args.out, "model.int8.onnx"), weight_type=QuantType.QInt8)
        os.remove(clean_path)
        print(f"[ok] {os.path.join(args.out, 'model.int8.onnx')}")

    print("[done] artifacts:", os.listdir(args.out))


if __name__ == "__main__":
    main()
