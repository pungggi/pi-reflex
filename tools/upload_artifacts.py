#!/usr/bin/env python3
"""
DEV-ONLY: upload exported engine artifacts to the HF artifacts repo that
`ensureEngine` downloads from (default: pungggi/pi-reflex-artifacts).

One-time setup (per machine):
    hf auth login            # or: set HF_TOKEN=<write-capable token>
    python tools/upload_artifacts.py            # int8 (the downloader default)
    python tools/upload_artifacts.py --fp32     # optionally add fp32 graphs

Repo layout created (matches src/engine/download.ts):
    <repo>/english/{model.int8.onnx, rl_agent_config.json, tokenizer/*}
    <repo>/multilingual/...  <repo>/typed-decisions/...
"""
import argparse
import os
import sys

ENGINES = ["english", "multilingual", "typed-decisions"]
INT8_FILES = ["model.int8.onnx", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
FP32_FILES = ["model.onnx", "model.onnx.data", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_REPO", "pungggi/pi-reflex-artifacts"))
    ap.add_argument("--artifacts", default=os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    ap.add_argument("--fp32", action="store_true", help="upload fp32 graphs too (default: int8 only)")
    ap.add_argument("--private", action="store_true", help="create the repo private (requires the user's plan to allow)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub (tools/.venv has it)")

    api = HfApi()
    who = api.whoami()
    print(f"[auth] {who['name']}")

    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)

    files = INT8_FILES + (FP32_FILES if args.fp32 else [])
    uploaded = 0
    for engine in ENGINES:
        for rel in files:
            local = os.path.join(args.artifacts, engine, *rel.split("/"))
            if not os.path.isfile(local):
                print(f"[skip] {engine}/{rel} (not exported locally)")
                continue
            path_in_repo = f"{engine}/{rel}"
            api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=args.repo, repo_type="model")
            size_mb = os.path.getsize(local) / 1e6
            print(f"[ok] {path_in_repo} ({size_mb:.1f} MB)")
            uploaded += 1
    if uploaded == 0:
        sys.exit("nothing uploaded — run tools/export_onnx.py first")
    print(f"[done] {uploaded} files → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
