"""Generate deterministic HumanEval(+) solutions for manifest-listed HF models.

This script only generates code. Execute and score the generated programs in
the EvalPlus Docker sandbox as documented in README.md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("model_manifest.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("results") / "evalplus")
    parser.add_argument("--models", default=".*", help="Regex matched against model_key")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Allow mutable HF main revisions (smoke tests only)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    selected = [row for row in rows if row["backend"] == "hf"
                and re.search(args.models, row["model_key"])]
    if not selected:
        raise SystemExit("No matching HF models in manifest")
    unpinned = [row["model_key"] for row in selected if not row.get("revision")]
    if unpinned and not args.allow_unpinned:
        raise SystemExit("Pin HF revisions first (or use --allow-unpinned for a smoke test): "
                         + ", ".join(unpinned))
    args.output.mkdir(parents=True, exist_ok=True)
    for row in selected:
        model_path = row["model_id"]
        if row.get("revision") and not args.dry_run:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(row["model_id"], revision=row["revision"],
                                           cache_dir=args.cache_dir)
        command = [
            "evalplus.codegen",
            "--model", model_path,
            "--dataset", "humaneval",
            "--backend", "hf",
            "--root", str(args.output),
            "--greedy",
            "--force-base-prompt",
            "--trust-remote-code", str(row["trust_remote_code"]),
            "--dtype", args.dtype,
            "--attn-implementation", args.attn_implementation,
            "--device-map", args.device_map,
        ]
        print("RUN", row["model_key"], "humaneval")
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
