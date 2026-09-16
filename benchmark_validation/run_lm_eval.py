"""Run pinned MMLU/LAMBADA protocols for manifest-listed HF causal LMs.

Use --dry-run first. Each benchmark is a separate lm-eval invocation because
MMLU is five-shot and LAMBADA is zero-shot. A DONE.json marker makes restarts
safe without interpreting lm-eval's version-dependent result filenames.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROTOCOLS = {
    "mmlu": {"task": "mmlu", "num_fewshot": 5},
    "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0},
}


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def model_args(row: dict, dtype: str, parallelize: bool) -> str:
    values = {
        "pretrained": row["model_id"],
        "tokenizer": row["tokenizer_id"],
        "backend": "causal",
        "dtype": dtype,
        "trust_remote_code": str(row["trust_remote_code"]).lower(),
        "add_bos_token": str(row["lm_eval_add_bos_token"]).lower(),
    }
    if parallelize:
        values["parallelize"] = "true"
    if row.get("revision"):
        values["revision"] = row["revision"]
    return ",".join(f"{key}={value}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("model_manifest.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("results") / "lm_eval")
    parser.add_argument("--tasks", nargs="+", choices=sorted(PROTOCOLS),
                        default=sorted(PROTOCOLS))
    parser.add_argument("--models", default=".*", help="Regex matched against model_key")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--parallelize", action="store_true")
    parser.add_argument("--limit", help="Smoke-test document count; never use for final scores")
    parser.add_argument("--write-out", action="store_true",
                        help="Print the first rendered prompts for protocol inspection")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Allow mutable HF main revisions (smoke tests only)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [row for row in load_manifest(args.manifest)
                if row["backend"] == "hf" and re.search(args.models, row["model_key"])]
    if not selected:
        raise SystemExit("No matching HF models in manifest")
    unpinned = [row["model_key"] for row in selected if not row.get("revision")]
    if unpinned and not args.allow_unpinned:
        raise SystemExit("Pin HF revisions first (or use --allow-unpinned for a smoke test): "
                         + ", ".join(unpinned))
    for row in selected:
        for protocol_name in args.tasks:
            protocol = PROTOCOLS[protocol_name]
            run_name = protocol_name if args.limit is None else f"{protocol_name}_limit_{args.limit}"
            destination = args.output / row["model_key"] / run_name
            done = destination / "DONE.json"
            if done.exists() and not args.force:
                print(f"SKIP {row['model_key']} {protocol_name}")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                "lm-eval", "run", "--model", "hf",
                "--model_args", model_args(row, args.dtype, args.parallelize),
                "--tasks", protocol["task"],
                "--num_fewshot", str(protocol["num_fewshot"]),
                "--batch_size", args.batch_size,
                "--device", args.device,
                "--output_path", str(destination),
                "--log_samples",
                "--show_config",
                "--seed", "0,1234,1234,1234",
            ]
            if args.limit is not None:
                command.extend(["--limit", args.limit])
            if args.write_out:
                command.append("--write_out")
            print("RUN", row["model_key"], protocol_name)
            print(subprocess.list2cmdline(command))
            if args.dry_run:
                continue
            subprocess.run(command, check=True)
            done.write_text(json.dumps({
                "model_key": row["model_key"], "protocol": protocol_name,
                "command": command,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
