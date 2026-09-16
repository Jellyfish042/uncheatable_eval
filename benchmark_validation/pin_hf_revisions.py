"""Resolve remote HF model IDs to immutable commit revisions in a manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("model_manifest.jsonl"))
    parser.add_argument("--token", help="HF token; defaults to the configured login")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    api = HfApi(token=args.token)
    failed = []
    for row in rows:
        if row["backend"] not in {"hf", "mistral3"} or Path(row["model_id"]).exists():
            continue
        try:
            row["revision"] = api.model_info(row["model_id"], revision="main").sha
            print(row["model_key"], row["revision"])
        except Exception as error:  # gated/private repositories may require login
            failed.append((row["model_key"], str(error)))
    with args.manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if failed:
        details = "\n".join(f"  {name}: {error}" for name, error in failed)
        raise SystemExit("Could not pin every remote model:\n" + details)


if __name__ == "__main__":
    main()
