"""Build a conventional-benchmark manifest from one leaderboard release.

The leaderboard JSON files are the source of truth for checkpoint, tokenizer,
parameter count, and the prefix convention used by Uncheatable Eval. Identical
model records across categories are collapsed and inconsistent records fail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def backend(model_id: str, tokenizer_id: str) -> str:
    name = Path(model_id).name.lower()
    if name.startswith("rwkv") or tokenizer_id.startswith("rwkv_vocab"):
        return "rwkv"
    if name.startswith("ministral-3-"):
        return "mistral3"
    return "hf"


def safe_name(model_id: str) -> str:
    return model_id.rstrip("/\\").replace("\\", "/").split("/")[-1].removesuffix(".pth")


def build(release_dir: Path) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted(release_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(safe_name(row["model_name_or_path"]), []).append(row)
    records = []
    fields = ("model_name_or_path", "tokenizer_name", "parameters count", "bos_mode")
    for name, rows in grouped.items():
        signatures = {tuple(str(row.get(field)) for field in fields) for row in rows}
        if len(signatures) != 1:
            raise ValueError(f"Inconsistent model configuration across categories: {name}")
        row = rows[0]
        model_id, tokenizer_id = row["model_name_or_path"], row["tokenizer_name"]
        bos_mode = row["bos_mode"]
        records.append({
            "model_key": name,
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "parameters_b": row["parameters count"],
            "uncheatable_bos_mode": bos_mode,
            "backend": backend(model_id, tokenizer_id),
            # lm-eval does not implement the newline/EOS prefix modes. Mainline
            # conventional scores therefore use standard HF prompt tokenization.
            "lm_eval_add_bos_token": bos_mode == "add_default_bos",
            "trust_remote_code": bool(row.get("model_args", {}).get("trust_remote_code", True)),
            "revision": row.get("model_revision") or row.get("revision")
                        or row.get("model_args", {}).get("revision"),
            "status": "ready" if backend(model_id, tokenizer_id) == "hf" else "adapter_required",
        })
    return sorted(records, key=lambda x: (float(x["parameters_b"]), x["model_key"].lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("model_manifest.jsonl"))
    args = parser.parse_args()
    records = build(args.release_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {key: sum(row["backend"] == key for row in records)
              for key in sorted({row["backend"] for row in records})}
    print(f"Wrote {len(records)} models to {args.output}: {counts}")


if __name__ == "__main__":
    main()
