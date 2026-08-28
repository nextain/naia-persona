#!/usr/bin/env python3
"""Compile consented persona/conversation JSONL into immutable train/eval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_dataset import validate

ALLOWED_SOURCE_TYPES = {"persona", "conversation"}


def canonical(row: dict) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compile_rows(source: Path, seed: str, holdout_ratio: float) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    source_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            meta = row.get("meta", {})
            source_id = meta.get("source_id")
            source_type = meta.get("source_type")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"line {line_number}: meta.source_id is required")
            if source_id in source_ids:
                raise ValueError(f"line {line_number}: duplicate meta.source_id")
            if source_type not in ALLOWED_SOURCE_TYPES:
                raise ValueError(f"line {line_number}: meta.source_type must be persona or conversation")
            source_ids.add(source_id)
            rows.append(row)

    ordered = sorted(rows, key=lambda row: row["meta"]["source_id"])
    train: list[dict] = []
    holdout: list[dict] = []
    boundary = int(holdout_ratio * 10_000)
    for row in ordered:
        source_id = row["meta"]["source_id"]
        bucket = int(sha256(f"{seed}:{source_id}".encode("utf-8"))[:8], 16) % 10_000
        (holdout if bucket < boundary else train).append(row)
    if len(ordered) >= 2 and not holdout:
        holdout.append(train.pop())
    if len(ordered) >= 2 and not train:
        train.append(holdout.pop())
    return train, holdout


def write_jsonl(path: Path, rows: list[dict]) -> tuple[int, str]:
    data = b"".join(canonical(row) for row in rows)
    path.write_bytes(data)
    return len(data), sha256(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", default="naia-persona-v1")
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    if not 0 < args.holdout_ratio < 1:
        parser.error("--holdout-ratio must be between 0 and 1")
    if not args.source.is_file():
        print(json.dumps({"ok": False, "errors": ["source file not found"]}))
        return 2

    rows, _, errors = validate(args.source)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False))
        return 1
    try:
        train, holdout = compile_rows(args.source, args.seed, args.holdout_ratio)
    except (ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False))
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, split in (("train", train), ("holdout", holdout)):
        size, digest = write_jsonl(args.output_dir / f"{name}.jsonl", split)
        artifacts[name] = {"file": f"{name}.jsonl", "rows": len(split), "bytes": size, "sha256": digest}
    manifest = {
        "schema_version": 1,
        "dataset_name": args.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compiler": "naia-persona/compile_dataset.py:v1",
        "source": {"rows": rows, "sha256": sha256(args.source.read_bytes())},
        "split": {"seed": args.seed, "holdout_ratio": args.holdout_ratio},
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.json").write_bytes(canonical(manifest))
    print(json.dumps({"ok": True, "manifest": manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
