#!/usr/bin/env python3
"""Append H22 public rows to the frozen H21 training artifact without reshuffling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

H21_SHA256 = "0c5ca272c3fe090c1e977671bddfd3654f73ea411fa842969c8776b8a3d76f16"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h21-train", type=Path, required=True)
    parser.add_argument("--public-curriculum", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if digest(args.h21_train) != H21_SHA256:
        raise RuntimeError("H21 training prefix hash mismatch")
    base = rows(args.h21_train)
    additions = rows(args.public_curriculum)
    if len(base) != 450 or len(additions) != 24:
        raise RuntimeError(f"expected 450+24 rows, got {len(base)}+{len(additions)}")
    if any(row.get("meta", {}).get("provenance") != "public-synthetic" for row in additions):
        raise RuntimeError("every H22 addition must be public-synthetic")
    ids = [row.get("meta", {}).get("source_id") for row in additions]
    if len(set(ids)) != 24 or any(not value or not value.startswith("naia-v11-") for value in ids):
        raise RuntimeError("H22 source IDs must be unique naia-v11 IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = args.output_dir / "train.jsonl"
    prefix = args.h21_train.read_bytes()
    suffix = args.public_curriculum.read_bytes()
    train.write_bytes(prefix + suffix)
    if train.read_bytes()[: len(prefix)] != prefix:
        raise RuntimeError("H21 prefix changed")

    manifest = {
        "schema_version": 1,
        "experiment": "H22",
        "rows": {"h21_prefix": 450, "public_synthetic_append": 24, "total": 474},
        "h21_prefix_sha256": H21_SHA256,
        "public_curriculum_sha256": digest(args.public_curriculum),
        "train_sha256": digest(train),
        "source_ids_in_order": ids,
        "split_or_shuffle": False,
        "alpha_or_private_persona_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
