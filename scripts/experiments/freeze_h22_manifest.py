#!/usr/bin/env python3
"""Create the fail-closed H22 pre-GPU provenance and leakage manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h21-train", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--public-builder", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--blind-suite", type=Path, required=True)
    parser.add_argument("--suite", type=Path, action="append", default=[])
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--train-code", type=Path, required=True)
    parser.add_argument("--eval-code", type=Path, required=True)
    parser.add_argument("--validator-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefix = args.h21_train.read_bytes()
    source = args.source.read_bytes()
    train = args.train.read_bytes()
    if train != prefix + source:
        raise RuntimeError("H22 train must be the exact H21 byte prefix plus exact public source bytes")
    rows = jsonl(args.train)
    if len(rows) != 474 or len(jsonl(args.source)) != 24:
        raise RuntimeError("H22 row contract requires exactly 450+24 rows")
    serialized = args.train.read_text(encoding="utf-8")
    if any(marker.casefold() in serialized.casefold() for marker in ("alpha", "알파", "cafelua", "luke", "루크")):
        raise RuntimeError("private/Alpha marker found in H22 training data")
    prefix_rows = jsonl(args.h21_train)
    source_rows = jsonl(args.source)
    prefix_prompts = {
        message["content"]
        for row in prefix_rows
        for message in row["messages"]
        if message.get("role") == "user"
    }
    curriculum_prompts = {
        message["content"]
        for row in source_rows
        for message in row["messages"]
        if message.get("role") == "user"
    }
    suites = [*args.suite, args.blind_suite]
    curriculum_overlaps: dict[str, list[str]] = {}
    inherited_prefix_overlaps: dict[str, list[str]] = {}
    for suite in suites:
        payload = json.loads(suite.read_text(encoding="utf-8"))
        cases = payload["prompts"] if isinstance(payload, dict) else payload
        suite_prompts = {case["prompt"] for case in cases}
        curriculum_overlaps[str(suite)] = sorted(curriculum_prompts & suite_prompts)
        inherited_prefix_overlaps[str(suite)] = sorted(prefix_prompts & suite_prompts)
    if any(curriculum_overlaps.values()):
        raise RuntimeError(f"exact H22 curriculum/evaluation prompt overlap found: {curriculum_overlaps}")
    audit = json.loads(args.independent_audit.read_text(encoding="utf-8"))
    if audit.get("decision") != "PASS":
        raise RuntimeError("independent pre-GPU audit is not PASS")

    artifact_paths = {
        "h21_train": args.h21_train, "train": args.train, "source": args.source,
        "holdout": args.holdout, "public_builder": args.public_builder,
        "compiler": args.compiler, "blind_suite": args.blind_suite,
        "independent_audit": args.independent_audit, "train_code": args.train_code,
        "eval_code": args.eval_code, "validator_code": args.validator_code,
        **{f"development_suite_{index}": path for index, path in enumerate(args.suite, 1)},
    }
    manifest = {
        "schema_version": 1, "experiment": "H22", "pre_gpu_freeze": True,
        "rows": {"h21_prefix": 450, "public_synthetic_append": 24, "total": 474},
        "byte_identity": {"train_equals_h21_prefix_plus_source": True},
        "alpha_or_private_persona_used": False,
        "exact_prompt_overlap": {
            "h22_curriculum": curriculum_overlaps,
            "inherited_h21_prefix_disclosed": inherited_prefix_overlaps,
        },
        "artifacts": {name: {"path": str(path), "sha256": digest(path)} for name, path in artifact_paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
