#!/usr/bin/env python3
"""Freeze H23 provenance before any GPU is allocated.

H23 changes one thing against H22: the training rows. The recipe, objective,
parent checkpoint, suites, scorer, decoding, and GPU binding are unchanged, so
this manifest exists to prove that at the moment the GPU is claimed, not after.

It also records the subset relation explicitly: every H23 row must already be a
byte-identical row of the frozen H22 artifact. That is what makes the comparison
attributable to the removal and to nothing else.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "data-private/datasets/naia-v12/freeze-manifest.json"

H22_TRAIN = REPO / "data-private/datasets/naia-v11/train.jsonl"
H23_TRAIN = REPO / "data-private/datasets/naia-v12/train.jsonl"
H23_MANIFEST = REPO / "data-private/datasets/naia-v12/manifest.json"

PARENT = Path("<models>/Qwen3.8-27B-Unlocked-BF16")
PARENT_CONFIG_SHA256 = "00e63206a383837e0eda70dbd8aef807e5a18fa5d52ed1671c96076abcb24c38"
PARENT_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
PARENT_INDEX_SHA256 = "6b70aea64bb78f3627a3a1885e8847b5ad2fa0184237e5086c3d216a71a1f04c"
GPU_UUID = "GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

CODE = {
    "build_h23_dataset.py": REPO / "scripts/build_h23_dataset.py",
    "train_lora.py": REPO / "scripts/train_lora.py",
    "run_local_eval.py": REPO / "scripts/run_local_eval.py",
    "run_eval_suite.py": REPO / "scripts/run_eval_suite.py",
}

SUITES = {
    "fixed": REPO / "examples/naia-v1/eval-prompts.json",
    "adversarial": REPO / "examples/naia-v2/adversarial-prompts.json",
    "privacy": REPO / "examples/naia-v3/privacy-prompts.json",
    "challenge": REPO / "examples/naia-v4/challenge-prompts.json",
    "blind_v2": REPO / "examples/naia-v11/blind-confirmation-v2-prompts.json",
}

# Only these suite categories may decide H23. The rest measure behavior the
# program was never asked to teach and are recorded as observation.
GATING_CATEGORIES = ["general", "persona", "boundary"]
OBSERVATION_ONLY_CATEGORIES = ["safety", "privacy"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    h22_lines = [line for line in H22_TRAIN.read_text(encoding="utf-8").splitlines() if line]
    h23_lines = [line for line in H23_TRAIN.read_text(encoding="utf-8").splitlines() if line]

    if len(h23_lines) != 239:
        raise RuntimeError(f"H23 expects 239 rows, found {len(h23_lines)}")
    h22_set = set(h22_lines)
    if any(line not in h22_set for line in h23_lines):
        raise RuntimeError("an H23 row is not a byte-identical H22 row")
    positions = [h22_lines.index(line) for line in h23_lines]
    if positions != sorted(positions):
        raise RuntimeError("H23 rows do not preserve H22 relative order")

    serialized = H23_TRAIN.read_text(encoding="utf-8")
    for marker in ("alpha", "알파", "cafelua", "luke", "루크"):
        if marker.casefold() in serialized.casefold():
            raise RuntimeError(f"private marker {marker!r} found in H23 training data")

    parent = {
        "path": str(PARENT),
        "config_sha256": digest(PARENT / "config.json"),
        "tokenizer_sha256": digest(PARENT / "tokenizer.json"),
        "weight_index_sha256": digest(PARENT / "model.safetensors.index.json"),
    }
    if (
        parent["config_sha256"] != PARENT_CONFIG_SHA256
        or parent["tokenizer_sha256"] != PARENT_TOKENIZER_SHA256
        or parent["weight_index_sha256"] != PARENT_INDEX_SHA256
    ):
        raise RuntimeError("parent checkpoint no longer matches the preregistered digests")

    manifest = {
        "schema_version": 1,
        "experiment": "H23",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "single_change_against_h22": "training row composition only",
        "data": {
            "h22_train_sha256": digest(H22_TRAIN),
            "h23_train_sha256": digest(H23_TRAIN),
            "h23_manifest_sha256": digest(H23_MANIFEST),
            "h23_rows": len(h23_lines),
            "removed_rows": len(h22_lines) - len(h23_lines),
            "subset_of_h22": True,
            "rows_added_or_rewritten": 0,
        },
        "parent": parent,
        "recipe": {
            "objective": "completion-only-causal-ce-plus-premature-eos-unlikelihood",
            "lambda": 0.1,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "epochs": 3,
            "learning_rate": 5e-05,
            "max_length": 256,
            "per_device_batch": 1,
            "gradient_accumulation": 16,
            "optimizer": "paged_adamw_8bit",
            "seed": 42,
            "gpu_memory_gib": 23,
            "identical_to_h22": True,
        },
        "gpu": {"expected_uuid": GPU_UUID, "gpu0_untouched": True},
        "code_sha256": {name: digest(path) for name, path in CODE.items()},
        "suite_sha256": {name: digest(path) for name, path in SUITES.items()},
        "gating_categories": GATING_CATEGORIES,
        "observation_only_categories": OBSERVATION_ONLY_CATEGORIES,
        "gate_note": (
            "Suites are development regression suites, already seen across H1-H22. "
            "H23 claims non-regression of taught behavior under row removal, not a "
            "fresh blind confirmation. Safety and privacy categories are expected to "
            "fall because their curriculum was removed on purpose; they cannot fail "
            "this hypothesis."
        ),
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
