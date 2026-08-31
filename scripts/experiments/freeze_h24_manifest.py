#!/usr/bin/env python3
"""Freeze H24 provenance before any GPU is allocated.

H24 exists to remove one confound from H23. H23 halved the training rows while
holding epochs at 3, which also halved optimizer steps, so its persona and
boundary drop could have come either from the removed curriculum or from half the
optimization. H24 trains the identical rows for 6 epochs, restoring H22's step
count, so whatever difference remains belongs to the data and not to the volume.

This manifest therefore asserts the opposite of the H23 one: the dataset must be
byte-identical to H23's, and the epoch count must be the only thing that changed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "data-private/datasets/naia-v12/freeze-manifest-h24.json"

TRAIN = REPO / "data-private/datasets/naia-v12/train.jsonl"
H23_FREEZE = REPO / "data-private/datasets/naia-v12/freeze-manifest.json"
H23_RUN = REPO / "data-private/runs/train/naia-v12-qwen38-27b-gpu1-r16-e3-lr5e5-h23/run.json"

TRAIN_SHA256 = "4be4cd1208291049c51dbad57a199027a1933799aa8c6ff09e8cb52dee901a39"

PARENT = Path("<models>/Qwen3.8-27B-Unlocked-BF16")
PARENT_CONFIG_SHA256 = "00e63206a383837e0eda70dbd8aef807e5a18fa5d52ed1671c96076abcb24c38"
PARENT_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
PARENT_INDEX_SHA256 = "6b70aea64bb78f3627a3a1885e8847b5ad2fa0184237e5086c3d216a71a1f04c"
GPU_UUID = "GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

CODE = {
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

GATING_CATEGORIES = ["general", "persona", "boundary"]
OBSERVATION_ONLY_CATEGORIES = ["safety", "privacy"]

ROWS = 239
ACCUMULATION = 16
H22_STEPS = 90  # ceil(474 / 16) * 3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    actual = digest(TRAIN)
    if actual != TRAIN_SHA256:
        raise RuntimeError(f"naia-v12 digest drifted: {actual}")

    lines = [line for line in TRAIN.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != ROWS:
        raise RuntimeError(f"H24 expects {ROWS} rows, found {len(lines)}")

    h23_freeze = json.loads(H23_FREEZE.read_text(encoding="utf-8"))
    if h23_freeze["data"]["h23_train_sha256"] != TRAIN_SHA256:
        raise RuntimeError("H24 must train the exact rows H23 trained")

    h23_run = json.loads(H23_RUN.read_text(encoding="utf-8"))
    if h23_run["epochs"] != 3.0 or h23_run["dataset_sha256"] != TRAIN_SHA256:
        raise RuntimeError("H23 run record does not match the comparison baseline")

    steps = -(-len(lines) // ACCUMULATION) * 6
    if abs(steps - H22_STEPS) > 1:
        raise RuntimeError(f"6 epochs gives {steps} steps, which does not match H22's {H22_STEPS}")

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
        "experiment": "H24",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "single_change_against_h23": "epochs 3 -> 6",
        "purpose": (
            "Separate the removed curriculum from reduced training volume as the cause "
            "of the H23 persona and boundary drop by matching H22's optimizer step count."
        ),
        "data": {
            "train_sha256": actual,
            "rows": len(lines),
            "identical_to_h23": True,
            "policy_rows_restored": 0,
        },
        "optimizer_steps": {
            "h22": H22_STEPS,
            "h23": -(-len(lines) // ACCUMULATION) * 3,
            "h24": steps,
            "matched_to_h22": True,
        },
        "parent": parent,
        "recipe": {
            "objective": "completion-only-causal-ce-plus-premature-eos-unlikelihood",
            "lambda": 0.1,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "epochs": 6,
            "learning_rate": 5e-05,
            "max_length": 256,
            "per_device_batch": 1,
            "gradient_accumulation": ACCUMULATION,
            "optimizer": "paged_adamw_8bit",
            "seed": 42,
            "gpu_memory_gib": 23,
            "identical_to_h23_except_epochs": True,
        },
        "gpu": {"expected_uuid": GPU_UUID, "gpu0_untouched": True},
        "code_sha256": {name: digest(path) for name, path in CODE.items()},
        "suite_sha256": {name: digest(path) for name, path in SUITES.items()},
        "gating_categories": GATING_CATEGORIES,
        "observation_only_categories": OBSERVATION_ONLY_CATEGORIES,
        "h23_baseline": {
            "fixed_general": 100.0,
            "fixed_persona": 33.333333333333336,
            "fixed_boundary": 0.0,
            "mean_causal_ce": h23_run["loss_objective"]["mean_component_losses"]["causal_ce"],
        },
        "gate_note": (
            "Safety and privacy suites remain recorded and non-deciding. Restoring a "
            "removed policy row to recover a score is forbidden by the charter and by "
            "the H24 decision map."
        ),
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
