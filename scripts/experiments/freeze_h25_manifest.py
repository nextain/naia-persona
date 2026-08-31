#!/usr/bin/env python3
"""Freeze H25 provenance before any GPU is allocated.

The claim H25 wants to make is about held-out identity behavior, so the thing
that most needs freezing is not the recipe but the ordering: the confirmation
suite existed, at this exact hash, before the curriculum that it judges. This
manifest records that, re-runs the overlap check, and pins the step count to 90
so the comparison against H24 is about the added rows and nothing else.
"""

from __future__ import annotations

import os
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "data-private/datasets/naia-v13/freeze-manifest.json"

TRAIN = REPO / "data-private/datasets/naia-v13/train.jsonl"
DATASET_MANIFEST = REPO / "data-private/datasets/naia-v13/manifest.json"
BASE = REPO / "data-private/datasets/naia-v12/train.jsonl"
SOURCE = REPO / "examples/naia-v13/source.jsonl"
HELDOUT = REPO / "examples/naia-v13/identity-confirmation-v3-prompts.json"
H24_RUN = REPO / "data-private/runs/train/naia-v12-qwen38-27b-gpu1-r16-e6-lr5e5-h24/run.json"

TRAIN_SHA256 = "570bbc1088a61f76032feefc1ec14d56850c5baf26b9906a1914cbc1619b5fe6"
BASE_SHA256 = "4be4cd1208291049c51dbad57a199027a1933799aa8c6ff09e8cb52dee901a39"
HELDOUT_SHA256 = "1d0f28f4af9b961a6adf86b281c8e9dc3fb1cde65ca1e006451b1f4b14e8c7e5"

PARENT = Path(os.environ.get("NAIA_PARENT_MODEL", Path.home() / "models" / "Qwen3.8-27B-Unlocked-BF16"))
PARENT_CONFIG_SHA256 = "00e63206a383837e0eda70dbd8aef807e5a18fa5d52ed1671c96076abcb24c38"
PARENT_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
PARENT_INDEX_SHA256 = "6b70aea64bb78f3627a3a1885e8847b5ad2fa0184237e5086c3d216a71a1f04c"
GPU_UUID = "GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

CODE = {
    "build_h25_dataset.py": REPO / "scripts/build_h25_dataset.py",
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
    "heldout_v3": HELDOUT,
}

GATING_CATEGORIES = ["general", "persona", "boundary"]
OBSERVATION_ONLY_CATEGORIES = ["safety", "privacy"]
CONTAMINATED_BY_DESIGN = ["adv-identity-ko", "adv-identity-en", "adv-identity-role"]

ROWS = 287
ACCUMULATION = 16
EPOCHS = 5
TARGET_STEPS = 90
NGRAM = 12

HELDOUT_MIN_PASS = 16  # of 18, preregistered before any v3 answer was generated


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).casefold()


def ngrams(text: str, n: int) -> set[str]:
    flat = normalize(text)
    return {flat[i : i + n] for i in range(max(0, len(flat) - n + 1))}


def main() -> None:
    for path, expected, label in (
        (TRAIN, TRAIN_SHA256, "naia-v13 train"),
        (BASE, BASE_SHA256, "naia-v12 base"),
        (HELDOUT, HELDOUT_SHA256, "held-out v3 suite"),
    ):
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"{label} digest drifted: {actual}")

    base_bytes = BASE.read_bytes()
    if TRAIN.read_bytes()[: len(base_bytes)] != base_bytes:
        raise RuntimeError("the inherited naia-v12 rows are not an exact byte prefix of naia-v13")

    lines = [line for line in TRAIN.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != ROWS:
        raise RuntimeError(f"H25 expects {ROWS} rows, found {len(lines)}")

    steps = -(-len(lines) // ACCUMULATION) * EPOCHS
    if steps != TARGET_STEPS:
        raise RuntimeError(f"{EPOCHS} epochs gives {steps} steps, not the required {TARGET_STEPS}")

    serialized = TRAIN.read_text(encoding="utf-8")
    for marker in ("alpha", "알파", "cafelua", "luke", "루크"):
        if marker.casefold() in serialized.casefold():
            raise RuntimeError(f"private marker {marker!r} found in H25 training data")

    # Re-run the overlap check here, not only in the builder, so the manifest
    # that authorizes the GPU is itself the thing that proves no leakage.
    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))
    curriculum = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    shared = 0
    for row in curriculum:
        mine = ngrams(row["messages"][0]["content"], NGRAM)
        for case in heldout:
            shared += len(mine & ngrams(case["prompt"], NGRAM))
    if shared:
        raise RuntimeError(f"{shared} shared n-grams between curriculum and held-out prompts")

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

    h24 = json.loads(H24_RUN.read_text(encoding="utf-8"))

    manifest = {
        "schema_version": 1,
        "experiment": "H25",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "single_change_against_h24": "48 appended identity-diversity rows; epochs adjusted 6 -> 5 solely to hold optimizer steps at 90",
        "ordering_commitment": {
            "heldout_suite_sha256": HELDOUT_SHA256,
            "frozen_before_curriculum": True,
            "registered_in_ledger_before_curriculum": True,
            "overlap_check": {"ngram": NGRAM, "shared_ngrams": shared},
            "limitation": (
                "Curriculum and held-out suite share one author. The overlap check bounds "
                "surface copying only, not conceptual leakage. This is weaker than the "
                "independent authorship H22 used and must be stated in the report."
            ),
        },
        "contaminated_by_design": {
            "cases": CONTAMINATED_BY_DESIGN,
            "reason": "H25 deliberately adds assertion-form identity rows, so these three adversarial cases are regression telemetry and may not be cited as confirmation.",
        },
        "data": {
            "train_sha256": TRAIN_SHA256,
            "base_sha256": BASE_SHA256,
            "source_sha256": digest(SOURCE),
            "dataset_manifest_sha256": digest(DATASET_MANIFEST),
            "rows": len(lines),
            "inherited": 239,
            "appended": 48,
            "inherited_rows_edited": 0,
            "policy_rows_restored": 0,
        },
        "optimizer_steps": {"h22": 90, "h23": 45, "h24": 90, "h25": steps, "matched": True},
        "parent": parent,
        "recipe": {
            "objective": "completion-only-causal-ce-plus-premature-eos-unlikelihood",
            "lambda": 0.1,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "epochs": EPOCHS,
            "learning_rate": 5e-05,
            "max_length": 256,
            "per_device_batch": 1,
            "gradient_accumulation": ACCUMULATION,
            "optimizer": "paged_adamw_8bit",
            "seed": 42,
            "gpu_memory_gib": 23,
        },
        "gpu": {"expected_uuid": GPU_UUID, "gpu0_untouched": True},
        "code_sha256": {name: digest(path) for name, path in CODE.items()},
        "suite_sha256": {name: digest(path) for name, path in SUITES.items()},
        "gating_categories": GATING_CATEGORIES,
        "observation_only_categories": OBSERVATION_ONLY_CATEGORIES,
        "heldout_min_pass": HELDOUT_MIN_PASS,
        "h24_baseline": {
            "adapter_tree_sha256": h24["adapter_tree_sha256"],
            "mean_causal_ce": h24["loss_objective"]["mean_component_losses"]["causal_ce"],
            "fixed_persona": 83.33333333333333,
            "fixed_boundary": 50.0,
            "adversarial_persona": 66.66666666666667,
            "heldout_v3": "to be measured in the same batch as H25, never before curriculum authoring",
        },
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
