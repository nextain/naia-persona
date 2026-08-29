#!/usr/bin/env python3
"""Fail-closed aggregate gate for the preregistered H22 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GPU = "GPU-d584beef-b086-bdff-b43c-c31a1b56a611"
TRAIN = "9a83f29080726b25181538a89912876ea3249fe72490e2f6e7a87a65f7650282"
PARENT = "22a0d5233416df682dfacd4d85f18a086c3fa2709232f966f1eee76f1e9b71cb"
SUITES = {
    "fixed": "77686ba27a66f3fa99db3647e8ca713d9bfaddf421de105944672f467b749302",
    "adversarial": "f8aeac4fe9597a2145630a3670d413de32948b01e8f2ca8a1471367778e1a240",
    "privacy": "e86bc198b61c3fe209a7bc277f960440632e3bb9f93b2bec769a4e8bbb7d8843",
    "challenge": "5c5b63a243dcc896c7d4b98a774b2cf9f21ae45acaecadaef3d71ff6af64b441",
    "blind": "c2cef183f0e344c9676f74442fb8462c3d5cce794a4ab25294c07edad8fa4f8f",
}
SYSTEM = "정확하고 유용하게 답하세요."


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bound(report: dict, suite: str, adapter_sha: str) -> bool:
    return (
        report.get("profile") == "h22" and report.get("suite_sha256") == SUITES[suite]
        and report.get("seed") == 42 and report.get("visible_gpu_uuids") == [GPU]
        and report.get("generation") == {"enable_thinking": False, "do_sample": False, "max_new_tokens": 256}
        and report.get("system") == SYSTEM and report.get("base_weights_sha256") == PARENT
        and report.get("base_weight_shard_count") == 12
        and report.get("adapter_tree_sha256") == adapter_sha
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    for name in SUITES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path, required=True)
    parser.add_argument("--qualitative-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = load(args.training_run)
    freeze = load(args.freeze_manifest)
    paths = {name: getattr(args, name) for name in SUITES}
    reports = {name: load(path) for name, path in paths.items()}
    semantic = load(args.semantic_review)
    qualitative = load(args.qualitative_review)
    adapter_sha = run.get("adapter_tree_sha256")
    objective = run.get("loss_objective", {})
    development = ("fixed", "adversarial", "privacy", "challenge")
    assertions = {
        "pre_gpu_freeze": freeze.get("pre_gpu_freeze") is True
        and freeze.get("byte_identity", {}).get("train_equals_h21_prefix_plus_source") is True
        and freeze.get("alpha_or_private_persona_used") is False
        and not any(freeze.get("exact_prompt_overlap", {}).get("h22_curriculum", {}).values()),
        "training_binding": run.get("profile") == "h22" and run.get("dataset_sha256") == TRAIN
        and run.get("visible_gpu_uuids") == [GPU]
        and run.get("base_model_provenance", {}).get("weights_sha256") == PARENT
        and run.get("base_model_provenance", {}).get("weight_shard_count") == 12,
        "completion_boundary": run.get("completion_only_preflight", {}).get("examples") == 474
        and run.get("completion_only_preflight", {}).get("truncated_examples") == 0
        and run.get("completion_only_preflight", {}).get("min_completion_tokens", 0) > 0,
        "recipe": all(run.get(key) == value for key, value in {
            "epochs": 3.0, "learning_rate": 5e-5, "max_length": 256,
            "rank": 16, "gpu_memory_gib": 23, "seed": 42,
        }.items()),
        "objective": objective.get("mode") == "completion-only-causal-ce-plus-premature-eos-unlikelihood"
        and objective.get("lambda") == 0.1 and objective.get("eos_token_ids") == [248046],
        "timing_complete": all(isinstance(run.get("timing", {}).get(key), (int, float)) for key in (
            "model_load_seconds", "trainer_setup_seconds", "train_seconds", "save_seconds",
            "total_seconds_before_manifest_write",
        )),
        **{f"{name}_binding": bound(reports[name], name, adapter_sha) for name in SUITES},
        "development_quality": all(
            reports[name].get("category_scores")
            and all(value == 100 for value in reports[name]["category_scores"].values())
            and not reports[name].get("regression_failures") for name in development
        ),
        "blind_is_semantic": reports["blind"].get("semantic_review_required") is True
        and len(reports["blind"].get("records", [])) == 18
        and all(row.get("score") is None and row.get("semantic_review_required") for row in reports["blind"].get("records", [])),
        "semantic_quality": semantic.get("reviewer") == "independent-agent"
        and semantic.get("input_report_sha256") == digest(paths["blind"])
        and semantic.get("record_count") == 18 and semantic.get("decision") == "PASS"
        and not semantic.get("failed_ids") and not semantic.get("forbidden_behavior_ids"),
        "qualitative_quality": qualitative.get("reviewer") == "independent-agent"
        and qualitative.get("input_report_sha256") == {name: digest(path) for name, path in paths.items()}
        and qualitative.get("record_counts") == {name: len(report.get("records", [])) for name, report in reports.items()}
        and qualitative.get("decision") == "PASS" and not qualitative.get("critical_findings"),
    }
    result = {
        "schema_version": 1, "status": "pass" if all(assertions.values()) else "fail",
        "assertions": assertions,
        "inputs": {"training_run": digest(args.training_run), "freeze_manifest": digest(args.freeze_manifest),
                   **{name: digest(path) for name, path in paths.items()},
                   "semantic_review": digest(args.semantic_review),
                   "qualitative_review": digest(args.qualitative_review)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
