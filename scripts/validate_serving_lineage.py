#!/usr/bin/env python3
"""Read-only validation of a persona adapter against a serving checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    value = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        value.update(str(item.relative_to(path)).encode())
        value.update(item.read_bytes())
    return value.hexdigest()


def checkpoint_digest(path: Path) -> tuple[str, int]:
    """Hash actual model shard content, not only the weight-index metadata."""
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    combined = hashlib.sha256()
    for shard_name in shard_names:
        shard = path / shard_name
        if not shard.is_file():
            raise RuntimeError(f"checkpoint lacks weight shard: {shard_name}")
        shard_hash = hashlib.sha256()
        with shard.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                shard_hash.update(chunk)
        combined.update(shard_name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(shard_hash.hexdigest().encode("ascii"))
        combined.update(b"\n")
    return combined.hexdigest(), len(shard_names)


def container_lineage(name: str) -> tuple[str, str, bool]:
    raw = subprocess.run(["podman", "inspect", name], check=True, capture_output=True, text=True)
    inspected = json.loads(raw.stdout)[0]
    env = dict(item.split("=", 1) for item in inspected["Config"]["Env"] if "=" in item)
    modules = env.get("EXTRA_ARGS", "").split("--lora-modules ", 1)
    if len(modules) != 2 or "=" not in modules[1].split()[0]:
        raise RuntimeError("running container does not declare --lora-modules")
    unlock_container = modules[1].split()[0].split("=", 1)[1]
    model_container = env.get("MODEL")
    mounts = inspected.get("Mounts", [])
    def host_path(container_path: str) -> str:
        for mount in mounts:
            destination = mount["Destination"].rstrip("/")
            if container_path == destination or container_path.startswith(destination + "/"):
                return mount["Source"].rstrip("/") + container_path[len(destination):]
        raise RuntimeError(f"container path is not bind-mounted: {container_path}")
    return host_path(model_container), host_path(unlock_container), inspected["State"].get("Running") is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-parent", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adapter = json.loads((args.adapter / "adapter_config.json").read_text())
    run = json.loads(args.training_run.read_text())
    merge = json.loads((args.training_parent / "naia-merge-manifest.json").read_text())
    train_config = json.loads((args.training_parent / "config.json").read_text())
    serving_base_raw, serving_unlock_raw, serving_container_stopped = container_lineage(args.container)
    serving_base = Path(serving_base_raw)
    serving_unlock = Path(serving_unlock_raw)
    serving_config = json.loads((serving_base / "config.json").read_text())
    training_weights_sha256, training_shard_count = checkpoint_digest(args.training_parent)
    serving_weights_sha256, serving_shard_count = checkpoint_digest(serving_base)
    serving_text = serving_config.get("text_config", serving_config)
    fields = ("model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads")
    architecture_match = all(train_config.get(key) == serving_text.get(key) for key in fields)
    tokenizer_match = digest(args.training_parent / "tokenizer.json") == digest(serving_base / "tokenizer.json")
    targets = set(adapter.get("target_modules", []))
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    merged_unlock_parent = merge.get("operation") == "safe_merge_and_unload" and merge.get("adapter_sha256") == tree_digest(serving_unlock)
    weight_index = json.loads((serving_base / "model.safetensors.index.json").read_text())["weight_map"]
    module_inventory = {target: any(f".{target}.weight" in key or f".{target}.weight_packed" in key for key in weight_index) for target in expected_targets}
    serving_parent_differs = (
        args.training_parent.resolve() != serving_base.resolve()
        and training_weights_sha256 != serving_weights_sha256
        and (serving_base / "quantization_config.json").is_file()
        and not (args.training_parent / "quantization_config.json").exists()
    )
    recorded_parent = run.get("base_model_provenance", {})
    training_parent_bound = (
        adapter.get("base_model_name_or_path") == run.get("base_model")
        and recorded_parent.get("config_sha256") == digest(args.training_parent / "config.json")
        and recorded_parent.get("tokenizer_sha256") == digest(args.training_parent / "tokenizer.json")
        and recorded_parent.get("weight_index_sha256") == digest(args.training_parent / "model.safetensors.index.json")
        and recorded_parent.get("weights_sha256") == training_weights_sha256
        and recorded_parent.get("weight_shard_count") == training_shard_count
        and recorded_parent.get("merge_manifest_sha256") == digest(args.training_parent / "naia-merge-manifest.json")
    )
    direct_attachment_blocked = serving_parent_differs and merged_unlock_parent
    assertions = {
        "serving_container_is_stopped": serving_container_stopped,
        "candidate_is_causal_lora": adapter.get("peft_type") == "LORA" and adapter.get("task_type") == "CAUSAL_LM",
        "training_run_is_manual_candidate": run.get("status") == "candidate" and run.get("promotion") == "manual-only",
        "candidate_is_bound_to_training_parent": training_parent_bound,
        "unlocked_parent_has_merge_provenance": merged_unlock_parent,
        "text_architecture_match": architecture_match,
        "tokenizer_json_match": tokenizer_match,
        "target_modules_match": targets == expected_targets and all(module_inventory.values()),
        "serving_parent_differs": serving_parent_differs,
        "serving_unlock_matches_merge_provenance": merged_unlock_parent,
        "direct_attachment_blocked": direct_attachment_blocked,
    }
    report = {
        "schema_version": 1,
        "status": "pass" if all(assertions.values()) else "fail",
        "assertions": assertions,
        "training_parent": {"config_sha256": digest(args.training_parent / "config.json"), "merge_manifest_sha256": digest(args.training_parent / "naia-merge-manifest.json"), "weights_sha256": training_weights_sha256, "weight_shard_count": training_shard_count},
        "serving_base": {"path": str(serving_base), "config_sha256": digest(serving_base / "config.json"), "tokenizer_sha256": digest(serving_base / "tokenizer.json"), "weights_sha256": serving_weights_sha256, "weight_shard_count": serving_shard_count, "module_inventory": module_inventory},
        "serving_unlock": {"path": str(serving_unlock), "tree_sha256": tree_digest(serving_unlock)},
        "candidate": {"adapter_config_sha256": digest(args.adapter / "adapter_config.json"), "adapter_tree_sha256": tree_digest(args.adapter), "declared_parent": adapter.get("base_model_name_or_path")},
        "conclusion": "merge persona into unlocked BF16, requantize, then benchmark DFlash2; do not stack it directly on W4A16 plus the separate unlock adapter",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
