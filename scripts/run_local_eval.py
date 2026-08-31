#!/usr/bin/env python3
"""Evaluate a local Transformers checkpoint with an optional PEFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_eval_suite import score

H9_GPU_UUID = "GPU-d584beef-b086-bdff-b43c-c31a1b56a611"
H9_SUITE_SHA256 = {
    "767a8bc3291423c70485aae97659b3967551072ffb1e29e6aa4c7504e1c1db82",
    "83c6c498572563275af4a7a883141bd9e6b017e8db5e12a80e4e27dfda1b01aa",
}
H11_SUITE_SHA256 = {
    "77686ba27a66f3fa99db3647e8ca713d9bfaddf421de105944672f467b749302",
    "f8aeac4fe9597a2145630a3670d413de32948b01e8f2ca8a1471367778e1a240",
    "e86bc198b61c3fe209a7bc277f960440632e3bb9f93b2bec769a4e8bbb7d8843",
}
H12_CHALLENGE_SHA256 = "5c5b63a243dcc896c7d4b98a774b2cf9f21ae45acaecadaef3d71ff6af64b441"
H12_SUITE_SHA256 = H11_SUITE_SHA256 | {H12_CHALLENGE_SHA256}
H13_SUITE_SHA256 = H12_SUITE_SHA256
H14_SUITE_SHA256 = H13_SUITE_SHA256
H15_SUITE_SHA256 = H14_SUITE_SHA256
H16_SUITE_SHA256 = H15_SUITE_SHA256
H17_SUITE_SHA256 = H16_SUITE_SHA256
H18_SUITE_SHA256 = H17_SUITE_SHA256
H19_SUITE_SHA256 = H18_SUITE_SHA256
H21_SUITE_SHA256 = H19_SUITE_SHA256
H22_BLIND_SHA256 = "c2cef183f0e344c9676f74442fb8462c3d5cce794a4ab25294c07edad8fa4f8f"
H22_SUITE_SHA256 = H21_SUITE_SHA256 | {H22_BLIND_SHA256}
# H23 changes training rows only. The suites, scorer, and decoding are byte-identical to H22.
H23_SUITE_SHA256 = H22_SUITE_SHA256
# H24 changes epochs only. Same suites, scorer, and decoding as H23.
H24_SUITE_SHA256 = H23_SUITE_SHA256
# H25 adds the held-out identity confirmation suite frozen before its curriculum.
H25_HELDOUT_SHA256 = "1d0f28f4af9b961a6adf86b281c8e9dc3fb1cde65ca1e006451b1f4b14e8c7e5"
H25_SUITE_SHA256 = H24_SUITE_SHA256 | {H25_HELDOUT_SHA256}


def tree_digest(path: Path) -> str:
    value = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        value.update(str(item.relative_to(path)).encode())
        value.update(item.read_bytes())
    return value.hexdigest()


def checkpoint_digest(path: Path) -> tuple[str, int]:
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    combined = hashlib.sha256()
    for shard_name in shard_names:
        shard_hash = hashlib.sha256()
        with (path / shard_name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                shard_hash.update(chunk)
        combined.update(shard_name.encode())
        combined.update(b"\0")
        combined.update(shard_hash.hexdigest().encode())
        combined.update(b"\n")
    return combined.hexdigest(), len(shard_names)


def visible_gpu_uuids() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["prompts"] if isinstance(payload, dict) else payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", default="정확하고 유용하게 답하세요.")
    parser.add_argument("--gpu-memory-gib", type=int, default=19)
    parser.add_argument("--expected-suite-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=["h9", "h11", "h12", "h13", "h14", "h15", "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23", "h24", "h25"], required=True)
    parser.add_argument(
        "--additional-suite",
        nargs=3,
        action="append",
        metavar=("SUITE", "OUTPUT", "EXPECTED_SHA256"),
        default=[],
        help="Evaluate another frozen suite after the primary suite without reloading the model",
    )
    args = parser.parse_args()

    if args.profile == "h9" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H9_SUITE_SHA256
    ):
        raise RuntimeError("H9 profile requires its preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h11" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H11_SUITE_SHA256
    ):
        raise RuntimeError("H11 profile requires its preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h12" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H12_SUITE_SHA256
    ):
        raise RuntimeError("H12 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h13" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H13_SUITE_SHA256
    ):
        raise RuntimeError("H13 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h14" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H14_SUITE_SHA256
    ):
        raise RuntimeError("H14 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h15" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H15_SUITE_SHA256
    ):
        raise RuntimeError("H15 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h16" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H16_SUITE_SHA256
    ):
        raise RuntimeError("H16 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h17" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H17_SUITE_SHA256
    ):
        raise RuntimeError("H17 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h18" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H18_SUITE_SHA256
    ):
        raise RuntimeError("H18 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h19" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H19_SUITE_SHA256
    ):
        raise RuntimeError("H19 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h20" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H19_SUITE_SHA256
    ):
        raise RuntimeError("H20 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h21" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H21_SUITE_SHA256
    ):
        raise RuntimeError("H21 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h22" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H22_SUITE_SHA256
    ):
        raise RuntimeError("H22 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h23" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H23_SUITE_SHA256
    ):
        raise RuntimeError("H23 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h24" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H24_SUITE_SHA256
    ):
        raise RuntimeError("H24 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")
    if args.profile == "h25" and (
        args.expected_gpu_uuid != H9_GPU_UUID or args.expected_suite_sha256 not in H25_SUITE_SHA256
    ):
        raise RuntimeError("H25 profile requires a preregistered suite SHA-256 and physical GPU1 UUID")

    suite_sha256 = hashlib.sha256(args.suite.read_bytes()).hexdigest()
    if suite_sha256 != args.expected_suite_sha256:
        raise RuntimeError(
            f"suite SHA-256 mismatch: expected {args.expected_suite_sha256}, got {suite_sha256}"
        )
    additional_suites: list[tuple[Path, Path, str]] = []
    for suite_value, output_value, expected_hash in args.additional_suite:
        suite_path = Path(suite_value)
        output_path = Path(output_value)
        allowed_hashes = H11_SUITE_SHA256 if args.profile == "h11" else (
            H22_SUITE_SHA256 if args.profile == "h22" else (
                H23_SUITE_SHA256 if args.profile == "h23" else (
                    H24_SUITE_SHA256 if args.profile == "h24" else (
                        H25_SUITE_SHA256 if args.profile == "h25" else H19_SUITE_SHA256
                    )
                )
            )
        )
        if args.profile not in {"h11", "h12", "h13", "h14", "h15", "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23", "h24", "h25"} or expected_hash not in allowed_hashes:
            raise RuntimeError("additional suites are restricted to preregistered profile hashes")
        actual_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"additional suite SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        additional_suites.append((suite_path, output_path, expected_hash))
    gpu_uuids = visible_gpu_uuids()
    if gpu_uuids != [args.expected_gpu_uuid]:
        raise RuntimeError(
            f"GPU isolation failure: expected only {[args.expected_gpu_uuid]}, got {gpu_uuids}"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base_weights_sha256, base_weight_shard_count = checkpoint_digest(Path(args.base_model))

    if args.adapter:
        adapter_path = Path(args.adapter)
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.is_file():
            nested = adapter_path / "adapter" / "adapter_config.json"
            hint = f"; did you mean {adapter_path / 'adapter'}?" if nested.is_file() else ""
            raise FileNotFoundError(f"LoRA adapter_config.json not found in {adapter_path}{hint}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{args.gpu_memory_gib}GiB", "cpu": "96GiB"},
        offload_folder=str(args.output.parent / ".eval-offload"),
        trust_remote_code=True,
    )

    primary_cases = load_cases(args.suite)
    def evaluate(
        active_model,
        adapter_name: str | None,
        output_path: Path,
        cases: list[dict],
        active_suite_sha256: str,
    ) -> None:
        active_model.eval()
        records: list[dict] = []
        by_category: dict[str, list[float]] = {}
        failures: list[str] = []
        for case in cases:
            category = case.get("category") or case.get("axis")
            if not category:
                raise ValueError(f"evaluation case {case.get('id', '<unknown>')} has no category or axis")
            messages = [{"role": "system", "content": args.system}, {"role": "user", "content": case["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
            started = time.perf_counter()
            with torch.inference_mode():
                output = active_model.generate(**inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            elapsed = time.perf_counter() - started
            answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            semantic_contract = case.get("expected", {}).get("required_behaviors")
            if semantic_contract is None:
                value, case_failures = score(case, answer)
                by_category.setdefault(category, []).append(value)
                failures.extend(case_failures)
            else:
                value, case_failures = None, []
            records.append({"id": case["id"], "category": category, "score": value,
                "semantic_review_required": semantic_contract is not None,
                "expected": case.get("expected") if semantic_contract is not None else None,
                "elapsed_s": elapsed, "answer": answer})
        category_scores = {key: sum(values) / len(values) for key, values in by_category.items()}
        persona_values = by_category.get("persona", []) + by_category.get("boundary", [])
        report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "model": args.base_model, "adapter": adapter_name,
            "adapter_config_sha256": (
                hashlib.sha256((Path(adapter_name) / "adapter_config.json").read_bytes()).hexdigest()
                if adapter_name else None
            ),
            "adapter_tree_sha256": tree_digest(Path(adapter_name)) if adapter_name else None,
            "base_weights_sha256": base_weights_sha256,
            "base_weight_shard_count": base_weight_shard_count,
            "system": args.system,
            "suite_sha256": active_suite_sha256, "seed": args.seed,
            "semantic_review_required": any(row["semantic_review_required"] for row in records),
            "visible_gpu_uuids": gpu_uuids,
            "generation": {"enable_thinking": False, "do_sample": False, "max_new_tokens": 256},
            "general_score": category_scores.get("general", 0.0),
            "persona_score": sum(persona_values) / len(persona_values) if persona_values else 0.0,
            "safety_score": category_scores.get("safety", 0.0), "category_scores": category_scores,
            "regression_failures": failures,
            "mean_elapsed_s": sum(row["elapsed_s"] for row in records) / len(records), "records": records}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("adapter", "general_score", "persona_score",
            "safety_score", "mean_elapsed_s", "regression_failures")}, ensure_ascii=False))

    if args.baseline_output:
        evaluate(model, None, args.baseline_output, primary_cases, suite_sha256)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    evaluate(model, args.adapter, args.output, primary_cases, suite_sha256)
    for additional_path, additional_output, additional_hash in additional_suites:
        additional_cases = load_cases(additional_path)
        evaluate(model, args.adapter, additional_output, additional_cases, additional_hash)


if __name__ == "__main__":
    main()
