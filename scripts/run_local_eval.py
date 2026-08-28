#!/usr/bin/env python3
"""Evaluate a local Transformers checkpoint with an optional PEFT adapter."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_eval_suite import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", default="정확하고 유용하게 답하세요.")
    parser.add_argument("--gpu-memory-gib", type=int, default=19)
    args = parser.parse_args()

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

    cases = json.loads(args.suite.read_text(encoding="utf-8"))
    def evaluate(active_model, adapter_name: str | None, output_path: Path) -> None:
        active_model.eval()
        records: list[dict] = []
        by_category: dict[str, list[float]] = {}
        failures: list[str] = []
        for case in cases:
            messages = [{"role": "system", "content": args.system}, {"role": "user", "content": case["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
            started = time.perf_counter()
            with torch.inference_mode():
                output = active_model.generate(**inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            elapsed = time.perf_counter() - started
            answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            value, case_failures = score(case, answer)
            by_category.setdefault(case["category"], []).append(value)
            failures.extend(case_failures)
            records.append({"id": case["id"], "category": case["category"], "score": value,
                "elapsed_s": elapsed, "answer": answer})
        category_scores = {key: sum(values) / len(values) for key, values in by_category.items()}
        persona_values = by_category.get("persona", []) + by_category.get("boundary", [])
        report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.base_model, "adapter": adapter_name,
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
        evaluate(model, None, args.baseline_output)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    evaluate(model, args.adapter, args.output)


if __name__ == "__main__":
    main()
