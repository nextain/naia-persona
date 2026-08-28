#!/usr/bin/env python3
"""24 GB-oriented QLoRA entrypoint. Produces a candidate adapter only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def to_prompt_completion(example: dict) -> dict:
    """Train only the final assistant turn while retaining prior context."""
    messages = example["messages"]
    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
        # Keep training and serving behavior aligned.  Qwen3.8 otherwise adds
        # a thinking preamble whose template details can change the boundary
        # used by TRL's completion-only mask.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_completion_boundaries(dataset, tokenizer, max_length: int) -> dict:
    """Fail before model loading if any completion-only mask would be invalid."""
    failures = []
    prompt_lengths = []
    completion_lengths = []
    total_lengths = []

    for index, example in enumerate(dataset):
        kwargs = example.get("chat_template_kwargs", {})
        prompt_ids = tokenizer.apply_chat_template(
            example["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False, **kwargs
        )
        full_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"], tokenize=True, return_dict=False, **kwargs
        )
        completion_length = len(full_ids) - len(prompt_ids)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            failures.append({"index": index, "reason": "prompt-prefix-mismatch"})
        elif completion_length <= 0:
            failures.append({"index": index, "reason": "empty-completion"})
        elif len(prompt_ids) >= max_length:
            failures.append({"index": index, "reason": "completion-truncated", "prompt_tokens": len(prompt_ids)})
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(completion_length)
        total_lengths.append(len(full_ids))

    if failures:
        raise RuntimeError(f"completion-only preflight failed: {json.dumps(failures[:10], ensure_ascii=False)}")
    return {
        "examples": len(total_lengths),
        "max_prompt_tokens": max(prompt_lengths, default=0),
        "min_completion_tokens": min(completion_lengths, default=0),
        "max_total_tokens": max(total_lengths, default=0),
        "truncated_examples": sum(length > max_length for length in total_lengths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL", "Qwen/Qwen3.8-27B"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data and completion-only token boundaries without loading the model",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        default=19,
        help="GPU allocation ceiling; remaining frozen blocks are CPU-offloaded",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = Path(__file__).with_name("validate_dataset.py")
    subprocess.run([sys.executable, str(validator), str(args.data)], check=True)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    dataset = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)
    mask_preflight = validate_completion_boundaries(dataset, tokenizer, args.max_length)
    print(json.dumps({"completion_only_preflight": mask_preflight}, ensure_ascii=False))
    if args.preflight_only:
        return 0

    args.output.mkdir(parents=True, exist_ok=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{args.gpu_memory_gib}GiB", "cpu": "96GiB"},
        offload_folder=str(args.output / "offload"),
        trust_remote_code=True,
    )
    # PEFT's generic helper casts every non-Params4bit tensor to FP32.  Qwen3.8
    # keeps a large shared embedding/output tensor outside bitsandbytes, so that
    # cast alone can require another ~4.7 GiB and defeat the 24 GB target.  The
    # base is frozen and remains in its loaded BF16/NF4 dtype; only LoRA weights
    # created below are trainable.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    persona_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_config = persona_config

    # Qwen3.8's stock chat template has no Jinja generation markers, so TRL's
    # assistant_only_loss cannot derive an assistant mask.  Prompt/completion
    # records provide the same safety property without replacing Qwen's template:
    # user/system tokens are context and only the final assistant turn is trained.
    config = SFTConfig(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_length=args.max_length,
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(args.output / "adapter"))
    dataset_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    manifest = {
        "status": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "dataset": str(args.data),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "rank": args.rank,
        "gpu_memory_gib": args.gpu_memory_gib,
        "seed": args.seed,
        "dataset_sha256": dataset_sha256,
        "completion_only_preflight": mask_preflight,
        "adapter_layout": "persona-on-declared-base",
        "promotion": "manual-only",
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
