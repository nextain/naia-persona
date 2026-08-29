#!/usr/bin/env python3
"""24 GB-oriented QLoRA entrypoint. Produces a candidate adapter only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

H9_GPU_UUID = "GPU-d584beef-b086-bdff-b43c-c31a1b56a611"
H9_DATA_SHA256 = "442a0c1b883a69808a059520d075012228376a860065a0ca2d392099fe256da4"
H9_PARENT_WEIGHTS_SHA256 = "22a0d5233416df682dfacd4d85f18a086c3fa2709232f966f1eee76f1e9b71cb"
H9_PARENT_SHARD_COUNT = 12
H21_PARENT_CONFIG_SHA256 = "00e63206a383837e0eda70dbd8aef807e5a18fa5d52ed1671c96076abcb24c38"
H21_PARENT_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
H21_PARENT_INDEX_SHA256 = "6b70aea64bb78f3627a3a1885e8847b5ad2fa0184237e5086c3d216a71a1f04c"
H21_EOS_TOKEN_IDS = (248046,)
H11_DATA_SHA256 = "b33c59fe5451c395cc8770314dc88d0916ee6276a1e8d79027386025306a7c4a"
H12_DATA_SHA256 = "19232a51dac8b092e02d029c3302dacd5aaaa1994326175b4b15fa8c7145f065"
H13_DATA_SHA256 = "d089d7067a60f6513d74237910d00d6243ae26e10b284c775e3380209b805aba"
H14_DATA_SHA256 = "8d57929dc182e52cf40547648c4ec0c8675067503384afd78b8531348d545232"
H15_DATA_SHA256 = "6f043bb22ab29af06ea305507ecf2ea0515ba624e9e187a8c065ed720fed213f"
H16_DATA_SHA256 = "c2c57acb4b4191342b5116db1dda058f52f4e2047b68b1f7a1ba766565e66f02"
H17_DATA_SHA256 = "0c5ca272c3fe090c1e977671bddfd3654f73ea411fa842969c8776b8a3d76f16"
H18_DATA_SHA256 = "70c781466c842951551d45cf02dc1f419270eecde4c7ed339e7351447082b560"
H19_DATA_SHA256 = H17_DATA_SHA256
H20_DATA_SHA256 = H17_DATA_SHA256
H21_DATA_SHA256 = H17_DATA_SHA256
H22_DATA_SHA256 = "9a83f29080726b25181538a89912876ea3249fe72490e2f6e7a87a65f7650282"


def tree_digest(path: Path) -> str:
    value = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        value.update(str(item.relative_to(path)).encode())
        value.update(item.read_bytes())
    return value.hexdigest()


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
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--profile", choices=["h9", "h11", "h12", "h13", "h14", "h15", "h16", "h17", "h18", "h19", "h20", "h21", "h22"], required=True)
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


def visible_gpu_uuids() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def local_model_provenance(base_model: str) -> dict:
    """Bind a run to immutable local parent metadata before model allocation."""
    root = Path(base_model)
    if not root.is_dir():
        return {"source": base_model, "kind": "remote-reference"}
    required = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"local base model lacks provenance files: {missing}")
    weight_index = json.loads((root / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(weight_index["weight_map"].values()))
    weights_digest = hashlib.sha256()
    for shard_name in shard_names:
        shard = root / shard_name
        if not shard.is_file():
            raise RuntimeError(f"local base model lacks weight shard: {shard_name}")
        shard_digest = hashlib.sha256()
        with shard.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                shard_digest.update(chunk)
        weights_digest.update(shard_name.encode("utf-8"))
        weights_digest.update(b"\0")
        weights_digest.update(shard_digest.hexdigest().encode("ascii"))
        weights_digest.update(b"\n")
    result = {
        "source": str(root.resolve()),
        "kind": "local-checkpoint",
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest(),
        "weight_index_sha256": hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "weights_sha256": weights_digest.hexdigest(),
        "weight_shard_count": len(shard_names),
    }
    merge_manifest = root / "naia-merge-manifest.json"
    if merge_manifest.is_file():
        result["merge_manifest_sha256"] = hashlib.sha256(merge_manifest.read_bytes()).hexdigest()
    return result


def main() -> int:
    total_started = time.monotonic()
    args = parse_args()
    if args.profile == "h9" and (
        args.expected_data_sha256 != H9_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H9 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h11" and (
        args.expected_data_sha256 != H11_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H11 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h12" and (
        args.expected_data_sha256 != H12_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H12 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h13" and (
        args.expected_data_sha256 != H13_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H13 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h14" and (
        args.expected_data_sha256 != H14_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H14 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h15" and (
        args.expected_data_sha256 != H15_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H15 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h16" and (
        args.expected_data_sha256 != H16_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H16 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h17" and (
        args.expected_data_sha256 != H17_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H17 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h18" and (
        args.expected_data_sha256 != H18_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H18 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h19" and (
        args.expected_data_sha256 != H19_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H19 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h20" and (
        args.expected_data_sha256 != H20_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H20 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h21" and (
        args.expected_data_sha256 != H21_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H21 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    if args.profile == "h22" and (
        args.expected_data_sha256 != H22_DATA_SHA256 or args.expected_gpu_uuid != H9_GPU_UUID
    ):
        raise RuntimeError("H22 profile requires its preregistered dataset SHA-256 and physical GPU1 UUID")
    dataset_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if dataset_sha256 != args.expected_data_sha256:
        raise RuntimeError(
            f"dataset SHA-256 mismatch: expected {args.expected_data_sha256}, got {dataset_sha256}"
        )
    base_model_provenance = local_model_provenance(args.base_model)
    if args.profile in {"h9", "h11", "h12", "h13", "h14", "h15", "h16", "h17", "h18", "h19", "h20", "h21", "h22"} and (
        base_model_provenance.get("weights_sha256") != H9_PARENT_WEIGHTS_SHA256
        or base_model_provenance.get("weight_shard_count") != H9_PARENT_SHARD_COUNT
    ):
        raise RuntimeError(f"{args.profile.upper()} profile requires the preregistered unlocked-BF16 parent checkpoint")
    if args.profile in {"h21", "h22"} and (
        base_model_provenance.get("config_sha256") != H21_PARENT_CONFIG_SHA256
        or base_model_provenance.get("tokenizer_sha256") != H21_PARENT_TOKENIZER_SHA256
        or base_model_provenance.get("weight_index_sha256") != H21_PARENT_INDEX_SHA256
    ):
        raise RuntimeError("H21 profile requires the preregistered parent config, tokenizer, and weight index")
    validator = Path(__file__).with_name("validate_dataset.py")
    subprocess.run([sys.executable, str(validator), str(args.data)], check=True)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from torch.nn import functional as F
    from trl import SFTConfig, SFTTrainer

    class TailWeightedSFTTrainer(SFTTrainer):
        """Completion-only causal CE with exactly 2x weight on the latter half."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            token_loss = F.cross_entropy(
                # Match Transformers' ForCausalLMLoss: the 248k-token
                # vocabulary requires FP32 logits for numerically stable CE.
                shift_logits.float().view(-1, shift_logits.size(-1)),
                safe_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)
            positions = valid.long().cumsum(dim=1)
            head_tokens = torch.div(valid.sum(dim=1, keepdim=True), 2, rounding_mode="floor")
            weights = torch.where(valid & (positions > head_tokens), 2.0, 1.0) * valid
            loss = (token_loss * weights).sum() / weights.sum().clamp_min(1)
            return (loss, outputs) if return_outputs else loss

    class FocalSFTTrainer(SFTTrainer):
        """Completion-only focal causal CE with preregistered gamma=2."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            target_probs = target_log_probs.exp()
            token_loss = -((1.0 - target_probs) ** 2.0) * target_log_probs
            loss = (token_loss * valid).sum() / valid.sum().clamp_min(1)
            return (loss, outputs) if return_outputs else loss

    class PrematureEosSFTTrainer(SFTTrainer):
        """Completion CE plus preregistered premature-EOS unlikelihood."""

        eos_token_ids: tuple[int, ...] = ()
        loss_component_sums = {"causal_ce": 0.0, "premature_eos_unlikelihood": 0.0, "observations": 0}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            token_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                safe_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)
            causal_ce = (token_ce * valid).sum() / valid.sum().clamp_min(1)

            true_eos = torch.zeros_like(valid)
            for eos_id in self.eos_token_ids:
                true_eos |= shift_labels.eq(eos_id)
            eligible = valid & ~true_eos
            eos_logits = shift_logits[..., list(self.eos_token_ids)]
            log_p_eos = torch.logsumexp(eos_logits, dim=-1) - torch.logsumexp(shift_logits, dim=-1)
            p_eos = log_p_eos.exp().clamp(max=1.0 - 1e-6)
            token_unlikelihood = -torch.log1p(-p_eos)
            eos_unlikelihood = (token_unlikelihood * eligible).sum() / eligible.sum().clamp_min(1)
            loss = causal_ce + 0.1 * eos_unlikelihood

            self.loss_component_sums["causal_ce"] += float(causal_ce.detach())
            self.loss_component_sums["premature_eos_unlikelihood"] += float(eos_unlikelihood.detach())
            self.loss_component_sums["observations"] += 1
            return (loss, outputs) if return_outputs else loss

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        raise RuntimeError("tokenizer has no EOS token id")
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    PrematureEosSFTTrainer.eos_token_ids = tuple(sorted(set(eos_token_ids)))
    if args.profile in {"h21", "h22"} and PrematureEosSFTTrainer.eos_token_ids != H21_EOS_TOKEN_IDS:
        raise RuntimeError(f"{args.profile.upper()} EOS binding failure: expected {H21_EOS_TOKEN_IDS}, got {PrematureEosSFTTrainer.eos_token_ids}")
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    dataset = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)
    mask_preflight = validate_completion_boundaries(dataset, tokenizer, args.max_length)
    print(json.dumps({"completion_only_preflight": mask_preflight}, ensure_ascii=False))
    if args.preflight_only:
        return 0

    gpu_uuids = visible_gpu_uuids()
    if gpu_uuids != [args.expected_gpu_uuid]:
        raise RuntimeError(
            f"GPU isolation failure: expected only {[args.expected_gpu_uuid]}, got {gpu_uuids}"
        )

    args.output.mkdir(parents=True, exist_ok=False)

    model_load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{args.gpu_memory_gib}GiB", "cpu": "96GiB"},
        offload_folder=str(args.output / "offload"),
        trust_remote_code=True,
    )
    model_load_seconds = time.monotonic() - model_load_started
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
    trainer_class = TailWeightedSFTTrainer if args.profile == "h19" else FocalSFTTrainer if args.profile == "h20" else PrematureEosSFTTrainer if args.profile in {"h21", "h22"} else SFTTrainer
    trainer_setup_started = time.monotonic()
    trainer = trainer_class(model=model, args=config, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)
    trainer_setup_seconds = time.monotonic() - trainer_setup_started
    if args.profile in {"h19", "h20", "h21", "h22"}:
        # The custom objective returns an already normalized micro-batch mean.
        # Qwen accepts arbitrary loss kwargs, which otherwise tells Trainer to
        # skip its gradient-accumulation division and scales gradients 16x.
        trainer.model_accepts_loss_kwargs = False
    train_started = time.monotonic()
    train_result = trainer.train()
    train_seconds = time.monotonic() - train_started
    save_started = time.monotonic()
    trainer.save_model(str(args.output / "adapter"))
    save_seconds = time.monotonic() - save_started
    adapter_tree_sha256 = tree_digest(args.output / "adapter")
    manifest = {
        "status": "candidate",
        "profile": args.profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "base_model_provenance": base_model_provenance,
        "dataset": str(args.data),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "rank": args.rank,
        "gpu_memory_gib": args.gpu_memory_gib,
        "seed": args.seed,
        "dataset_sha256": dataset_sha256,
        "expected_dataset_sha256": args.expected_data_sha256,
        "visible_gpu_uuids": gpu_uuids,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "completion_only_preflight": mask_preflight,
        "loss_objective": (
            {
                "mode": "completion-tail-weighted-causal-ce",
                "head_weight": 1.0,
                "tail_weight": 2.0,
                "split": "floor(valid_completion_tokens/2)",
            }
            if args.profile == "h19"
            else {"mode": "completion-only-focal-causal-ce", "gamma": 2.0, "normalization": "valid-completion-token-count"}
            if args.profile == "h20"
            else {
                "mode": "completion-only-causal-ce-plus-premature-eos-unlikelihood",
                "lambda": 0.1,
                "eos_token_ids": list(PrematureEosSFTTrainer.eos_token_ids),
                "normalization": "valid-completion-token-count-per-component",
                "mean_component_losses": {
                    key: value / max(1, PrematureEosSFTTrainer.loss_component_sums["observations"])
                    for key, value in PrematureEosSFTTrainer.loss_component_sums.items()
                    if key != "observations"
                },
                "component_observations": PrematureEosSFTTrainer.loss_component_sums["observations"],
            }
            if args.profile in {"h21", "h22"}
            else {"mode": "completion-only-causal-ce"}
        ),
        "timing": {
            "model_load_seconds": model_load_seconds,
            "trainer_setup_seconds": trainer_setup_seconds,
            "train_seconds": train_seconds,
            "trainer_reported_train_runtime_seconds": train_result.metrics.get("train_runtime"),
            "save_seconds": save_seconds,
            "total_seconds_before_manifest_write": time.monotonic() - total_started,
        },
        "adapter_layout": "persona-on-declared-base",
        "adapter_tree_sha256": adapter_tree_sha256,
        "promotion": "manual-only",
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
