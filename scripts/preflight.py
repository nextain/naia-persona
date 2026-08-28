#!/usr/bin/env python3
"""Read-only compatibility checks to run before allocating Qwen3.8 weights."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoConfig, AutoTokenizer
    from trl import SFTConfig

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type != "qwen3_5":
        raise RuntimeError(f"expected Qwen3.8 model_type qwen3_5, got {model_type!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "preflight-user"},
            {"role": "assistant", "content": "preflight-assistant"},
        ],
        tokenize=False,
        enable_thinking=False,
    )
    if "preflight-user" not in rendered or "preflight-assistant" not in rendered:
        raise RuntimeError("chat template did not preserve the smoke-test messages")

    parameters = inspect.signature(SFTConfig).parameters
    if "completion_only_loss" not in parameters or "max_length" not in parameters:
        raise RuntimeError("installed TRL lacks required SFTConfig options")

    adapter_parent = None
    if args.adapter:
        adapter_file = Path(args.adapter) / "adapter_config.json"
        payload = json.loads(adapter_file.read_text(encoding="utf-8"))
        if payload.get("peft_type") != "LORA" or payload.get("task_type") != "CAUSAL_LM":
            raise RuntimeError("adapter is not a causal-language-model LoRA")
        adapter_parent = payload.get("base_model_name_or_path")
        if not adapter_parent:
            raise RuntimeError("adapter has no base_model_name_or_path")

    print(
        json.dumps(
            {
                "status": "pass",
                "model_type": model_type,
                "architectures": getattr(config, "architectures", None),
                "adapter_parent": adapter_parent,
                "loss": "completion-only",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
