#!/usr/bin/env python3
"""Bake one LoRA adapter into a full-precision base checkpoint on CPU.

This is intentionally a separate, auditable step.  PEFT's trainable model can
activate only one ordinary adapter at a time, so a frozen unlock adapter must be
merged before training a second persona adapter if both effects are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # The published Qwen3.8 unlock adapter was exported through a wrapper that
    # inserted ``model.language_model`` into every tensor key.  Transformers'
    # native Qwen3.8 class exposes the same tensors under ``model``.  PEFT's
    # explicit regex mapping removes only that wrapper segment.
    key_mapping = {r"^model\.language_model\.": "model."}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(
            model, args.adapter, is_trainable=False, key_mapping=key_mapping
        )
    missing = [str(item.message) for item in caught if "missing adapter keys" in str(item.message)]
    if missing:
        raise RuntimeError("adapter key mapping is incomplete: " + missing[0])
    merged = model.merge_and_unload(safe_merge=True, progressbar=True)
    args.output.mkdir(parents=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(args.output)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "operation": "safe_merge_and_unload",
        "adapter_key_mapping": key_mapping,
        "dtype": "bfloat16",
        "base_sha256": tree_digest(args.base_model),
        "adapter_sha256": tree_digest(args.adapter),
    }
    (args.output / "naia-merge-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
