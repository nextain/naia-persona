#!/usr/bin/env python3
"""Fail-closed validator for Naia persona JSONL datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SENSITIVE_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?82[- ]?)?0?1[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)"),
    "resident_id": re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)"),
    "secret": re.compile(r"\b(?:api[_ -]?key|token|password|secret)\s*[:=]\s*\S+", re.I),
}


def validate(path: Path) -> tuple[int, int, list[str]]:
    rows = 0
    fingerprints: set[str] = set()
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            rows += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue

            messages = row.get("messages")
            meta = row.get("meta")
            if not isinstance(meta, dict) or meta.get("consent") is not True:
                errors.append(f"line {line_number}: meta.consent must be true")
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"line {line_number}: messages must contain at least two entries")
                continue
            if [messages[-2].get("role"), messages[-1].get("role")] != ["user", "assistant"]:
                errors.append(f"line {line_number}: final turn must be user then assistant")

            normalized: list[dict[str, str]] = []
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    errors.append(f"line {line_number}: message {index} must be an object")
                    continue
                role = message.get("role")
                content = message.get("content")
                if role not in {"system", "user", "assistant"}:
                    errors.append(f"line {line_number}: message {index} has invalid role")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"line {line_number}: message {index} has blank content")
                    continue
                for label, pattern in SENSITIVE_PATTERNS.items():
                    if pattern.search(content):
                        errors.append(f"line {line_number}: message {index} contains possible {label}")
                normalized.append({"role": str(role), "content": content.strip()})

            digest = hashlib.sha256(
                json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if digest in fingerprints:
                errors.append(f"line {line_number}: duplicate conversation")
            fingerprints.add(digest)

    if rows == 0:
        errors.append("dataset contains no records")
    return rows, len(fingerprints), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    if not args.dataset.is_file():
        print(json.dumps({"ok": False, "errors": ["dataset file not found"]}))
        return 2
    rows, unique, errors = validate(args.dataset)
    print(json.dumps({"ok": not errors, "rows": rows, "unique": unique, "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
