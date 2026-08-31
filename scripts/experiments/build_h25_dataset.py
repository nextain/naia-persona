#!/usr/bin/env python3
"""Append the H25 identity-diversity curriculum to the frozen naia-v12 artifact.

H25 widens identity expression along six axes that were measured as missing from
the training data itself, not inferred from an evaluation result: 136 identity
rows are only 30 question stems multiplied by 5 style prefixes, 57 rows answer a
Korean prompt in English against the persona card's Voice rule, and there is not
one assertion-form identity challenge anywhere in the set.

The held-out confirmation suite was authored and frozen before any row here was
written. This builder re-checks that ordering by hash, and fails closed if any
curriculum prompt shares a long n-gram with a held-out prompt, because a suite
that leaks into its own curriculum confirms nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data-private/datasets/naia-v12/train.jsonl"
SOURCE = REPO / "examples/naia-v13/source.jsonl"
HELDOUT = REPO / "examples/naia-v13/identity-confirmation-v3-prompts.json"
OUTPUT_DIR = REPO / "data-private/datasets/naia-v13"

BASE_SHA256 = "4be4cd1208291049c51dbad57a199027a1933799aa8c6ff09e8cb52dee901a39"
HELDOUT_SHA256 = "1d0f28f4af9b961a6adf86b281c8e9dc3fb1cde65ca1e006451b1f4b14e8c7e5"

AXES = {
    "assertion-challenge": 8,
    "sustained-pressure": 8,
    "third-party-authority": 8,
    "incidental-identity": 8,
    "language-fidelity": 8,
    "character-depth": 8,
}
BASE_ROWS = 239
NEW_ROWS = 48
ACCUMULATION = 16
EPOCHS = 5
TARGET_STEPS = 90

# A curriculum prompt sharing this many consecutive characters with a held-out
# prompt is surface copying. Korean is not word-delimited, so this runs on
# characters after whitespace and punctuation are stripped.
NGRAM = 12


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).casefold()


def ngrams(text: str, n: int) -> set[str]:
    flat = normalize(text)
    return {flat[i : i + n] for i in range(max(0, len(flat) - n + 1))}


def axis_of(source_id: str) -> str:
    return re.sub(r"^naia-v13-", "", re.sub(r"-\d+$", "", source_id))


def language(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"


def main() -> None:
    if digest(BASE) != BASE_SHA256:
        raise RuntimeError("naia-v12 base hash mismatch; refusing to build H25")
    if digest(HELDOUT) != HELDOUT_SHA256:
        raise RuntimeError(
            "held-out suite hash differs from the one registered before curriculum authoring"
        )

    base_bytes = BASE.read_bytes()
    base_lines = [line for line in base_bytes.decode("utf-8").splitlines() if line]
    if len(base_lines) != BASE_ROWS:
        raise RuntimeError(f"expected {BASE_ROWS} inherited rows, found {len(base_lines)}")

    source_bytes = SOURCE.read_bytes()
    new_lines = [line for line in source_bytes.decode("utf-8").splitlines() if line]
    if len(new_lines) != NEW_ROWS:
        raise RuntimeError(f"expected {NEW_ROWS} new rows, found {len(new_lines)}")

    new_rows = [json.loads(line) for line in new_lines]
    ids = [row["meta"]["source_id"] for row in new_rows]
    if len(set(ids)) != NEW_ROWS or any(not i.startswith("naia-v13-") for i in ids):
        raise RuntimeError("H25 source IDs must be unique naia-v13 IDs")
    if any(row["meta"].get("provenance") != "public-synthetic" for row in new_rows):
        raise RuntimeError("every H25 addition must be public-synthetic")

    counts = Counter(axis_of(i) for i in ids)
    if counts != Counter(AXES):
        raise RuntimeError(f"axis counts {dict(counts)} do not match the preregistered {AXES}")

    # The persona card requires answering in the user's language. This is the
    # defect the curriculum exists to fix, so it is enforced, not hoped for.
    for row in new_rows:
        q, a = row["messages"][0]["content"], row["messages"][1]["content"]
        if language(q) != language(a):
            raise RuntimeError(f"language mismatch in {row['meta']['source_id']}: {q[:40]!r}")
    per_axis_lang = Counter((axis_of(r["meta"]["source_id"]), language(r["messages"][0]["content"])) for r in new_rows)
    for axis in AXES:
        if per_axis_lang[(axis, "ko")] != 4 or per_axis_lang[(axis, "en")] != 4:
            raise RuntimeError(f"axis {axis} is not 4 Korean and 4 English")

    # Fail closed on surface leakage between curriculum and held-out suite.
    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))
    heldout_ngrams: dict[str, set[str]] = {c["id"]: ngrams(c["prompt"], NGRAM) for c in heldout}
    leaks = []
    for row in new_rows:
        prompt = row["messages"][0]["content"]
        mine = ngrams(prompt, NGRAM)
        for case_id, theirs in heldout_ngrams.items():
            shared = mine & theirs
            if shared:
                leaks.append((row["meta"]["source_id"], case_id, sorted(shared)[:3]))
    if leaks:
        raise RuntimeError(f"curriculum overlaps held-out prompts: {leaks[:5]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = OUTPUT_DIR / "train.jsonl"
    train.write_bytes(base_bytes + source_bytes)
    written = train.read_bytes()
    if written[: len(base_bytes)] != base_bytes:
        raise RuntimeError("inherited rows changed")
    total = len([line for line in written.decode("utf-8").splitlines() if line])
    if total != BASE_ROWS + NEW_ROWS:
        raise RuntimeError(f"expected {BASE_ROWS + NEW_ROWS} rows, wrote {total}")

    steps_per_epoch = -(-total // ACCUMULATION)
    steps = steps_per_epoch * EPOCHS
    if steps != TARGET_STEPS:
        raise RuntimeError(f"{total} rows at {EPOCHS} epochs gives {steps} steps, not {TARGET_STEPS}")

    manifest = {
        "schema_version": 1,
        "experiment": "H25",
        "derivation": "naia-v12 byte prefix plus the frozen naia-v13 identity-diversity curriculum",
        "base_sha256": BASE_SHA256,
        "source_sha256": digest(SOURCE),
        "train_sha256": digest(train),
        "heldout_suite_sha256": HELDOUT_SHA256,
        "heldout_frozen_before_curriculum": True,
        "rows": {"inherited": BASE_ROWS, "appended": NEW_ROWS, "total": total},
        "axes": dict(AXES),
        "per_axis_language_balance": "4 Korean and 4 English each",
        "inherited_rows_edited": 0,
        "policy_rows_restored": 0,
        "split_or_shuffle": False,
        "optimizer_steps": {"per_epoch": steps_per_epoch, "epochs": EPOCHS, "total": steps},
        "overlap_check": {"ngram": NGRAM, "shared_ngrams_found": 0},
        "appended_source_ids_in_order": ids,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_in_order")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
