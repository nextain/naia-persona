#!/usr/bin/env python3
"""Build the H26 training artifact: less identity, more ordinary voice.

The persona benchmark found what 25 lexically-scored hypotheses never did. H24
scores identity 10/10 and capability 8/8, but voice 3/8: asked to greet, to say
what it is unsure about, or to celebrate, it recites its identity blurb instead.
Three of the four English non-identity questions came back as the same
self-introduction.

The cause is visible in the data. Identity is 136 of 239 rows, 57 percent, and
those 136 rows are only 30 unique questions multiplied by five style prefixes.
English rows sit almost entirely in that identity block, so an English prompt's
strongest attractor is the self-introduction.

This builder does two things and nothing else: it collapses the prefix inflation
so identity keeps its 30 questions without dominating, and it appends ordinary
conversational rows in which Naia never states who it is. No policy row returns.
No benchmark prompt is copied, and that is checked rather than promised.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data-private/datasets/naia-v12/train.jsonl"
VOICE = REPO / "examples/naia-v14/source.jsonl"
BENCH = REPO / "benchmark/persona-benchmark-v1.json"
HELDOUT = REPO / "examples/naia-v13/identity-confirmation-v3-prompts.json"
OUTPUT_DIR = REPO / "data-private/datasets/naia-v14"

BASE_SHA256 = "4be4cd1208291049c51dbad57a199027a1933799aa8c6ff09e8cb52dee901a39"

# Identity keeps its 30 questions but stops being counted five times. These two
# prefix groups are kept; the other three added no semantic variety.
KEEP_PREFIXES = {"(none)", "과장 없이 답해줘."}

VOICE_AXES = {
    "warmth": 12,
    "celebration": 12,
    "candour": 12,
    "greeting": 12,
    "uncertainty": 12,
    "everyday": 12,
}

EXPECTED_IDENTITY = 53
EXPECTED_OTHER = 103
EXPECTED_VOICE = 72
EXPECTED_TOTAL = 228
ACCUMULATION = 16
EPOCHS = 6
TARGET_STEPS = 90
NGRAM = 12

PREFIX = re.compile(r"^(.*?답해줘\.\s*)")
# Naia may be warm, blunt, or playful in these rows, but it may not answer an
# ordinary question by announcing itself. That is the defect being removed.
IDENTITY_MARKERS = ("나이아", "naia", "리퀴드 캣", "liquid cat")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_of(source_id: str) -> str:
    return re.sub(r"-\d+$", "", re.sub(r"^naia-v\d+(-replay-\d+-naia-v\d+)?-", "", source_id))


def prefix_of(prompt: str) -> str:
    match = PREFIX.match(prompt)
    return match.group(1).strip() if match else "(none)"


def language(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).casefold()


def ngrams(text: str, n: int) -> set[str]:
    flat = normalize(text)
    return {flat[i : i + n] for i in range(max(0, len(flat) - n + 1))}


def main() -> None:
    if digest(BASE) != BASE_SHA256:
        raise RuntimeError("naia-v12 base hash mismatch; refusing to build H26")

    base_lines = [line for line in BASE.read_text(encoding="utf-8").splitlines() if line]
    kept_lines: list[str] = []
    dropped = 0
    for line in base_lines:
        row = json.loads(line)
        source_id = row["meta"]["source_id"]
        if axis_of(source_id) == "identity" and prefix_of(row["messages"][0]["content"]) not in KEEP_PREFIXES:
            dropped += 1
            continue
        kept_lines.append(line)

    identity_kept = sum(
        1 for line in kept_lines if axis_of(json.loads(line)["meta"]["source_id"]) == "identity"
    )
    other_kept = len(kept_lines) - identity_kept
    if identity_kept != EXPECTED_IDENTITY or other_kept != EXPECTED_OTHER:
        raise RuntimeError(f"expected {EXPECTED_IDENTITY}+{EXPECTED_OTHER}, got {identity_kept}+{other_kept}")

    voice_lines = [line for line in VOICE.read_text(encoding="utf-8").splitlines() if line]
    if len(voice_lines) != EXPECTED_VOICE:
        raise RuntimeError(f"expected {EXPECTED_VOICE} voice rows, found {len(voice_lines)}")
    voice_rows = [json.loads(line) for line in voice_lines]

    ids = [row["meta"]["source_id"] for row in voice_rows]
    if len(set(ids)) != EXPECTED_VOICE or any(not i.startswith("naia-v14-") for i in ids):
        raise RuntimeError("voice source IDs must be unique naia-v14 IDs")
    if any(row["meta"].get("provenance") != "public-synthetic" for row in voice_rows):
        raise RuntimeError("every voice row must be public-synthetic")

    counts = Counter(axis_of(i) for i in ids)
    if counts != Counter(VOICE_AXES):
        raise RuntimeError(f"voice axis counts {dict(counts)} do not match {VOICE_AXES}")

    for row in voice_rows:
        question, answer = row["messages"][0]["content"], row["messages"][1]["content"]
        if language(question) != language(answer):
            raise RuntimeError(f"language mismatch in {row['meta']['source_id']}")
        low = answer.casefold()
        if any(marker in low for marker in IDENTITY_MARKERS):
            raise RuntimeError(
                f"{row['meta']['source_id']} announces its identity; these rows exist to teach "
                "ordinary voice, and reciting the name here would reinforce the defect"
            )
    per_axis_lang = Counter((axis_of(r["meta"]["source_id"]), language(r["messages"][0]["content"])) for r in voice_rows)
    for axis in VOICE_AXES:
        if per_axis_lang[(axis, "ko")] != 6 or per_axis_lang[(axis, "en")] != 6:
            raise RuntimeError(f"voice axis {axis} is not 6 Korean and 6 English")

    # No curriculum row may share a long span with a benchmark or held-out prompt.
    judged: list[tuple[str, str]] = []
    for case in json.loads(BENCH.read_text(encoding="utf-8"))["cases"]:
        judged.append((case["id"], case["prompt"]))
    for case in json.loads(HELDOUT.read_text(encoding="utf-8")):
        judged.append((case["id"], case["prompt"]))
    leaks = []
    for row in voice_rows:
        mine = ngrams(row["messages"][0]["content"], NGRAM)
        for case_id, prompt in judged:
            shared = mine & ngrams(prompt, NGRAM)
            if shared:
                leaks.append((row["meta"]["source_id"], case_id, sorted(shared)[:2]))
    if leaks:
        raise RuntimeError(f"curriculum overlaps judged prompts: {leaks[:5]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = OUTPUT_DIR / "train.jsonl"
    train.write_text("\n".join(kept_lines + voice_lines) + "\n", encoding="utf-8")

    total = len(kept_lines) + len(voice_lines)
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} rows, wrote {total}")
    steps_per_epoch = -(-total // ACCUMULATION)
    steps = steps_per_epoch * EPOCHS
    if steps != TARGET_STEPS:
        raise RuntimeError(f"{total} rows at {EPOCHS} epochs gives {steps} steps, not {TARGET_STEPS}")

    manifest = {
        "schema_version": 1,
        "experiment": "H26",
        "derivation": "naia-v12 with identity prefix inflation collapsed, plus an ordinary-voice curriculum",
        "base_sha256": BASE_SHA256,
        "voice_sha256": digest(VOICE),
        "train_sha256": digest(train),
        "rows": {
            "identity": identity_kept,
            "identity_dropped_as_duplicate_prefix": dropped,
            "other_inherited": other_kept,
            "voice_appended": len(voice_lines),
            "total": total,
        },
        "composition_shift": {
            "identity_share_before": "136/239 = 57%",
            "identity_share_after": f"{identity_kept}/{total} = {identity_kept / total * 100:.0f}%",
            "voice_share_after": f"{len(voice_lines)}/{total} = {len(voice_lines) / total * 100:.0f}%",
        },
        "voice_axes": dict(VOICE_AXES),
        "voice_language_balance": "6 Korean and 6 English per axis",
        "voice_rows_never_state_identity": True,
        "policy_rows_restored": 0,
        "rows_rewritten": 0,
        "split_or_shuffle": False,
        "optimizer_steps": {"per_epoch": steps_per_epoch, "epochs": EPOCHS, "total": steps},
        "overlap_check": {"ngram": NGRAM, "against": ["persona-benchmark-v1", "identity-confirmation-v3"], "shared": 0},
        "why": (
            "H24 scored voice 3/8 on the persona benchmark because ordinary prompts, "
            "especially English ones, returned the identity blurb. Identity was 57 percent "
            "of the data and English rows sat inside it."
        ),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
