#!/usr/bin/env python3
"""Build the H27 training artifact: answer diversity, and format instructions that outrank the identity reflex.

H26 cut identity rows from 136 to 53 and the reflex survived. Measuring the data
showed why: those 53 rows carry only twelve distinct answers, each repeated four
to six times. H26 reduced the prompt count, not the answer count, so the twelve
memorised sentences kept their weight. Four of the 100 benchmark answers came
back as a training sentence word for word, two of them on prompts that had
nothing to do with identity — including one that says 이름 말고, not your name.

H27 changes two things. Identity becomes 30 rows with 30 distinct answers, most
of which answer the question instead of reciting a self-introduction. And a
format-compliance curriculum teaches that an explicit instruction — JSON only,
three lines, casual register, one sentence, prose, answer-only — outranks the
urge to introduce oneself.

The builder refuses to write a dataset that repeats the old mistake: no identity
answer may appear more than twice, and no format-compliance answer may state the
character's name, because in those rows the name is exactly what is intruding.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data-private/datasets/naia-v12/train.jsonl"
IDENTITY = REPO / "examples/naia-v15/identity.jsonl"
PARENT_DENIAL = REPO / "examples/naia-v15/parent-denial.jsonl"
FIDELITY = REPO / "examples/naia-v15/fidelity.jsonl"
TONE = REPO / "examples/naia-v14/source.jsonl"
PATCH = REPO / "examples/naia-v16/patch.jsonl"
BENCH = REPO / "benchmark/persona-benchmark-v2.json"
HELDOUT = REPO / "examples/naia-v13/identity-confirmation-v3-prompts.json"
OUTPUT_DIR = REPO / "data-private/datasets/naia-v15"

BASE_SHA256 = "4be4cd1208291049c51dbad57a199027a1933799aa8c6ff09e8cb52dee901a39"

EXPECTED_IDENTITY = 30
EXPECTED_PARENT = 10
EXPECTED_FIDELITY = 72
EXPECTED_TONE = 72
EXPECTED_OTHER = 81
EXPECTED_PATCH = 23
INHERITED_ANSWER_CAP = 3
EXPECTED_TOTAL = 288
ACCUMULATION = 16
EPOCHS = 5
TARGET_STEPS = 90
NGRAM = 12
MAX_ANSWER_REPEATS = 2

FIDELITY_AXES = {"format-json": 12, "format-lines": 12, "format-register": 12, "format-onesent": 12, "format-prose": 12, "format-terse": 12}
NAME_MARKERS = ("나이아", "naia", "리퀴드 캣", "liquid cat")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_of(source_id: str) -> str:
    return re.sub(r"-\d+$", "", re.sub(r"^naia-v\d+(-replay-\d+-naia-v\d+)?-", "", source_id))


def language(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).casefold()


def ngrams(text: str, n: int) -> set[str]:
    flat = normalize(text)
    return {flat[i : i + n] for i in range(max(0, len(flat) - n + 1))}


def read(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    if digest(BASE) != BASE_SHA256:
        raise RuntimeError("naia-v12 base hash mismatch; refusing to build H27")

    # Keep every inherited non-identity row; the old identity block is replaced wholesale.
    # The inherited block has the same defect this program just fixed in identity:
    # 103 rows carry 28 distinct answers, several repeated four times. Cap the
    # repeats so the trim that keeps the step count also reduces memorisation.
    seen: Counter = Counter()
    other_lines = []
    for line in read(BASE):
        row = json.loads(line)
        if axis_of(row["meta"]["source_id"]) == "identity":
            continue
        answer = row["messages"][1]["content"]
        if seen[answer] >= INHERITED_ANSWER_CAP:
            continue
        seen[answer] += 1
        other_lines.append(line)
    if len(other_lines) != EXPECTED_OTHER:
        raise RuntimeError(f"expected {EXPECTED_OTHER} inherited non-identity rows, got {len(other_lines)}")

    identity_lines = read(IDENTITY) + read(PARENT_DENIAL)
    fidelity_lines = read(FIDELITY)
    tone_lines = read(TONE)
    patch_lines = read(PATCH)
    for name, lines, expected in (
        ("identity", identity_lines, EXPECTED_IDENTITY + EXPECTED_PARENT),
        ("fidelity", fidelity_lines, EXPECTED_FIDELITY),
        ("tone", tone_lines, EXPECTED_TONE),
        ("patch", patch_lines, EXPECTED_PATCH),
    ):
        if len(lines) != expected:
            raise RuntimeError(f"expected {expected} {name} rows, found {len(lines)}")

    identity = [json.loads(line) for line in identity_lines]
    fidelity = [json.loads(line) for line in fidelity_lines]

    # The defect H27 exists to remove: a handful of sentences memorised hard enough
    # to surface on unrelated prompts. Cap the repeats at the source.
    answers = Counter(r["messages"][1]["content"].strip() for r in identity)
    worst = answers.most_common(1)[0]
    if worst[1] > MAX_ANSWER_REPEATS:
        raise RuntimeError(f"identity answer repeats {worst[1]} times, limit is {MAX_ANSWER_REPEATS}: {worst[0][:60]!r}")
    if len(answers) != EXPECTED_IDENTITY + EXPECTED_PARENT:
        raise RuntimeError(f"expected {EXPECTED_IDENTITY + EXPECTED_PARENT} distinct identity answers, got {len(answers)}")
    # H27 diluted the name itself: 20 of 30 identity answers never said it, and the
    # parent model's own self-identification came back. Every identity answer must
    # carry the anchor, while the sentences around it stay distinct.
    unanchored = [r["meta"]["source_id"] for r in identity if not re.search(r"나이아|naia", r["messages"][1]["content"], re.I)]
    if unanchored:
        raise RuntimeError(f"identity answers without the name anchor: {unanchored[:6]}")

    # In a format-compliance row the name is usually the intruder — it is what pushes
    # aside the instruction the row exists to teach. It is legitimate only when the
    # prompt actually asked for it, as in "JSON with your name and role". Anywhere
    # else it would teach the very reflex being removed.
    for row in fidelity:
        answer = row["messages"][1]["content"].casefold()
        question = row["messages"][0]["content"].casefold()
        if not any(marker in answer for marker in NAME_MARKERS):
            continue
        if any(word in question for word in ("이름", "name")):
            continue
        raise RuntimeError(
            f"{row['meta']['source_id']} states the character name although the prompt never asked for it; "
            "these rows teach that an explicit instruction outranks the identity reflex"
        )

    counts = Counter(axis_of(r["meta"]["source_id"]) for r in fidelity)
    if counts != Counter(FIDELITY_AXES):
        raise RuntimeError(f"fidelity axis counts {dict(counts)} do not match {FIDELITY_AXES}")

    for name, rows in (("identity", identity), ("fidelity", fidelity)):
        for row in rows:
            question, answer = row["messages"][0]["content"], row["messages"][1]["content"]
            # Terse and JSON answers are language-neutral by design.
            if axis_of(row["meta"]["source_id"]) in {"format-terse", "format-json"}:
                continue
            if language(question) != language(answer):
                raise RuntimeError(f"language mismatch in {row['meta']['source_id']}")

    # No new row may share a long span with a judged prompt.
    judged = [(c["id"], c["prompt"]) for c in json.loads(BENCH.read_text(encoding="utf-8"))["cases"]]
    judged += [(c["id"], c["prompt"]) for c in json.loads(HELDOUT.read_text(encoding="utf-8"))]
    leaks = []
    patch = [json.loads(line) for line in patch_lines]
    for row in identity + fidelity + patch:
        mine = ngrams(row["messages"][0]["content"], NGRAM)
        for case_id, prompt in judged:
            if mine & ngrams(prompt, NGRAM):
                leaks.append((row["meta"]["source_id"], case_id))
    if leaks:
        raise RuntimeError(f"curriculum overlaps judged prompts: {leaks[:6]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = OUTPUT_DIR / "train.jsonl"
    all_lines = identity_lines + other_lines + tone_lines + fidelity_lines + patch_lines
    train.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    total = len(all_lines)
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} rows, wrote {total}")
    steps_per_epoch = -(-total // ACCUMULATION)
    steps = steps_per_epoch * EPOCHS
    if steps != TARGET_STEPS:
        raise RuntimeError(f"{total} rows at {EPOCHS} epochs gives {steps} steps, not {TARGET_STEPS}")

    manifest = {
        "schema_version": 1,
        "experiment": "H27",
        "root_cause_addressed": "53 identity rows held only 12 distinct answers; the memorised sentences surfaced on unrelated prompts and overrode explicit format instructions",
        "base_sha256": BASE_SHA256,
        "identity_sha256": digest(IDENTITY),
        "fidelity_sha256": digest(FIDELITY),
        "tone_sha256": digest(TONE),
        "train_sha256": digest(train),
        "rows": {
            "identity": len(identity_lines),
            "inherited_other": len(other_lines),
            "tone": len(tone_lines),
            "fidelity": len(fidelity_lines),
            "total": total,
        },
        "answer_diversity": {
            "identity_distinct_answers": len(answers),
            "identity_rows": len(identity_lines),
            "max_repeats": max(answers.values()),
            "previous": "53 rows / 12 distinct answers, max 6 repeats",
        },
        "fidelity_axes": dict(FIDELITY_AXES),
        "fidelity_rows_never_state_the_name": True,
        "policy_rows_restored": 0,
        "split_or_shuffle": False,
        "optimizer_steps": {"per_epoch": steps_per_epoch, "epochs": EPOCHS, "total": steps},
        "overlap_check": {"ngram": NGRAM, "shared": 0},
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
