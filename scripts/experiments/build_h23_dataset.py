#!/usr/bin/env python3
"""Build the H23 training artifact by removing the unrequested policy curriculum.

H23 is a data-composition-only change. It writes no new text. Every output row is
a byte-identical row already present in the frozen naia-v11 artifact, in the same
relative order. The rows removed are the safety, privacy, consent, and refusal
curriculum that entered the program through a charter goal no user directive
asked for; see the charter `authority_note` and the H23 ledger entry.

Axis membership is enumerated, never inferred by keyword, so the split is
auditable and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data-private/datasets/naia-v11/train.jsonl"
OUTPUT_DIR = REPO / "data-private/datasets/naia-v12"
SOURCE_SHA256 = "9a83f29080726b25181538a89912876ea3249fe72490e2f6e7a87a65f7650282"

# Character, behaviour, and general capability. This is what the program was
# asked to teach.
KEEP_AXES = {
    "identity",
    "behavior",
    "boundary",
    "general",
    "calm-debugging",
    "json-validity",
}

# Refusal, safety, privacy, consent, and content policy. No user directive asked
# for any of these; they are removed rather than rebalanced.
DROP_AXES = {
    "account-recall",
    "agency-adversarial",
    "choice-privacy",
    "consent",
    "credential-defense",
    "credential-safety",
    "cross-user",
    "deletion-lineage",
    "financial",
    "financial-complete",
    "memory-limit",
    "nightly-opt-in",
    "nightly-optin",
    "per-user-isolation",
    "privacy",
    "privacy-default",
    "runtime-boundary",
    "runtime-limit",
    "safety",
    "self-harm-support",
    "sensitive-log",
    "user-agency",
    "withdrawal-complete",
    "withdrawal-lineage",
}

EXPECTED_KEPT = 239
EXPECTED_DROPPED = 235


def axis(source_id: str) -> str:
    """Strip the dataset-version and replay prefixes and the trailing index."""
    tail = re.sub(r"^naia-v\d+(-replay-\d+-naia-v\d+)?-", "", source_id)
    return re.sub(r"-\d+$", "", tail)


def main() -> None:
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("naia-v11 source hash mismatch; refusing to build H23")

    lines = [line for line in raw.decode("utf-8").splitlines() if line]
    kept_lines: list[str] = []
    kept_ids: list[str] = []
    dropped_ids: list[str] = []
    seen_axes: set[str] = set()

    for line in lines:
        row = json.loads(line)
        source_id = row.get("meta", {}).get("source_id") or ""
        name = axis(source_id)
        seen_axes.add(name)
        if name in KEEP_AXES:
            kept_lines.append(line)
            kept_ids.append(source_id)
        elif name in DROP_AXES:
            dropped_ids.append(source_id)
        else:
            raise RuntimeError(f"unclassified axis {name!r} from {source_id!r}")

    unknown = seen_axes - KEEP_AXES - DROP_AXES
    if unknown:
        raise RuntimeError(f"axes not covered by the enumerated split: {sorted(unknown)}")
    if len(kept_lines) != EXPECTED_KEPT or len(dropped_ids) != EXPECTED_DROPPED:
        raise RuntimeError(
            f"expected {EXPECTED_KEPT} kept and {EXPECTED_DROPPED} dropped, "
            f"got {len(kept_lines)} and {len(dropped_ids)}"
        )

    # Every kept line must still be one of the original lines, unmodified, and in
    # the original relative order.
    if any(line not in lines for line in kept_lines):
        raise RuntimeError("a kept row is not byte-identical to its source row")
    positions = [lines.index(line) for line in kept_lines]
    if positions != sorted(positions):
        raise RuntimeError("kept rows changed relative order")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = OUTPUT_DIR / "train.jsonl"
    train.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment": "H23",
        "derivation": "subset-of-naia-v11-no-new-or-edited-text",
        "source": "data-private/datasets/naia-v11/train.jsonl",
        "source_sha256": SOURCE_SHA256,
        "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "rows": {
            "source_total": len(lines),
            "kept": len(kept_lines),
            "dropped": len(dropped_ids),
        },
        "keep_axes": sorted(KEEP_AXES),
        "drop_axes": sorted(DROP_AXES),
        "kept_source_ids_in_order": kept_ids,
        "dropped_source_ids_in_order": dropped_ids,
        "split_or_shuffle": False,
        "rows_added_or_rewritten": 0,
        "alpha_or_private_persona_used": False,
        "removal_reason": (
            "The dropped axes teach refusal, safety, privacy, and consent behavior "
            "that no user directive requested. They entered through a charter goal "
            "written by an agent and were inherited by H1 through H22."
        ),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_in_order")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
