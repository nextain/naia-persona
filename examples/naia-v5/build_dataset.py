#!/usr/bin/env python3
"""Build Naia v5: H11 curriculum plus three declared replays of H12 corrections."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V3 = ROOT.parent / "naia-v3"
V4 = ROOT.parent / "naia-v4"


def main() -> None:
    runpy.run_path(str(V4 / "build_dataset.py"), run_name="__main__")
    v4_rows = [
        json.loads(line)
        for line in (V4 / "source.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    original = v4_rows[:300]
    corrections = v4_rows[300:]
    assert len(original) == 300 and len(corrections) == 24

    rows = list(original)
    for replay_index in range(3):
        for correction in corrections:
            replay = json.loads(json.dumps(correction, ensure_ascii=False))
            original_id = correction["meta"]["source_id"]
            replay["meta"]["source_id"] = f"naia-v5-replay-{replay_index + 1}-{original_id}"
            replay["meta"]["replay_of"] = original_id
            rows.append(replay)

    # The first occurrence of each correction must be the declared replay target.
    # Include one canonical H12 correction before its two further replays while
    # retaining the H13 contract of exactly three correction copies total.
    correction_ids = {row["meta"]["source_id"] for row in corrections}
    for row in rows[300:324]:
        row["meta"].pop("replay_of", None)
        row["meta"]["source_id"] = row["meta"]["source_id"].replace("naia-v5-replay-1-", "")
    for row in rows[324:]:
        assert row["meta"]["replay_of"] in correction_ids

    assert len(rows) == 372
    output = ROOT / "source.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
