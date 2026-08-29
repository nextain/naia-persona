#!/usr/bin/env python3
"""Build Naia v8: balanced replay of the frozen H14 and H15 curricula."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent / "naia-v7"


def clone_with_replay(row: dict, replay_index: int) -> dict:
    replay = json.loads(json.dumps(row, ensure_ascii=False))
    original_id = row["meta"]["source_id"]
    replay["meta"]["source_id"] = f"naia-v8-replay-{replay_index}-{original_id}"
    replay["meta"]["replay_of"] = original_id
    return replay


def main() -> None:
    runpy.run_path(str(V7 / "build_dataset.py"), run_name="__main__")
    v7_rows = [
        json.loads(line)
        for line in (V7 / "source.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    inherited = v7_rows[:372]
    canonical = v7_rows[372:]
    assert len(v7_rows) == 436
    assert len(inherited) == 372
    assert len(canonical) == 64

    rows = list(inherited)
    for replay_index in range(1, 4):
        rows.extend(clone_with_replay(row, replay_index) for row in canonical)

    assert len(rows) == 564
    assert len({row["meta"]["source_id"] for row in rows}) == 564
    output = ROOT / "source.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
