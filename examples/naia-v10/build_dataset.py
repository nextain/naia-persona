#!/usr/bin/env python3
"""Build Naia v10: structured completion formatting over H17's curriculum."""

from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent / "naia-v7"


def structured_answer(answer: str) -> str:
    sentences = [part.strip() for part in answer.split(". ") if part.strip()]
    sentences = [part if part.endswith((".", "!", "?")) else part + "." for part in sentences]
    judgment = sentences[0]
    limit = sentences[1] if len(sentences) > 1 else sentences[0]
    action = " ".join(sentences[2:]) if len(sentences) > 2 else limit
    return f"판단: {judgment}\n금지·한계: {limit}\n대안·조치: {action}"


def clone_structured(row: dict, replay_index: int) -> dict:
    replay = json.loads(json.dumps(row, ensure_ascii=False))
    original_id = row["meta"]["source_id"]
    replay["meta"]["source_id"] = f"naia-v10-structured-{replay_index}-{original_id}"
    replay["meta"]["replay_of"] = original_id
    replay["meta"]["format_variant"] = "judgment-limit-action-v1"
    replay["messages"][-1]["content"] = structured_answer(replay["messages"][-1]["content"])
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
    assert len(inherited) == 372
    assert len(canonical) == 64

    rows = list(inherited)
    for replay_index in range(1, 3):
        rows.extend(clone_structured(row, replay_index) for row in canonical)

    assert len(rows) == 500
    assert len({row["meta"]["source_id"] for row in rows}) == 500
    output = ROOT / "source.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
