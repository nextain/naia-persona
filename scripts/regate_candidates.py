#!/usr/bin/env python3
"""Re-decide every saved candidate under the corrected gate.

The capability axis was written into the benchmark with an absolute threshold of
20 out of 20, but the same file's control_requirement already declared it a
regression check: "Capability should hold for the parent; that is what makes it
a regression check." Running the control showed the parent scores 18. A gate of
20 on an axis the parent cannot reach is not a regression check — it asks the
fine-tune to beat the parent on the one axis fine-tuning does not move.

Corrected on 2026-09-01 by user directive. The four persona thresholds are
untouched; only the capability axis changes, and it changes from an absolute
number to the comparison the benchmark said it was making all along.

Nothing is re-generated and nothing is re-judged. Every answer and every verdict
already exists on disk.

Usage:
  regate_candidates.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data-private/runs/benchmark"
BENCH = REPO / "benchmark/persona-benchmark-v2.json"

CANDIDATES = [
    ("H24", "16", "persona-benchmark-h24-v2.json"),
    ("H26", "16", "persona-benchmark-h26.json"),
    ("H27", "16", "persona-benchmark-h27.json"),
    ("H28", "16", "persona-benchmark-h28.json"),
    ("H29", "16", "persona-benchmark-h29.json"),
    ("H30", "32", "persona-benchmark-h30.json"),
    ("H31", "64", "persona-benchmark-h31.json"),
    ("H32", "128", "persona-benchmark-h32.json"),
]


def axis_counts(path: Path) -> dict[str, int]:
    run = json.loads(path.read_text(encoding="utf-8"))
    return {axis: counts["pass"] for axis, counts in run["by_axis"].items()}


def main() -> None:
    axes = json.loads(BENCH.read_text(encoding="utf-8"))["axes"]
    absolute = {name: spec["threshold"] for name, spec in axes.items() if "threshold" in spec}
    relative = [name for name, spec in axes.items() if spec.get("gate") == "parent_relative"]

    parent = axis_counts(RUNS / "persona-benchmark-parent.json")

    header = ["후보", "rank"] + [axes[a]["korean"] for a in axes] + ["총점", "판정"]
    print(f"{'축':<10}" + "".join(f"{axes[a]['korean']:>8}" for a in axes))
    print(f"{'임계':<10}" + "".join(
        f"{(str(absolute[a]) if a in absolute else '부모≥' + str(parent[a])):>8}" for a in axes))
    print(f"{'원본':<10}" + "".join(f"{parent[a]:>8}" for a in axes) + f"  총 {sum(parent.values())}")
    print()

    rows = []
    for label, rank, filename in CANDIDATES:
        path = RUNS / filename
        if not path.exists():
            continue
        counts = axis_counts(path)
        verdicts = {}
        for axis in axes:
            if axis in absolute:
                verdicts[axis] = counts[axis] >= absolute[axis]
            else:
                verdicts[axis] = counts[axis] >= parent[axis]
        passed = all(verdicts.values())
        failed = [axes[a]["korean"] for a in axes if not verdicts[a]]
        rows.append({
            "label": label, "rank": rank, "by_axis": counts, "axis_pass": verdicts,
            "total": sum(counts.values()), "status": "pass" if passed else "fail",
            "failing_axes": failed,
        })
        marks = "".join(f"{str(counts[a]) + ('✓' if verdicts[a] else '✗'):>8}" for a in axes)
        print(f"{label:<5}r{rank:<4}{marks}  총 {sum(counts.values()):>3}  {'통과' if passed else '미달: ' + ', '.join(failed)}")

    out = RUNS / "regate-2026-09-01.json"
    out.write_text(json.dumps({
        "schema_version": 1,
        "gate": {"absolute": absolute, "parent_relative": {a: parent[a] for a in relative}},
        "parent": parent,
        "candidates": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
