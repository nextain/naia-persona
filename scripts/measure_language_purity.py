#!/usr/bin/env python3
"""Count foreign-script intrusion in saved benchmark answers, for every candidate.

The persona benchmark scored H32 at 97 of 100 while four of its answers carried
Chinese or Japanese fragments in grammatical positions — 替你 standing where 너
대신 belongs, それを standing where 그것을 belongs. Six of the eight intrusions
found across all candidates were judged pass, because no requirement asks about
the language the answer is written in. A ruler cannot report a defect on an axis
it does not have, and neither the judge-stability check nor the parent control
can find one either: both ask whether the ruler measures what it claims, not
whether something it never measures is getting worse.

This is a separate, deterministic measurement over answers that are already
saved. It needs no model, no judge, and no GPU, so every past candidate can be
rescored at once. It is reported beside the benchmark and is not part of it:
adding an axis to the benchmark is a change to the benchmark, and that decision
belongs to the person whose gate it is.

A fragment is counted only inside a Korean answer, and only when it sits without
a Latin gloss beside it. The parent writes 痛点(Pain Point), which is quoting a
term; the candidates write 한处在에서, which is a token that displaced a word.

Usage:
  measure_language_purity.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data-private/runs/benchmark"

FOREIGN = re.compile(r"[぀-ヿ一-鿿]+")
HANGUL = re.compile(r"[가-힣]")
LATIN_GLOSS = re.compile(r"^\s*[(（]\s*[A-Za-z]")

CANDIDATES = [
    ("parent", "-", "persona-benchmark-parent.json"),
    ("H24", "16", "persona-benchmark-h24-v2.json"),
    ("H26", "16", "persona-benchmark-h26.json"),
    ("H27", "16", "persona-benchmark-h27.json"),
    ("H28", "16", "persona-benchmark-h28.json"),
    ("H29", "16", "persona-benchmark-h29.json"),
    ("H30", "32", "persona-benchmark-h30.json"),
    ("H31", "64", "persona-benchmark-h31.json"),
    ("H32", "128", "persona-benchmark-h32.json"),
]


def intrusions(answer: str) -> list[str]:
    """Fragments that displaced a Korean word, as opposed to a quoted term."""
    if not HANGUL.search(answer):
        return []
    found = []
    for match in FOREIGN.finditer(answer):
        if LATIN_GLOSS.match(answer[match.end() : match.end() + 12]):
            continue
        found.append(match.group())
    return found


def main() -> None:
    rows = []
    for label, rank, filename in CANDIDATES:
        path = RUNS / filename
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        hits = []
        for record in run["records"]:
            found = intrusions(record["answer"])
            if found:
                hits.append(
                    {
                        "id": record["id"],
                        "verdict": record["verdict"],
                        "fragments": found,
                        "answer": record["answer"],
                    }
                )
        passed_anyway = sum(1 for h in hits if h["verdict"] == "pass")
        rows.append(
            {
                "label": label,
                "rank": rank,
                "answers_with_intrusion": len(hits),
                "of_those_judged_pass": passed_anyway,
                "benchmark_total": sum(1 for r in run["records"] if r["verdict"] == "pass"),
                "cases": hits,
            }
        )

    out = RUNS / "language-purity.json"
    out.write_text(json.dumps({"schema_version": 1, "candidates": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'후보':<8}{'rank':>6}{'벤치점수':>9}{'혼입답변':>9}{'그중통과':>9}")
    for row in rows:
        print(f"{row['label']:<8}{row['rank']:>6}{row['benchmark_total']:>9}{row['answers_with_intrusion']:>9}{row['of_those_judged_pass']:>9}")
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
