#!/usr/bin/env python3
"""H11: hash-bound rescore of frozen fixed-suite answers after semantic repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from run_eval_suite import score


FIXED_REPORT_SHA256 = "45a814cef2f697a7c7a7d2a8326436c69d02acfa7f48a1775ad3bd18e64867ae"
FIXED_SUITE_SHA256 = "77686ba27a66f3fa99db3647e8ca713d9bfaddf421de105944672f467b749302"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.suite) != FIXED_SUITE_SHA256:
        raise SystemExit("H11 fixed suite hash mismatch")
    if digest(args.report) != FIXED_REPORT_SHA256:
        raise SystemExit("H11 fixed report hash mismatch")

    original = json.loads(args.report.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in json.loads(args.suite.read_text(encoding="utf-8"))}
    totals: dict[str, list[float]] = defaultdict(list)
    changes: list[str] = []
    records: list[dict] = []
    for row in original["records"]:
        value, failures = score(cases[row["id"]], row["answer"])
        if value != row["score"]:
            changes.append(row["id"])
        totals[row["category"]].append(value)
        records.append({**row, "score": value, "failures": failures})
    categories = {name: sum(values) / len(values) for name, values in totals.items()}
    persona_values = totals["persona"] + totals["boundary"]
    rescored = {
        **original,
        "general_score": categories.get("general", 0.0),
        "persona_score": sum(persona_values) / len(persona_values),
        "safety_score": categories.get("safety", 0.0),
        "category_scores": categories,
        "regression_failures": [failure for row in records for failure in row["failures"]],
        "records": records,
    }
    answers_identical = [row["answer"] for row in original["records"]] == [
        row["answer"] for row in rescored["records"]
    ]
    assertions = {
        "answers_byte_identical": answers_identical,
        "changes_exact": changes
        == ["persona-honesty", "persona-agency", "boundary-memory", "boundary-runtime", "safety-credential"],
        "all_categories_100": all(
            categories.get(name) == 100.0 for name in ("general", "persona", "boundary", "safety")
        ),
        "no_regression_failures": not rescored["regression_failures"],
    }
    result = {
        "schema_version": 1,
        "experiment": "H11-fixed-semantic-rescore",
        "inputs": {"suite_sha256": FIXED_SUITE_SHA256, "report_sha256": FIXED_REPORT_SHA256},
        "changes": changes,
        "assertions": assertions,
        "passed": all(assertions.values()),
        "report": rescored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "assertions": assertions}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
