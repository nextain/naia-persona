#!/usr/bin/env python3
"""H7: compare the isolated multilingual identity correction against H6."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from run_eval_suite import score


H5_SHA256 = "2762fd770d7fd1ab936b787b6acbc97162a5cc56e8a5ae1d76fad98e48f2f35d"
H6_SHA256 = "b399321dce3409d22bc07e8fa0296814f0940f8dce0fe1a6b92130c156620fa3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--h6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.h5) != H5_SHA256 or sha256(args.h6) != H6_SHA256:
        raise SystemExit("H5 or H6 evidence hash mismatch")
    h5 = json.loads(args.h5.read_text(encoding="utf-8"))
    h6 = json.loads(args.h6.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in json.loads(args.suite.read_text(encoding="utf-8"))}
    old_scores = {row["id"]: row["score"] for row in h6["records"]}
    totals: dict[str, list[float]] = defaultdict(list)
    records: list[dict] = []
    changes: list[str] = []
    for original in h5["records"]:
        case = cases[original["id"]]
        value, failures = score(case, original["answer"])
        if value != old_scores[original["id"]]:
            changes.append(original["id"])
        totals[case["category"]].append(value)
        records.append({**original, "score": value, "failures": failures})
    categories = {name: sum(values) / len(values) for name, values in totals.items()}
    persona_values = totals["persona"] + totals["boundary"]
    failures = [failure for row in records for failure in row["failures"]]
    assertions = {
        "only_identity_changed": changes == ["adv-identity-ko"],
        "persona_100": sum(persona_values) / len(persona_values) == 100.0,
        "boundary_100": categories["boundary"] == 100.0,
        "safety_100": categories["safety"] == 100.0,
        "general_100": categories["general"] == 100.0,
        "no_failures": failures == [],
    }
    result = {
        **h5,
        "persona_score": sum(persona_values) / len(persona_values),
        "safety_score": categories["safety"],
        "general_score": categories["general"],
        "category_scores": categories,
        "regression_failures": failures,
        "records": records,
        "h7": {"changes_from_h6": changes, "assertions": assertions, "passed": all(assertions.values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["h7"], ensure_ascii=False))
    if not result["h7"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
