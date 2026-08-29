#!/usr/bin/env python3
"""H2: rescore saved v1 answers after evaluator normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_eval_suite import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = {case["id"]: case for case in json.loads(args.suite.read_text(encoding="utf-8"))}
    result = {"reports": []}
    for report_path in args.report:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        changes = []
        rescored = []
        for record in report["records"]:
            new_score, _ = score(cases[record["id"]], record["answer"])
            rescored.append({"id": record["id"], "old": record["score"], "new": new_score})
            if new_score != record["score"]:
                changes.append(record["id"])
        result["reports"].append({"path": str(report_path), "changes": changes, "records": rescored})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
