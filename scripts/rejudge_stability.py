#!/usr/bin/env python3
"""Measure how much of the benchmark's movement comes from the judge, not the model.

Candidate generation runs at temperature 0, so a served adapter answers the same
question the same way every time. Any case that flips between runs is therefore
either a real difference between two adapters or an unstable judge. This script
separates the two by holding the answers fixed: it re-judges saved answers from a
completed run several times and counts how often the same answer gets a different
verdict.

A case whose verdict is not reproducible on identical input is not measuring the
model. Knowing how many such cases exist bounds how much of any score difference
between two candidates can be believed.

Usage:
  rejudge_stability.py --run persona-benchmark-h29.json --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from run_persona_benchmark import CARD, BENCH, OUT_DIR, judge

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="filename under data-private/runs/benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("FATAL: GEMINI_API_KEY is not set")

    source = OUT_DIR / args.run
    run = json.loads(source.read_text(encoding="utf-8"))
    card = CARD.read_text(encoding="utf-8")
    requirements = {c["id"]: c["requirement"] for c in json.loads(BENCH.read_text(encoding="utf-8"))["cases"]}

    records = run["records"]
    jobs = [(rec, trial) for trial in range(args.repeats) for rec in records]

    def one(job):
        rec, trial = job
        verdict = judge(card, requirements[rec["id"]], rec["prompt"], rec["answer"], api_key)
        return rec["id"], trial, verdict["verdict"], verdict.get("reason", "")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, jobs))

    trials = {rec["id"]: [] for rec in records}
    for case_id, _trial, verdict, _reason in results:
        trials[case_id].append(verdict)

    original = {rec["id"]: rec["verdict"] for rec in records}
    axis = {rec["id"]: rec["axis"] for rec in records}

    unstable = []
    for case_id, verdicts in trials.items():
        seen = set(verdicts) | {original[case_id]}
        if len(seen) > 1:
            unstable.append(case_id)

    per_axis = Counter(axis[c] for c in unstable)
    totals = []
    for trial in range(args.repeats):
        totals.append(sum(1 for case_id in trials if trials[case_id][trial] == "pass"))

    report = {
        "schema_version": 1,
        "source_run": args.run,
        "source_label": run["label"],
        "benchmark_sha256": run["benchmark_sha256"],
        "repeats": args.repeats,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_total": sum(1 for v in original.values() if v == "pass"),
        "rejudged_totals": totals,
        "unstable_cases": sorted(unstable),
        "unstable_count": len(unstable),
        "unstable_by_axis": dict(per_axis),
        "detail": {c: {"original": original[c], "rejudged": trials[c]} for c in sorted(unstable)},
    }
    out = OUT_DIR / f"judge-stability-{run['label']}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"원본 총점 {report['original_total']}  재판정 총점 {totals}")
    print(f"같은 답인데 판정이 갈린 문항 {len(unstable)}/{len(records)}  축별 {dict(per_axis)}")
    print(f"기록: {out}")


if __name__ == "__main__":
    main()
