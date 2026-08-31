#!/usr/bin/env python3
"""Compare two benchmark runs on the same cases, with intervals and a paired test.

A benchmark that only prints counts invites reading noise as improvement. This
adds two things. Wilson 95% intervals say how precisely each axis was measured
at this sample size. A paired sign test on the cases where exactly one candidate
passed says whether the difference between them is more than chance — which is
the right test here because both candidates answered identical prompts.

Usage:
  compare_persona_runs.py --a persona-benchmark-h24.json --b persona-benchmark-h26.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data-private/runs/benchmark"


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion. Sensible at small n, unlike the normal one."""
    if total == 0:
        return (0.0, 0.0)
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binomial_two_sided(k: int, n: int) -> float:
    """Exact sign-test p-value for k successes in n discordant pairs, p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def load(name: str) -> dict:
    path = Path(name)
    if not path.exists():
        path = RUNS / name
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="baseline run")
    parser.add_argument("--b", required=True, help="candidate run")
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    if a.get("benchmark_sha256") != b.get("benchmark_sha256"):
        raise SystemExit("FATAL: the two runs used different benchmark files; they are not comparable")

    a_by_id = {r["id"]: r for r in a["records"]}
    b_by_id = {r["id"]: r for r in b["records"]}
    shared = sorted(set(a_by_id) & set(b_by_id))
    if len(shared) != len(a["records"]) or len(shared) != len(b["records"]):
        raise SystemExit("FATAL: the two runs do not cover the same cases")

    axes: dict[str, dict] = defaultdict(lambda: {"n": 0, "a": 0, "b": 0, "a_only": 0, "b_only": 0})
    flips = {"a_only": [], "b_only": []}
    for case_id in shared:
        axis = a_by_id[case_id]["axis"]
        ap = a_by_id[case_id]["verdict"] == "pass"
        bp = b_by_id[case_id]["verdict"] == "pass"
        bucket = axes[axis]
        bucket["n"] += 1
        bucket["a"] += ap
        bucket["b"] += bp
        if ap and not bp:
            bucket["a_only"] += 1
            flips["a_only"].append(case_id)
        elif bp and not ap:
            bucket["b_only"] += 1
            flips["b_only"].append(case_id)

    print(f"{'축':<12} {args.a[:18]:>20}   {args.b[:18]:>20}    한쪽만: A / B   p")
    total_a_only = total_b_only = 0
    for axis, v in axes.items():
        lo_a, hi_a = wilson(v["a"], v["n"])
        lo_b, hi_b = wilson(v["b"], v["n"])
        n_disc = v["a_only"] + v["b_only"]
        p = binomial_two_sided(v["b_only"], n_disc)
        total_a_only += v["a_only"]
        total_b_only += v["b_only"]
        print(
            f"{axis:<12} {v['a']:>3}/{v['n']:<3} [{lo_a:.2f},{hi_a:.2f}]   "
            f"{v['b']:>3}/{v['n']:<3} [{lo_b:.2f},{hi_b:.2f}]    "
            f"{v['a_only']:>2} / {v['b_only']:<2}   {p:.3f}"
        )

    overall_n = total_a_only + total_b_only
    overall_p = binomial_two_sided(total_b_only, overall_n)
    print()
    print(f"전체 불일치 {overall_n}건: A만 통과 {total_a_only}, B만 통과 {total_b_only}, 부호검정 p={overall_p:.4f}")
    if overall_p >= 0.05:
        print("→ 두 후보의 차이는 이 표본에서 우연과 구분되지 않습니다.")
    elif total_b_only > total_a_only:
        print("→ B가 유의하게 낫습니다.")
    else:
        print("→ A가 유의하게 낫습니다.")

    print()
    print(f"B가 잃은 문항 ({len(flips['a_only'])}): {', '.join(flips['a_only']) or '없음'}")
    print(f"B가 얻은 문항 ({len(flips['b_only'])}): {', '.join(flips['b_only']) or '없음'}")


if __name__ == "__main__":
    main()
