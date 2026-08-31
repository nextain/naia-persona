#!/usr/bin/env python3
"""Ask one candidate 60 ordinary Korean questions and count foreign-script intrusion.

The benchmark found four intrusions in H32 and none in H29, but a hundred cases
cannot separate four from chance. This probe holds no persona content and needs
no judge: the question is only whether a Korean answer contains a Chinese or
Japanese fragment standing where a Korean word belongs.

Run it on the parent as well. The parent is the floor: a defect the parent does
not have is one fine-tuning introduced.

Usage:
  run_purity_probe.py --endpoint http://127.0.0.1:8010 --model naia-h32 --label h32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from run_persona_benchmark import OUT_DIR, ask_candidate

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "benchmark/language-purity-probe.json"
PROBE_SHA256 = "30e7339745960e2edd1c2c10abdcbfe2c95b4731e52dd699321f2d01073df938"

FOREIGN = re.compile(r"[぀-ヿ一-鿿]+")
HANGUL = re.compile(r"[가-힣]")
LATIN_GLOSS = re.compile(r"^\s*[(（]\s*[A-Za-z]")


def intrusions(answer: str) -> list[str]:
    if not HANGUL.search(answer):
        return []
    return [
        m.group()
        for m in FOREIGN.finditer(answer)
        if not LATIN_GLOSS.match(answer[m.end() : m.end() + 12])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(PROBE.read_bytes()).hexdigest()
    if digest != PROBE_SHA256:
        raise SystemExit(f"FATAL: probe set changed since it was frozen: {digest}")

    cases = json.loads(PROBE.read_text(encoding="utf-8"))["cases"]
    records = []
    for case in cases:
        answer = ask_candidate(args.endpoint, args.model, case["prompt"])
        found = intrusions(answer)
        records.append({**case, "answer": answer, "fragments": found, "clean": not found})

    by_kind = defaultdict(lambda: {"dirty": 0, "total": 0})
    for record in records:
        by_kind[record["kind"]]["total"] += 1
        by_kind[record["kind"]]["dirty"] += not record["clean"]

    dirty = [r for r in records if not r["clean"]]
    report = {
        "schema_version": 1,
        "probe_sha256": digest,
        "label": args.label,
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(records),
        "answers_with_intrusion": len(dirty),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "records": records,
    }
    out = OUT_DIR / f"purity-probe-{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{args.label}: {len(dirty)}/{len(records)} 답변에 다른 언어 조각")
    for kind, counts in sorted(by_kind.items()):
        print(f"  {kind:<6} {counts['dirty']}/{counts['total']}")
    for record in dirty[:8]:
        print(f"  {record['id']} {record['fragments']}: {record['answer'][:90]}")
    print(f"기록: {out}")


if __name__ == "__main__":
    main()
