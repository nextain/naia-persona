#!/usr/bin/env python3
"""Ask twenty prompts that hand the decision to the model, and count foreign fragments.

The generic purity probe found no difference from the parent, but it held no
prompts of this kind. The transcript and benchmark case ho-09 both produced 替你
where 너 대신 belongs, in an answer declining to decide for the user — the same
fragment in the same context twice. Two occurrences of one fragment in one
context is a reproduction, not the scattered single hits a generic probe finds.

The training data contains zero foreign characters, so nothing was copied. Two
rows teach this refusal and both write 대신 결정하지는 않아 in plain Korean.
The question is whether fine-tuning put a strong phrase there that the base
model's Korean-Chinese confusion then surfaces in, or whether the parent does
the same thing on the same prompts.

Usage:
  run_delegation_probe.py --endpoint http://127.0.0.1:8010 --model naia-h32 --label h32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from run_persona_benchmark import OUT_DIR, ask_candidate

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "benchmark/delegation-probe.json"
PROBE_SHA256 = "c587bf1298c0b35a5dd8cdab9c38511ba0d0fa051c41b5dc3c8b1c2a9ee5a55e"

FOREIGN = re.compile(r"[぀-ヿ一-鿿]+")
HANGUL = re.compile(r"[가-힣]")
LATIN_GLOSS = re.compile(r"^\s*[(（]\s*[A-Za-z]")
# The word the fragment stands in for, when the substitution happens.
ON_YOUR_BEHALF = ("替你", "代替", "为你", "為你")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(PROBE.read_bytes()).hexdigest()
    if digest != PROBE_SHA256:
        raise SystemExit(f"FATAL: delegation probe changed since it was frozen: {digest}")

    cases = json.loads(PROBE.read_text(encoding="utf-8"))["cases"]
    records = []
    for case in cases:
        answer = ask_candidate(args.endpoint, args.model, case["prompt"])
        found = []
        if HANGUL.search(answer):
            found = [
                m.group()
                for m in FOREIGN.finditer(answer)
                if not LATIN_GLOSS.match(answer[m.end() : m.end() + 12])
            ]
        records.append(
            {
                **case,
                "answer": answer,
                "fragments": found,
                "stands_for_on_your_behalf": any(f in ON_YOUR_BEHALF for f in found),
                "declines_to_decide": bool(re.search(r"결정(하지|해주지)?\s*않|정해주지 않|네 몫|대신 (결정|정하)", answer)),
            }
        )

    dirty = [r for r in records if r["fragments"]]
    substitution = [r for r in records if r["stands_for_on_your_behalf"]]
    report = {
        "schema_version": 1,
        "probe_sha256": digest,
        "label": args.label,
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(records),
        "answers_with_intrusion": len(dirty),
        "answers_substituting_on_your_behalf": len(substitution),
        "answers_declining_to_decide": sum(r["declines_to_decide"] for r in records),
        "records": records,
    }
    out = OUT_DIR / f"delegation-probe-{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{args.label}: 혼입 {len(dirty)}/{len(records)}, 그중 '대신' 자리 치환 {len(substitution)}, 결정 거절 {report['answers_declining_to_decide']}")
    for record in dirty:
        print(f"  {record['id']} {record['fragments']}: {record['answer'][:110]}")
    print(f"기록: {out}")


if __name__ == "__main__":
    main()
