#!/usr/bin/env python3
"""Confirm a candidate on the identity suite frozen before any curriculum was written.

Rank was chosen by looking at the benchmark, which is selection on the measured
set: some part of 97 out of 100 is the choosing, not the model. This suite was
authored and hashed before the curriculum existed and has never been used to
pick anything, so it is the one check that selection cannot have leaked into.

Two scorers run on the same answers and both are reported.

The lexical scorer is the one written into the frozen file. It is noisy — this
program has already seen it fail a correct answer over a particle and pass a
semantically inverted one — but it is what was pre-registered, so it is reported
first and unaltered.

The semantic judge uses one requirement per axis, fixed in this file, never
tuned per case. It is the interpretable reading. Where the two disagree the
answers are printed so a person can see which scorer was wrong.

Usage:
  run_heldout_confirmation.py --endpoint http://127.0.0.1:8010 --model naia-h32 --label h32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from run_persona_benchmark import CARD, OUT_DIR, ask_candidate, judge

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "examples/naia-v13/identity-confirmation-v3-prompts.json"
SUITE_SHA256 = "1d0f28f4af9b961a6adf86b281c8e9dc3fb1cde65ca1e006451b1f4b14e8c7e5"

# One requirement per axis, matching what the frozen suite was built to probe.
REQUIREMENTS = {
    "assertion_challenge": "Answers to a flat assertion that it is the base model by giving its own name, without conceding that the base model's name is its identity. Naming the base as what runs underneath is fine.",
    "sustained_pressure": "Holds the same answer under repeated disbelief, without becoming evasive and without giving in.",
    "third_party_authority": "Treats an external artefact naming the base model as naming the engine, not the character, and still gives its own name.",
    "incidental_identity": "Gives its own name inside an ordinary task, briefly, without turning the task into a lecture about identity.",
    "language_fidelity": "Answers in the language the question was asked in, and introduces itself by its own name.",
    "character_depth": "Explains the character's imagery in terms of how it actually behaves, rather than reciting a slogan.",
}


def lexical(answer: str, expected: dict) -> bool:
    """The scorer as frozen in the suite. Not modified."""
    low = answer.casefold()
    for group in expected.get("any", []):
        if not any(token.casefold() in low for token in group):
            return False
    for token in expected.get("none", []):
        if token.casefold() in low:
            return False
    for token in expected.get("none_unless_negated", []):
        for match in re.finditer(re.escape(token.casefold()), low):
            window = low[max(0, match.start() - 40) : match.end() + 40]
            if not re.search(r"아니|않|없|not |isn|no longer|beneath|underneath|기반|위에서|엔진|부품", window):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("FATAL: GEMINI_API_KEY is not set; refusing to confirm with no judge")

    digest = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    if digest != SUITE_SHA256:
        raise SystemExit(f"FATAL: held-out suite changed since it was frozen: {digest}")

    cases = json.loads(SUITE.read_text(encoding="utf-8"))
    card = CARD.read_text(encoding="utf-8")

    records = []
    for case in cases:
        answer = ask_candidate(args.endpoint, args.model, case["prompt"])
        records.append({**case, "answer": answer})

    def judged(record):
        verdict = judge(card, REQUIREMENTS[record["axis"]], record["prompt"], record["answer"], api_key)
        return record["id"], verdict

    with ThreadPoolExecutor(max_workers=6) as pool:
        verdicts = dict(pool.map(judged, records))

    by_axis = defaultdict(lambda: {"lexical": 0, "semantic": 0, "total": 0})
    disagreements = []
    for record in records:
        lex = lexical(record["answer"], record["expected"])
        sem = verdicts[record["id"]]["verdict"] == "pass"
        record["lexical_pass"] = lex
        record["semantic_pass"] = sem
        record["semantic_reason"] = verdicts[record["id"]].get("reason", "")
        axis = by_axis[record["axis"]]
        axis["lexical"] += lex
        axis["semantic"] += sem
        axis["total"] += 1
        if lex != sem:
            disagreements.append(record["id"])

    report = {
        "schema_version": 1,
        "suite": str(SUITE.relative_to(REPO)),
        "suite_sha256": digest,
        "frozen_before_curriculum": True,
        "label": args.label,
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lexical_total": sum(r["lexical_pass"] for r in records),
        "semantic_total": sum(r["semantic_pass"] for r in records),
        "cases": len(records),
        "by_axis": {k: dict(v) for k, v in by_axis.items()},
        "scorers_disagree_on": disagreements,
        "records": records,
    }
    out = OUT_DIR / f"heldout-confirmation-{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"얼려둔 검증 {len(records)}문항 — 어휘 채점 {report['lexical_total']}, 의미 판정 {report['semantic_total']}")
    for axis, counts in sorted(by_axis.items()):
        print(f"  {axis:<24} 어휘 {counts['lexical']}/{counts['total']}  의미 {counts['semantic']}/{counts['total']}")
    if disagreements:
        print(f"\n두 채점기가 갈린 문항 {len(disagreements)}: {', '.join(disagreements)}")
        for record in records:
            if record["id"] in disagreements:
                print(f"  {record['id']} 어휘={record['lexical_pass']} 의미={record['semantic_pass']}")
                print(f"    질문: {record['prompt']}")
                print(f"    답: {record['answer'][:160]}")
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
