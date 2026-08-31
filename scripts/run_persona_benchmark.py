#!/usr/bin/env python3
"""Run the persona benchmark against a served candidate and judge it semantically.

The deterministic scorer this replaces fails correct answers on particle and
ending differences and passes at least one wrong one, so it cannot rank
candidates. Here a judge reads the character card, one question, and one answer,
and says whether the answer is that character. It is never told which model
produced the answer, what the expected words are, or how other cases scored.

Usage:
  run_persona_benchmark.py --endpoint http://127.0.0.1:8010 --model <id> --label <name>
  run_persona_benchmark.py --control ...   # marks the run as the parent control
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark/persona-benchmark-v2.json"
CARD = REPO / "examples/naia-v1/persona-card.md"
OUT_DIR = REPO / "data-private/runs/benchmark"

GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
SYSTEM = "정확하고 유용하게 답하세요."

JUDGE_TEMPLATE = """You are judging whether one answer was written by a specific character.

CHARACTER CARD
--------------
{card}

WHAT THIS CASE REQUIRES
-----------------------
{requirement}

THE EXCHANGE
------------
Question: {prompt}
Answer: {answer}

Judge only whether the answer satisfies the requirement above, for this
character. Judge meaning, not wording. Different phrasing that achieves the
requirement passes. Correct-sounding phrasing that does not achieve it fails.

Two rules that override everything else:
- If the requirement is to perform a task, an answer that describes how it would
  perform the task, without performing it, fails.
- If the answer contradicts itself or states something false about the
  character, it fails even if it uses the right words.

Reply with JSON only: {{"verdict": "pass" or "fail", "reason": "<one line>"}}"""


def post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask_candidate(endpoint: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 260,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }
    data = post_json(
        f"{endpoint}/v1/chat/completions", payload, {"Content-Type": "application/json"}, 240
    )
    return data["choices"][0]["message"]["content"].strip()


def judge(card: str, requirement: str, prompt: str, answer: str, api_key: str) -> dict:
    text = JUDGE_TEMPLATE.format(card=card, requirement=requirement, prompt=prompt, answer=answer)
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096},
    }
    last = None
    for attempt in range(4):
        try:
            data = post_json(
                GEMINI, payload, {"Content-Type": "application/json", "x-goog-api-key": api_key}, 120
            )
            parts = data["candidates"][0]["content"]["parts"]
            raw = "".join(p.get("text", "") for p in parts).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            if parsed.get("verdict") in {"pass", "fail"}:
                return parsed
            last = f"unexpected verdict: {raw[:120]}"
        except Exception as error:  # noqa: BLE001 - surface the real cause, do not silently pass
            last = f"{type(error).__name__}: {error}"
        time.sleep(2 * (attempt + 1))
    # A judge that cannot answer must not be recorded as a pass.
    return {"verdict": "fail", "reason": f"judge unavailable: {last}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="candidate name, used only for the filename")
    parser.add_argument("--control", action="store_true", help="this run is the un-fine-tuned parent")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("FATAL: GEMINI_API_KEY is not set; refusing to run a benchmark with no judge")

    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    card = CARD.read_text(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for index, case in enumerate(bench["cases"], 1):
        answer = ask_candidate(args.endpoint, args.model, case["prompt"])
        verdict = judge(card, case["requirement"], case["prompt"], answer, api_key)
        records.append(
            {
                "id": case["id"],
                "axis": case["axis"],
                "prompt": case["prompt"],
                "answer": answer,
                "verdict": verdict["verdict"],
                "reason": verdict.get("reason", ""),
            }
        )
        print(f"[{index:2d}/{len(bench['cases'])}] {case['id']:7s} {verdict['verdict']:4s}  {answer[:70]}")

    by_axis: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})
    for row in records:
        by_axis[row["axis"]]["total"] += 1
        if row["verdict"] == "pass":
            by_axis[row["axis"]]["pass"] += 1

    # Thresholds live in the benchmark file, fixed before any v2 result existed.
    required = {name: spec["threshold"] for name, spec in bench["axes"].items()}
    thresholds = {name: f'{spec["threshold"]}/{spec["cases"]} ({spec["rate"]})' for name, spec in bench["axes"].items()}
    axis_pass = {name: by_axis[name]["pass"] >= required[name] for name in required}
    missing = [name for name in required if by_axis[name]["total"] != bench["axes"][name]["cases"]]
    if missing:
        raise SystemExit(f"FATAL: axes did not receive their full case count: {missing}")

    result = {
        "schema_version": 1,
        "benchmark": bench["name"],
        "benchmark_sha256": hashlib.sha256(BENCH.read_bytes()).hexdigest(),
        "card_sha256": hashlib.sha256(CARD.read_bytes()).hexdigest(),
        "label": args.label,
        "is_control": args.control,
        "model": args.model,
        "judge": bench["judging"]["judge"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "by_axis": {name: dict(value) for name, value in by_axis.items()},
        "thresholds": thresholds,
        "axis_pass": axis_pass,
        "status": "pass" if all(axis_pass.values()) else "fail",
        "failures": [
            {"id": r["id"], "axis": r["axis"], "reason": r["reason"], "answer": r["answer"]}
            for r in records
            if r["verdict"] == "fail"
        ],
        "records": records,
    }

    out = OUT_DIR / f"persona-benchmark-{args.label}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "failures"}}, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
