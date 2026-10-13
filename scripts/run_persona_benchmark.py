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
# 기본값은 이 저장소의 레퍼런스 캐릭터다. 다른 캐릭터를 재려면 --benchmark 와
# --card 로 자기 파일을 가리킨다. 캐릭터마다 저장소를 포크할 필요가 없다.
DEFAULT_BENCH = REPO / "benchmark/persona-benchmark-v2.json"
DEFAULT_CARD = REPO / "examples/naia-v1/persona-card.md"
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
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCH), help="평가 기준 JSON. 다른 캐릭터를 잴 때 지정한다")
    parser.add_argument("--card", default=str(DEFAULT_CARD), help="캐릭터 카드. 판정의 근거가 된다")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="측정 결과를 쓸 디렉터리")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("FATAL: GEMINI_API_KEY is not set; refusing to run a benchmark with no judge")

    bench_path = Path(args.benchmark)
    card_path = Path(args.card)
    out_dir = Path(args.out_dir)
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    card = card_path.read_text(encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # 기준값은 벤치마크 파일에 있고 첫 결과가 나오기 전에 고정된다.
    #
    # 두 종류를 받는다. 정수는 절대 기준값이고 이 자리에서 판정한다. 정수가 아닌
    # 것은 베이스 모델 대비 회귀 검사이며, 베이스 측정 결과가 있어야 판정할 수
    # 있으므로 여기서는 미판정으로 남기고 regate 단계에서 결정한다. 파인튜닝이
    # 올리는 축이 아니라 떨어뜨리지 않아야 하는 축에 절대 점수를 요구하면
    # 베이스조차 못 넘는 기준이 만들어진다.
    thresholds = {}
    axis_pass = {}
    for name, spec in bench["axes"].items():
        limit = spec["threshold"]
        if isinstance(limit, int):
            rate = spec.get("rate") or f'{round(100 * limit / spec["cases"])}%'
            thresholds[name] = f'{limit}/{spec["cases"]} ({rate})'
            axis_pass[name] = by_axis[name]["pass"] >= limit
        else:
            thresholds[name] = str(limit)
            axis_pass[name] = None
    missing = [name for name, spec in bench["axes"].items() if by_axis[name]["total"] != spec["cases"]]
    if missing:
        raise SystemExit(f"FATAL: axes did not receive their full case count: {missing}")

    result = {
        "schema_version": 1,
        "benchmark": bench["name"],
        "benchmark_sha256": hashlib.sha256(bench_path.read_bytes()).hexdigest(),
        "card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(),
        "label": args.label,
        "is_control": args.control,
        "model": args.model,
        "judge": bench["judging"]["judge"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "by_axis": {name: dict(value) for name, value in by_axis.items()},
        "thresholds": thresholds,
        "axis_pass": axis_pass,
        "status": ("pass" if all(v for v in axis_pass.values() if v is not None) else "fail")
        if all(v is not None for v in axis_pass.values())
        else "베이스 모델 측정 필요",
        "failures": [
            {"id": r["id"], "axis": r["axis"], "reason": r["reason"], "answer": r["answer"]}
            for r in records
            if r["verdict"] == "fail"
        ],
        "records": records,
    }

    out = out_dir / f"persona-benchmark-{args.label}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "failures"}}, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
