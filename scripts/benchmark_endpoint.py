#!/usr/bin/env python3
"""Reproducible single-stream benchmark for an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPT = (
    "한국어로 인공지능 개인 비서가 장기 기억, 지식 검색, 음성 인터페이스를 "
    "결합할 때의 장점과 개인정보 보호 원칙을 자연스러운 설명문으로 작성해줘. "
    "제목이나 목록 없이 충분히 길게 설명해줘."
)


def request_once(url: str, api_key: str | None, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    tokens = int(body["usage"]["completion_tokens"])
    return {"elapsed_seconds": elapsed, "completion_tokens": tokens, "tokens_per_second": tokens / elapsed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="unlocked")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0 or args.max_tokens < 1:
        parser.error("runs/max-tokens must be positive and warmups must be non-negative")

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": args.max_tokens,
    }
    api_key = os.getenv(args.api_key_env)
    warmups = [request_once(args.url, api_key, payload) for _ in range(args.warmups)]
    runs = [request_once(args.url, api_key, payload) for _ in range(args.runs)]
    rates = [run["tokens_per_second"] for run in runs]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "endpoint": args.url,
        "model": args.model,
        "request": {"prompt": args.prompt, "temperature": 0, "seed": 42, "max_tokens": args.max_tokens},
        "warmups": warmups,
        "runs": runs,
        "summary": {
            "mean_tokens_per_second": statistics.mean(rates),
            "median_tokens_per_second": statistics.median(rates),
            "stdev_tokens_per_second": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
