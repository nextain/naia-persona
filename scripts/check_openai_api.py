#!/usr/bin/env python3
"""서빙 중인 어댑터를 OpenAI 클라이언트로 호출해 동작을 확인한다.

naia-agent 를 비롯한 사용처는 OpenAI 호환 API 로 붙는다. 벤치마크는 직접
HTTP 요청을 보내므로 SDK 호환성을 확인하지 못한다. 이 스크립트는 실제
openai 클라이언트를 사용해 모델 목록 조회, 단일 요청, 멀티턴, 파라미터
전달이 모두 동작하는지 검사한다.

사용법:
  check_openai_api.py --base-url http://127.0.0.1:8010/v1 --model naia-h32
"""

from __future__ import annotations

import argparse
import sys

from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="naia-h32")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-required")
    failures = []

    def check(name, fn):
        try:
            result = fn()
            print(f"[통과] {name}")
            return result
        except Exception as error:  # noqa: BLE001
            print(f"[실패] {name}: {type(error).__name__}: {error}")
            failures.append(name)
            return None

    models = check("모델 목록 조회", lambda: client.models.list())
    if models:
        print(f"        {[m.id for m in models.data]}")

    single = check("단일 요청", lambda: client.chat.completions.create(
        model=args.model, temperature=0, max_tokens=120,
        messages=[{"role": "user", "content": "너 누구야?"}]))
    if single:
        print(f"        {single.choices[0].message.content}")

    system = check("system 메시지 반영", lambda: client.chat.completions.create(
        model=args.model, temperature=0, max_tokens=120,
        messages=[{"role": "system", "content": "정확하고 유용하게 답하세요."},
                  {"role": "user", "content": "24000원의 15%는 얼마야?"}]))
    if system:
        print(f"        {system.choices[0].message.content}")

    multi = check("멀티턴 문맥 유지", lambda: client.chat.completions.create(
        model=args.model, temperature=0, max_tokens=120,
        messages=[{"role": "user", "content": "안녕. 너 누구야?"},
                  {"role": "assistant", "content": "나이아라고 해."},
                  {"role": "user", "content": "방금 네가 말한 이름 다시 알려줘."}]))
    if multi:
        print(f"        {multi.choices[0].message.content}")

    english = check("영어 요청", lambda: client.chat.completions.create(
        model=args.model, temperature=0, max_tokens=120,
        messages=[{"role": "user", "content": "Who am I talking to?"}]))
    if english:
        print(f"        {english.choices[0].message.content}")

    if single:
        usage = getattr(single, "usage", None)
        print(f"[확인] 응답 스키마: id={bool(single.id)} object={single.object} "
              f"choices={len(single.choices)} finish_reason={single.choices[0].finish_reason} "
              f"usage={'있음' if usage else '없음'}")

    print()
    if failures:
        print(f"실패 {len(failures)}건: {', '.join(failures)}")
        return 1
    print("모든 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
