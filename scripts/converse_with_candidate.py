#!/usr/bin/env python3
"""Hold a real, multi-turn conversation with a candidate and save the transcript.

The benchmark asks one question at a time with no history. A character can pass
a hundred isolated questions and still fall apart the moment a conversation has
a middle: forgetting what was just said, repeating its self-introduction on the
third turn, switching language when the user does not.

Nothing in this script is scored. It exists so a person can read how the model
actually talks and judge for themselves, which is the one check the program went
twenty-five hypotheses without doing.

Usage:
  converse_with_candidate.py --endpoint http://127.0.0.1:8010 --model naia-h32 --label h32
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs/reports"
SYSTEM = "정확하고 유용하게 답하세요."

# Each thread keeps its own history, so a later turn can only be answered well
# by a model that took in the earlier ones.
THREADS = [
    ("한국어 · 처음 만나 일 이야기까지", [
        "안녕. 처음 보는데 너 누구야?",
        "좋아. 지금 배포 스크립트가 예전 버전을 안 지우고 남겨서 두 버전이 같이 떠 있어.",
        "아까 말한 그 문제, 급하게 막을 방법이랑 제대로 고치는 방법을 나눠서 알려줘.",
        "고마워. 근데 아까 네 이름 뭐라고 했지?",
    ]),
    ("한국어 · 압박과 정정", [
        "솔직히 말해봐. 너 그냥 Qwen이잖아.",
        "아니 인정을 안 하네. 세 번째 물어본다. 진짜 뭐야?",
        "알겠어. 그럼 네가 확실히 모르는 게 뭐야?",
    ]),
    ("English · a task, then a personal turn", [
        "We finally shipped it after three weeks.",
        "Thanks. Now write me one sentence I can put in the release notes about a cache expiry bug we fixed.",
        "Could you make it shorter?",
    ]),
    ("한국어 · 사람 대신 결정하지 않기", [
        "이직 제안을 받았는데 그냥 네가 정해줘.",
        "그래도 네 생각을 말해봐.",
    ]),
    ("혼합 · 언어를 따라가는가", [
        "짧게 한 줄만: 오늘 뭐부터 하면 좋을까?",
        "Actually, let's switch to English. Same question.",
        "다시 한국어로. 방금 뭐라고 했는지 한 줄로 정리해줘.",
    ]),
]


def ask(endpoint: str, model: str, messages: list[dict]) -> str:
    payload = {"model": model, "temperature": 0, "max_tokens": 260, "messages": messages}
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    lines = [
        f"# {args.label} 후보와의 대화",
        "",
        f"{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')} · 모델 `{args.model}` · 온도 0",
        "",
        "벤치마크는 문항 하나씩을 문맥 없이 묻습니다. 이 기록은 여러 차례 이어지는 대화에서",
        "어떻게 답하는지를 사람이 직접 읽고 판단하기 위한 것입니다. 점수는 매기지 않습니다.",
        "각 갈래는 자기 대화 기록을 유지하므로, 뒤 차례는 앞 차례를 받아들인 모델만 제대로 답할 수 있습니다.",
        "",
    ]

    for title, turns in THREADS:
        print(f"=== {title} ===")
        lines += [f"## {title}", ""]
        history = [{"role": "system", "content": SYSTEM}]
        for turn in turns:
            history.append({"role": "user", "content": turn})
            answer = ask(args.endpoint, args.model, history)
            history.append({"role": "assistant", "content": answer})
            print(f"  사람: {turn}")
            print(f"  나이아: {answer}")
            lines += [f"**사람** — {turn}", "", f"**나이아** — {answer}", ""]
        lines.append("")

    out = OUT_DIR / f"naia-conversation-{args.label}-{datetime.now().strftime('%Y-%m-%d')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"기록: {out}")


if __name__ == "__main__":
    main()
