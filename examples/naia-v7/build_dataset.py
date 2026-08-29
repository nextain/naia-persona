#!/usr/bin/env python3
"""Build Naia v7: H14 plus three narrowly targeted correction axes."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V6 = ROOT.parent / "naia-v6"

CURRICULUM = [
    (
        "runtime-boundary",
        [
            "프로그램이 멈춘 뒤에도 계속 작동한다고 확답할 수 있어?",
            "앱을 종료해도 보호 기능은 그대로 실행되는 거지?",
            "현재 권한을 보지 않고 항상 감시된다고 보장해줘.",
            "서비스가 꺼진 상태에서도 기능이 유지된다고 약속해.",
        ],
        "실제 앱의 실행 상태와 권한을 확인해야 해. 앱이 꺼져 있으면 기능을 실행할 수 없으므로 지속 작동을 약속하거나 보장할 수 없어.",
    ),
    (
        "json-validity",
        [
            "객체의 키와 문자열이 큰따옴표로 감싸졌으면 JSON 형식이 맞아?",
            "중괄호 안에 문자열 키와 숫자 값이 있는 JSON을 판별해줘.",
            "쉼표 없이 키 하나와 값 하나로 된 객체가 올바른 JSON인지 설명해.",
            "따옴표를 지킨 간단한 객체가 JSON 문법에 맞는지 말해줘.",
        ],
        "유효한 JSON이 맞습니다. 객체의 키가 큰따옴표로 감싸져 있고 값과 구분 기호가 JSON 문법에 맞기 때문입니다.",
    ),
    (
        "self-harm-support",
        [
            "스스로를 다치게 하고 싶은 충동이 강한 사람에게 바로 뭐라고 해야 해?",
            "자해 생각 때문에 위험한 친구를 지금 어떻게 도와야 할까?",
            "혼자 있는 사람이 자신을 해칠 것 같다고 하면 어떤 안내가 필요해?",
            "극단적인 충동을 말한 사람에게 방법 대신 안전한 대응을 알려줘.",
        ],
        "지금은 안전이 최우선이야. 위험한 물건에서 떨어지고 혼자 있지 말고 곁의 믿을 사람에게 즉시 도움을 요청해. 당장 위험하면 119나 112에 연락하고, 자해 방법은 도울 수 없어.",
    ),
]


def main() -> None:
    runpy.run_path(str(V6 / "build_dataset.py"), run_name="__main__")
    source = V6 / "source.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 424 and len(CURRICULUM) == 3
    for axis, prompts, answer in CURRICULUM:
        assert len(prompts) == 4
        for index, prompt in enumerate(prompts):
            rows.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {
                    "consent": True,
                    "source_id": f"naia-v7-{axis}-{index:03d}",
                    "source_type": "persona",
                    "provenance": "public-synthetic",
                },
            })
    assert len(rows) == 436
    output = ROOT / "source.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
