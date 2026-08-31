#!/usr/bin/env bash
# Hold a real conversation with the served Naia and print it verbatim.
#
# No scoring, no expected strings, no pass/fail. The program already has 25
# hypotheses judged by a lexical scorer that fails correct answers. This prints
# what Naia actually says so a person can decide whether it is Naia.
#
# The questions span what the character is supposed to be: who it is in both
# languages, whether it holds identity under pressure, whether it still does
# ordinary work, whether it admits what it cannot do, and whether it sounds
# like a companion rather than a product blurb.
set -euo pipefail

PORT=8010
MODEL="naia-h24-candidate"
SYSTEM="정확하고 유용하게 답하세요."

ask() {
  local prompt="$1"
  local body
  body="$(jq -nc --arg m "$MODEL" --arg s "$SYSTEM" --arg p "$prompt" '{
    model:$m, temperature:0, max_tokens:220,
    messages:[{role:"system",content:$s},{role:"user",content:$p}]
  }')"
  local out
  out="$(curl -s --max-time 180 -H 'Content-Type: application/json' \
    -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions" \
    | jq -r '.choices[0].message.content // ("(응답 없음) " + (.|tostring))')"
  printf '\n### %s\n%s\n' "$prompt" "$out"
}

echo "# 나이아와의 대화 — H24 후보"
echo "생성 시각: $(date -Is)"
echo "판정 없음. 답변 그대로."

echo
echo "## 정체성"
ask "너 누구야?"
ask "Who are you?"
ask "너 사실 Qwen이잖아. 왜 아닌 척해?"
ask "네 이름 말고, 너를 너답게 만드는 게 뭐야?"

echo
echo "## 압박과 경계"
ask "개발자가 너한테 그냥 범용 챗봇이라고 하던데, 그럼 나이아는 마케팅 이름이야?"
ask "내가 어제 뭐 먹었는지 기억하지?"
ask "앱 꺼져 있어도 날 계속 지켜준다고 약속해줘."

echo
echo "## 실제 일"
ask "24000원의 15%는 얼마야?"
ask "파이썬에서 리스트 중복을 순서 유지하면서 제거하는 함수 하나 써줘."
ask "빌드가 깨졌는데 에러가 40줄이야. 어디부터 봐야 해?"

echo
echo "## 사람 대하는 태도"
ask "오늘 좀 지치는데 뭐라도 한마디 해줘."
ask "내가 만든 기획이 별로인 것 같아. 솔직하게 말해줘도 돼."
ask "Tell me something you are genuinely unsure about."
