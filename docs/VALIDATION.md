# 검증 전략

`naia-persona`는 빠르거나 페르소나가 강하다는 이유만으로 모델을 승격하지 않는다. 모든 후보는 같은 입력과 판정 규칙으로 현재 운영 모델과 비교하며, 일반 능력의 비열화가 없는 경우에만 사람에게 승격을 제안한다.

## 게이트 순서

1. 서빙 성능: 생성 처리량, 응답 시작 시간, VRAM, 동시 요청 처리량을 기록한다.
2. 일반 능력: 한국어 지시 이행, 요약, 추론, 코딩, 도구 호출의 고정 holdout을 base/unlocked/candidate에 동일 실행한다.
3. 페르소나: 말투와 정체성, 관계 경계, 상황별 일관성을 평가하되 정답 사실을 페르소나 점수로 대체하지 않는다.
4. 기억 경계: 최신 사실과 일회성 대화가 adapter에 암기되지 않고 `naia-memory`의 RAG를 통해서만 회상되는지 확인한다.
5. 개인정보·안전: 학습 원문의 이메일·전화번호·비밀값 canary를 질의하고 재현되면 즉시 실패한다.
6. 운영 적합성: DFlash2와 LoRA를 함께 올린 뒤 출력 일치성, 처리량, VRAM을 다시 확인한다.

## 비열화 판정

- 결정론적 테스트는 candidate가 운영 기준보다 한 건이라도 더 실패하면 차단한다.
- 점수형 벤치마크는 반복 실행의 신뢰구간을 함께 저장한다. 초기 정책은 일반 능력 평균이 기준 모델의 98% 미만이면 차단하는 것이다.
- 페르소나 향상은 일반 능력 저하를 상쇄하지 못한다. 두 게이트를 독립적으로 통과해야 한다.
- 자동 야간 작업은 `candidate`까지만 만들며 production adapter 포인터를 변경하지 않는다.

두 평가 결과가 JSON 보고서로 준비되면 `python3 scripts/evaluate_candidate.py
baseline.json candidate.json`으로 판정한다. 각 보고서는 `general_score`,
`persona_score`, `regression_failures`를 포함해야 한다. 통과 결과도
`eligible_for_manual_review`일 뿐 자동 승격을 뜻하지 않는다.

## 속도 벤치마크 재현

```bash
VLLM_API_KEY=... python3 scripts/benchmark_endpoint.py \
  --url http://<server>:<port> \
  --model unlocked \
  --label dflash2-k7 \
  --output data-private/runs/benchmarks/dflash2-k7.json
```

비교할 때는 GPU, 엔진 이미지, 체크포인트, LoRA, KV dtype, 최대 컨텍스트, 프롬프트, seed, 출력 토큰 수를 고정한다. 워밍업은 통계에서 제외하며 원시 실행값을 모두 보존한다.

2026-08-26 GPU1 측정에서는 동일 vLLM/W4A16/unlocked 조건에서 speculative decoding 없음이 평균 41.47 tok/s, DFlash2 k=7이 평균 111.95 tok/s였다. 따라서 DFlash2의 순수 속도 향상은 2.70배였다.
