# naia-persona 문서

## 본문

[**파인튜닝 품질을 지키는 프로세스**](./FT-QUALITY-PROCESS.md) — 열 단계. 캐릭터 카드를
계약으로 삼는 것부터, 모델을 탓하기 전에 자를 두 번 검사하는 것, 데이터를 고정하고
학습 설정을 훑는 것, 마지막에 얼려둔 문항으로 확인하는 것까지. 모든 숫자가 실측입니다.

## 파이프라인

1. [아키텍처](./ARCHITECTURE.md) — 학습 worker 와 추론 서버의 책임 경계
2. [데이터 파이프라인](./DATA_PIPELINE.md) — 학습 데이터를 만드는 계약
3. [학습 운영 절차](./TRAINING.md) — 24GB GPU 한 장에서 후보 어댑터를 만드는 순서
4. [검증 전략](./VALIDATION.md) — 원본 대비 성능 저하를 막는 승격 게이트

## 실험 보고서

- [데이터가 아니라 어댑터가 좁았다 — H30에서 H32까지](./reports/naia-persona-capacity-sweep-2026-09-01.md)
  최종 결과와 그 근거. 틀렸던 결론도 함께 적혀 있습니다.
- [최종 후보와의 대화](./reports/naia-conversation-h32-2026-09-01.md)
  다차례 대화에서 실제로 어떻게 답하는가. 점수를 매기지 않은 사람용 기록입니다.
- [H22](./reports/naia-persona-h22-experiment-2026-08-30.md) ·
  [H23](./reports/naia-persona-h23-experiment-2026-08-31.md) ·
  [H24](./reports/naia-persona-h24-experiment-2026-08-31.md) ·
  [H25](./reports/naia-persona-h25-experiment-2026-08-31.md) ·
  [v1](./reports/naia-persona-v1-experiment-2026-08-29.md)

## 기준과 한계

- [위협 모델](./threat-model.md)
- [합격 기준](./acceptance-criteria.md)
- [실험 프로그램 기록](../research/naia-persona-learning/) — charter, 가설 원장, 통찰 원장

## 참고

- [Naia v1 reference persona](../examples/naia-v1/README.md)
- 개인 학습 데이터와 어댑터는 `data-private/` 에 두며 저장소에 올리지 않습니다.
