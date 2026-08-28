# naia-persona 문서

이 디렉터리는 개인화 모델을 만드는 방법과 실제 검증 근거의 정본입니다.

## 처음 읽는 순서

1. [아키텍처](./ARCHITECTURE.md) — `naia-memory`, `kb-compiler`, 학습 worker와 추론 서버의 책임 경계
2. [데이터 파이프라인](./DATA_PIPELINE.md) — persona/대화 데이터를 만드는 계약
3. [학습 운영 절차](./TRAINING.md) — 24GB GPU1에서 후보 adapter를 만드는 순서
4. [검증 전략](./VALIDATION.md) — 기존 모델 대비 성능 저하를 막는 승격 게이트

## 현재 실험 근거

- [Alpha persona 파이프라인 실험 — 2026-08-26](./reports/alpha-persona-pipeline-experiment-2026-08-26.md)
  - DFlash2: 41.47 → 111.95 tok/s
  - RTX 3090 24GB QLoRA 완료
  - persona 향상 부재로 후보 승격 차단

보고서의 8-prompt 평가는 파이프라인 sanity check이며 제품 성능을 대표하는 벤치마크가 아닙니다. 다음 실험은 completion masking 수정, 더 큰 holdout과 반복 측정을 먼저 적용해야 합니다.

## 기반 저장소 문서

아래 문서는 기반 ADK 저장소의 개발·거버넌스 자료입니다. 모델 개인화 사용자는 위 네 문서와 실험 보고서부터 읽으면 됩니다.

- [프로젝트 구조](./project-structure.md)
- [위협 모델](./threat-model.md)
- [합격 기준](./acceptance-criteria.md)
- [LLM 역할 분담](./llm-roles.md)
- [작업 기록](./progress/README.md)
