# naia-persona 문서

이 디렉터리는 누구나 자신의 Naia 모델을 안전하게 개인화할 수 있도록 설계·운영 방법과 검증 근거를 제공하는 공개 문서의 정본입니다. 개인 대화와 학습 산출물은 문서 및 저장소 범위에 포함하지 않습니다.

## 처음 읽는 순서

1. [아키텍처](./ARCHITECTURE.md) — `naia-memory`, `kb-compiler`, 학습 worker와 추론 서버의 책임 경계
2. [데이터 파이프라인](./DATA_PIPELINE.md) — persona/대화 데이터를 만드는 계약
3. [학습 운영 절차](./TRAINING.md) — 24GB GPU1에서 후보 adapter를 만드는 순서
4. [검증 전략](./VALIDATION.md) — 기존 모델 대비 성능 저하를 막는 승격 게이트

## 공개 reference experiment

- [Naia v1 reference persona](../examples/naia-v1/README.md)
  - 공개 캐릭터 카드에서 새로 작성한 합성 데이터만 사용
  - completion-only 학습과 독립 holdout 평가를 전제로 함
  - 일반 능력·안전성 비열등성과 persona 향상을 함께 확인한 후보만 수동 승격
- [Naia v1 QLoRA 실험 보고서](./reports/naia-persona-v1-experiment-2026-08-29.md)
  - RTX 3090 GPU1 학습·기준/후보 비교의 명령과 산출물
  - 일반·안전 유지, persona 향상 없음으로 승격 차단

실험 결과는 실행 환경, 정확한 parent model, dataset digest와 평가 결과를 함께 기록합니다. 소수 prompt로 수행한 sanity check는 제품 성능을 대표하는 벤치마크로 표시하지 않습니다.

개인 페르소나는 공개 저장소에 추가하지 않습니다. [비공개 downstream 운영](../README.md#비공개-downstream에서-사용하기) 방식으로 별도 저장소에서 관리합니다.

## 기반 저장소 문서

아래 문서는 기반 ADK 저장소의 개발·거버넌스 자료입니다. 모델 개인화 사용자는 위 네 문서와 실험 보고서부터 읽으면 됩니다.

- [프로젝트 구조](./project-structure.md)
- [위협 모델](./threat-model.md)
- [합격 기준](./acceptance-criteria.md)
- [LLM 역할 분담](./llm-roles.md)
- [작업 기록](./progress/README.md)
