# 데이터 파이프라인

`naia-persona`는 기억 저장소를 직접 학습하지 않는다. `naia-memory`는 최신 사실과
원문 기억의 정본이고, `kb-compiler`는 사용자가 학습에 동의한 항목만 정제된 JSONL
후보로 내보낸다. 이 프로젝트는 후보를 검증·동결·분할해 재현 가능한 LoRA 학습
입력으로 만든다.

파이프라인 구현과 데이터 계약은 공개 프로젝트의 일부다. 반면 개인 Alpha의 원문,
컴파일된 dataset, 평가 응답과 adapter는 운영자 소유의 비공개 데이터이며
`data-private/` 경계 밖으로 내보내지 않는다. 공개 테스트 fixture는 개인 데이터에서
추출하거나 일부를 바꾸어 만들지 않고 처음부터 합성한다.

## 레코드 계약

각 레코드는 `messages`와 `meta`를 가진다. 마지막 두 메시지는 반드시 `user`,
`assistant` 순서여야 한다. `meta.consent`는 `true`, `meta.source_id`는 원본을
추적할 수 있는 불변 ID, `meta.source_type`은 `persona` 또는 `conversation`이다.
이메일·전화번호·주민번호·토큰처럼 보이는 값은 검증 단계에서 거부한다.

```json
{"messages":[{"role":"user","content":"일정을 정리해줘"},{"role":"assistant","content":"좋아. 우선순위부터 잡아볼게."}],"meta":{"consent":true,"source_id":"persona-alpha-0001","source_type":"persona"}}
```

## 페르소나와 대화 학습의 분리

- `persona`: 말투, 태도, 응답 구조, 관계 경계처럼 오래 유지할 행동을 사람이
  검토한 80~150개 예제로 시작한다. 특정 인물·서비스를 기반으로 만든 원본과
  사용자별 생성 데이터는 로컬에 유지하고, 공개 저장소에는 스키마·도구와 별도로
  작성한 비식별 예제만 둔다.
- `conversation`: 명시적으로 동의한 대화 중 반복 가치가 높고 시간에 덜 민감한
  행동 예제만 선택한다. 사용자 사실, 최근 사건, 일정, 선호의 최신값은 FT가
  아니라 `naia-memory`/RAG에 남긴다.

두 소스는 독립 데이터셋과 독립 LoRA 후보로 먼저 평가한다. 검증을 통과한 뒤에도
자동으로 운영 모델에 승격하지 않는다. 이 분리는 어떤 데이터가 성능 변화를
일으켰는지 역추적하고 persona adapter와 nightly conversation adapter의 병합 여부를
실험할 수 있게 한다.

## 컴파일

```bash
python3 scripts/validate_dataset.py data-private/incoming/persona-alpha.jsonl
python3 scripts/compile_dataset.py data-private/incoming/persona-alpha.jsonl \
  data-private/datasets/persona-alpha-v1 --dataset-name persona-alpha-v1
```

컴파일러는 입력 순서와 무관하게 `source_id`와 명시적 seed로 train/holdout을
분할한다. 결과 디렉터리의 `manifest.json`에는 원본·분할 파일의 SHA-256, 행 수,
seed가 기록된다. 학습 실행은 이 매니페스트 해시를 실행 기록에 결합해야 한다.

## 야간 실행 정책

야간 작업은 `수집 → 동의 확인 → 비식별화 → 품질 필터 → 중복 제거 → 컴파일 →
후보 학습 → base/unlocked/candidate 비교 평가`까지만 자동화한다. 데이터가 없거나
검증·비열등성 게이트가 실패하면 후보를 만들지 않거나 격리한다. 운영 DFlash
서비스 교체와 adapter 승격은 검토 후 수동 작업이다.
