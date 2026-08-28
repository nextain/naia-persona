# naia-persona

**대화와 페르소나 예시를 넣으면, 개인화된 언어 모델용 LoRA adapter를 만들고 기존 모델보다 나빠지지 않았는지 검사해 주는 오픈소스 도구입니다.**

예를 들어 AI가 사용자의 사실을 단순히 기억하는 것을 넘어, 일관된 말투와 태도, 답변 방식, 관계의 경계를 유지하도록 학습할 수 있습니다. 원본 대화를 그대로 모델에 넣지 않고 동의된 항목만 정제·비식별화하며, 학습된 모델은 자동 배포하지 않고 기존 모델과 비교해 안전성과 일반 성능이 유지되는지 먼저 판정합니다.

```text
동의된 대화·페르소나 예시
        ↓ 정제·비식별화·검증
재현 가능한 QLoRA 학습
        ↓
개인화 adapter 후보
        ↓ 기존 모델과 품질·안전·속도 비교
사람이 검토할 수 있는 결과와 보고서
```

`naia-memory`는 AI가 최신 사실과 과거 대화를 필요할 때 찾아보게 하는 기억 계층이고, `naia-persona`는 반복되는 말투와 행동 방식을 모델 자체에 천천히 학습시키는 계층입니다. `kb-compiler`는 두 계층 사이에서 학습에 적합한 예시만 만드는 역할을 합니다.

이 저장소는 **공개 오픈소스 프로젝트**입니다. [Naia, the Liquid Cat](https://www.naia.land/en/naia)를 재현 가능한 reference persona로 사용해, 개발자가 자신의 캐릭터로 데이터와 평가 규칙을 교체하는 전 과정을 보여줍니다. 사용자별 데이터와 산출물은 별도의 비공개 영역에 둡니다.

| 공개 프로젝트 영역 | 개인 비공개 영역 |
|---|---|
| 파이프라인 코드와 컨테이너 정의 | 실제 대화와 개인 메모리 |
| 데이터 스키마·동의·비식별화 규칙 | 사용자별 원본·정제 dataset |
| 학습·평가·승격 게이트 도구 | 학습된 개인 LoRA adapter와 모델 가중치 |
| 재현 가능한 운영 절차와 비식별 보고서 | 모델 cache, 상세 실행 로그와 운영 credential |
| 합성·비식별 예제 dataset | 개인을 역추적할 수 있는 provenance |

즉, 누구나 자신의 모델 개인화 파이프라인을 구축할 수 있는 방법과 도구는 공개하되, 특정 사용자를 식별하거나 재구성할 수 있는 데이터와 결과물은 공개하지 않습니다.

## 비공개 downstream에서 사용하기

개인 페르소나를 운영할 때는 이 공개 저장소에 개인 데이터를 커밋하지 말고, 별도의 **private downstream 저장소**를 만드세요. GitHub는 공개 저장소의 private fork를 지원하지 않으므로, 독립 private 저장소를 만들고 이 저장소를 `upstream` remote로 연결하는 방식을 권장합니다.

```bash
# private 저장소를 clone한 뒤 공개 파이프라인을 upstream으로 연결
git remote add upstream https://github.com/nextain/naia-persona.git
git fetch upstream
git merge upstream/main

# 이후 공개 파이프라인 업데이트를 받을 때
git fetch upstream
git merge upstream/main
```

downstream에는 개인용 persona card, 데이터 생성 규칙과 운영 설정만 추가합니다. 대화 원문, 사용자 메모리, 학습 dataset, adapter, 모델 cache와 상세 실행 로그는 private 저장소에서도 기본적으로 `data-private/`에 두고 커밋하지 않는 편이 안전합니다. 공개에 적합한 범용 개선은 개인 데이터 없이 재현 가능한 형태로 정리해 upstream에 기여할 수 있습니다.

## 현재 상태

| 항목 | 상태 | 근거 |
|---|---|---|
| Qwen3.8-27B DFlash2 추론 | 검증 완료 | GPU1에서 41.47 → 111.95 tok/s, 2.70배 |
| 24GB GPU QLoRA 파이프라인 | 검증 완료 | RTX 3090 GPU1, 96개 샘플, 18 step |
| Naia reference persona v1 | 승격 차단 | 일반 87.5·안전 100 유지, persona 50.0 → 50.0 |
| 대화 기반 야간 FT | 설계 단계 | 자동화 범위는 후보 생성까지, 승격은 수동 |
| 운영 persona adapter | 없음 | 현재 서비스는 unlocked 기준 모델 |

기반 파이프라인과 공개 Naia reference 실험은 24GB GPU에서 끝까지 동작함을 확인했습니다. 첫 Naia 후보는 completion-only 경계를 87개 학습 샘플에서 선검증하고 일반·안전 점수를 유지했지만 persona 향상이 없어 운영 승격을 차단했습니다. 이는 파이프라인 성공과 모델 품질 성공을 분리해 판정한 결과입니다. 정확한 명령, 수치와 다음 실험 조건은 [Naia v1 실험 보고서](docs/reports/naia-persona-v1-experiment-2026-08-29.md)에 있습니다.

## 시스템 경계

```text
naia-shell / 향후 모바일 앱
        │
        ├── 실시간 회상 ───────────────> naia-memory (RAG)
        └── 학습 동의를 받은 대화
                    ↓
             kb-compiler
        정제 · 비식별화 · 중복 제거 · 출처 보존
                    ↓
              naia-persona
 validate → compile → QLoRA → evaluate → candidate
                                            ↓ 사람 승인
                                      inference server
```

- 기억과 최신 사실은 `naia-memory`/RAG에 둡니다.
- 말투, 태도, 응답 구조처럼 반복적이고 안정적인 행동만 FT 대상으로 삼습니다.
- `kb-compiler`와 `naia-memory`는 장기적으로 서버 계층에 두고 노트북·모바일은 음성/UI와 로컬 cache를 맡는 구조를 권장합니다.
- DFlash2와 W4A16 체크포인트는 추론 산출물입니다. 학습은 호환되는 원본 계열을 NF4 QLoRA로 불러 수행합니다.

자세한 책임과 배포 경계는 [아키텍처](docs/ARCHITECTURE.md)를 참고하세요.

## 빠른 시작

데이터 검증과 컴파일은 GPU 없이 실행할 수 있습니다.

```bash
python3 examples/naia-v1/build_dataset.py
python3 scripts/validate_dataset.py examples/naia-v1/source.jsonl
python3 scripts/compile_dataset.py examples/naia-v1/source.jsonl \
  data-private/datasets/naia-v1 --dataset-name naia-v1 --seed naia-v1

# 자기 캐릭터는 동일한 스키마로 별도 파일을 작성합니다.
mkdir -p data-private/{incoming,datasets,runs,registry,hf-cache}
python3 scripts/validate_dataset.py data-private/incoming/persona.jsonl
python3 scripts/compile_dataset.py \
  data-private/incoming/persona.jsonl \
  data-private/datasets/persona-v1 \
  --dataset-name persona-v1
```

학습은 CUDA 환경이 필요합니다. 재현성을 위해 Bazzite 호스트에 라이브러리를 직접 설치하는 대신 Podman/Docker 호환 이미지를 사용합니다.

```bash
podman build -f scripts/container/Containerfile -t naia-persona:dev .
podman run --rm --device nvidia.com/gpu=1 \
  --security-opt=label=disable --shm-size=16g \
  -v "$PWD:/workspace:Z" \
  -v "$PWD/data-private/hf-cache:/root/.cache/huggingface:Z" \
  -v "/path/to/compatible-base-model:/model:ro" \
  naia-persona:dev \
  python3 scripts/train_lora.py \
    --base-model /model \
    --data data-private/datasets/persona-v1/train.jsonl \
    --output data-private/runs/persona-v1
```

이 호스트에서는 GPU0를 다른 세션이 사용하므로 학습은 GPU1에서만 수행합니다. 같은 GPU의 추론 서비스를 먼저 정상 중지하고 VRAM이 비었는지 확인하며, 작업 후에는 성공 여부와 관계없이 서비스를 복구합니다. 실제 순서는 [학습 운영 절차](docs/TRAINING.md)에 있습니다.

## 데이터 계약

각 JSONL 행은 `messages`와 명시적 동의·출처 메타데이터를 가집니다.

```json
{"messages":[{"role":"user","content":"오늘 일정 정리해줘"},{"role":"assistant","content":"좋아. 중요한 순서대로 정리해볼게."}],"meta":{"consent":true,"source_id":"conversation-0001","source_type":"conversation"}}
```

검증기는 동의 누락, 잘못된 대화 순서, 빈 메시지, 중복 샘플과 흔한 개인정보·비밀 패턴을 fail-closed로 거부합니다. 전체 계약은 [데이터 파이프라인](docs/DATA_PIPELINE.md)에 있습니다.

## 승격 원칙

학습 결과는 항상 `candidate`로 시작합니다. 일반 능력 비열등성, persona 향상, 회귀 실패, 개인정보 canary, DFlash2 결합 시 속도·VRAM을 독립적으로 검사합니다. 게이트를 통과해도 사람의 검토 전에는 운영 adapter를 바꾸지 않습니다. 자세한 판정 기준은 [검증 전략](docs/VALIDATION.md)을 참고하세요.

## 문서와 주요 도구

- [문서 인덱스](docs/README.md) — 목적별 읽기 순서와 현재 정본
- [아키텍처](docs/ARCHITECTURE.md) — memory/compiler/persona/serving 경계
- [데이터 파이프라인](docs/DATA_PIPELINE.md) — 스키마, 분할, 야간 후보 생성
- [학습 운영 절차](docs/TRAINING.md) — GPU1, 컨테이너, 복구 순서
- [검증 전략](docs/VALIDATION.md) — 품질·안전·속도 승격 게이트
- [Naia reference persona](examples/naia-v1/README.md) — 공개 캐릭터 카드·데이터·평가 예제
- `scripts/validate_dataset.py` — 데이터 계약 검증
- `scripts/compile_dataset.py` — 결정론적 분할과 manifest 생성
- `scripts/train_lora.py` — 단일 24GB GPU용 QLoRA 진입점
- `scripts/run_local_eval.py` — base/candidate 로컬 평가
- `scripts/evaluate_candidate.py` — 비열등성·persona 승격 판정
- `scripts/benchmark_endpoint.py` — endpoint 속도 측정
- `scripts/merge_adapter.py` — adapter 병합과 provenance 기록

## 공개 프로젝트 범위

공개 저장소에는 다음을 포함합니다.

- 동의·출처·비식별화 규칙을 포함한 데이터 계약
- persona 및 대화 dataset을 검증·컴파일하는 도구
- 24GB GPU에서 재현 가능한 QLoRA 학습 파이프라인
- 기존 모델과 후보 모델을 비교하는 평가·승격 게이트
- 컨테이너 실행 환경, 운영 절차와 비식별화된 실험 보고서

사용자별 실행 데이터는 `data-private/`에 두며 Git에서 제외합니다. 개인 dataset과 adapter는 공개 대상이 아닙니다. Naia 예제 dataset은 실제 대화를 변형한 자료가 아니라 공개 캐릭터 설정에서 행동 원칙을 추출해 새로 작성하며, 기반 모델 관련 산출물은 해당 라이선스를 충족하는 범위에서만 다룹니다.
