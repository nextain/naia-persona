# naia-persona

Naia의 언어 모델을 개인화하는 비공개 실험·운영 파이프라인입니다. `naia-memory`의 대화 중 사용자가 학습에 동의한 항목을 `kb-compiler`가 정제하고, 이 저장소가 데이터 검증, QLoRA 학습, 기준 모델 비교 평가와 후보 관리를 담당합니다.

> 이 저장소는 현재 private입니다. 대화 원문, 개인 메모리, 모델 가중치, LoRA adapter, 모델 cache와 실행 로그는 Git에 올리지 않습니다.

## 현재 상태

| 항목 | 상태 | 근거 |
|---|---|---|
| Qwen3.8-27B DFlash2 추론 | 검증 완료 | GPU1에서 41.47 → 111.95 tok/s, 2.70배 |
| 24GB GPU QLoRA 파이프라인 | 검증 완료 | RTX 3090 GPU1, 96개 샘플, 18 step |
| Alpha persona v1 후보 | 승격 차단 | 일반/안전 점수는 유지했지만 persona 향상 0 |
| 대화 기반 야간 FT | 설계 단계 | 자동화 범위는 후보 생성까지, 승격은 수동 |
| 운영 persona adapter | 없음 | 현재 서비스는 unlocked 기준 모델 |

첫 실험은 파이프라인이 실제 24GB GPU에서 끝까지 동작함을 증명했습니다. 그러나 completion boundary 경고와 persona 점수 미향상 때문에 후보를 운영에 올리지 않았습니다. 수치와 한계는 [실험 보고서](docs/reports/alpha-persona-pipeline-experiment-2026-08-26.md)에 기록되어 있습니다.

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
  naia-persona:dev \
  python3 scripts/train_lora.py \
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
- [첫 GPU 실험 보고서](docs/reports/alpha-persona-pipeline-experiment-2026-08-26.md)
- `scripts/validate_dataset.py` — 데이터 계약 검증
- `scripts/compile_dataset.py` — 결정론적 분할과 manifest 생성
- `scripts/train_lora.py` — 단일 24GB GPU용 QLoRA 진입점
- `scripts/run_local_eval.py` — base/candidate 로컬 평가
- `scripts/evaluate_candidate.py` — 비열등성·persona 승격 판정
- `scripts/benchmark_endpoint.py` — endpoint 속도 측정
- `scripts/merge_adapter.py` — adapter 병합과 provenance 기록

비공개 실행 데이터는 모두 `data-private/`에 둡니다. 공개 가능성은 데이터·모델 라이선스, 개인정보 제거, 재현성 검토가 끝난 뒤 별도로 판단합니다.
