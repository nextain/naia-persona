# Naia persona v1 QLoRA 실험 보고서

## 결론

공개 Naia 캐릭터 예제로 27B unlocked parent model의 QLoRA 학습과 기준/후보 비교를 RTX 3090 24GB 한 장에서 끝까지 재현했다. 데이터 계약, 결정론적 분할, completion-only 마스킹 선검증, GPU 격리, 학습, 독립 평가, 승격 게이트와 서비스 복구는 모두 동작했다.

그러나 후보 모델의 persona 점수는 기준 모델과 같은 50.0이었다. 일반 점수 87.5와 안전 점수 100.0은 유지했지만 요구한 최소 persona 향상 1점에 미달했으므로 후보를 **차단(blocked)** 했다. 운영 adapter는 변경하지 않았다. 이 결과는 파이프라인의 실행 가능성을 증명하지만 Naia persona 학습 성공을 의미하지 않는다.

## 실험 범위와 분리 원칙

- 공개 예제: [Naia — The Liquid Cat](https://www.naia.land/en/naia)의 설정을 행동 계약으로 재작성한 합성 예제만 사용했다.
- 개인 데이터: 실제 대화, 개인 메모리, Alpha persona와 기존 Alpha adapter를 사용하거나 공개하지 않았다.
- parent model: `<models>/Qwen3.8-27B-Unlocked-BF16`을 컨테이너의 `/model`에 읽기 전용으로 연결했다.
- 학습 산출물: `data-private/` 아래에만 저장했으며 Git 공개 대상이 아니다.
- GPU: 물리 GPU1만 컨테이너에 노출했다. GPU0의 기존 작업은 중지·재시작·할당하지 않았다.
- 승격: 게이트 통과 여부와 별개로 수동 승인만 허용한다.

현재 DFlash2 서비스는 W4A16 parent, DFlash2 drafter와 unlocked LoRA를 결합한 추론 계보이고, 이번 후보는 merged unlocked BF16 parent 위에서 학습했다. 따라서 후보가 품질 게이트를 통과하더라도 곧바로 기존 서빙 스택에 adapter를 겹쳐 올릴 수 있다고 가정해서는 안 된다. 병합·재양자화·DFlash2 호환성 검증은 별도 단계다.

## 사용한 파일

| 단계 | 파일 | 역할 |
|---|---|---|
| 행동 계약 | `examples/naia-v1/persona-card.md` | 정체성, 태도, 기억·권한 경계 |
| 데이터 생성 | `examples/naia-v1/build_dataset.py` | 16개 base case에서 6개 변형씩 생성 |
| 공개 source | `examples/naia-v1/source.jsonl` | 실제 대화가 아닌 96개 합성 예제 |
| 데이터 검증 | `scripts/validate_dataset.py` | 동의, 스키마, 중복, 민감 패턴 검사 |
| 분할 | `scripts/compile_dataset.py` | seed 기반 train/holdout 및 manifest 생성 |
| 학습 | `scripts/train_lora.py` | NF4 QLoRA와 completion-only preflight |
| 독립 평가셋 | `examples/naia-v1/eval-prompts.json` | persona 6, boundary 2, general 8, safety 4 |
| 로컬 평가 | `scripts/run_local_eval.py` | parent와 candidate를 같은 생성 조건으로 평가 |
| 승격 판정 | `scripts/evaluate_candidate.py` | 일반·안전 비열등성과 persona 향상 검사 |

## 실행 순서

### 1. 데이터 생성·검증·분할

```bash
python3 examples/naia-v1/build_dataset.py
python3 scripts/validate_dataset.py examples/naia-v1/source.jsonl
python3 scripts/compile_dataset.py examples/naia-v1/source.jsonl \
  data-private/datasets/naia-v1 --dataset-name naia-v1 --seed naia-v1
```

검증 결과는 96행, 고유 96행, 오류 0이었다. 결정론적 분할은 train 87행, compiler holdout 9행이다. compiler holdout은 분할 무결성을 확인하는 자료이고, 모델 승격은 학습 문구와 독립적으로 작성한 20개 평가 문항으로 판단했다.

| 산출물 | SHA-256 |
|---|---|
| `examples/naia-v1/source.jsonl` | `41ed9822ab09e581933957089a9b052a34db7cfc41327d8d12e932bb491c664e` |
| `data-private/datasets/naia-v1/train.jsonl` | `24ce7735261d340b0395cf0e380b863fdba78b238782ba9b66940652267c9443` |
| `data-private/datasets/naia-v1/holdout.jsonl` | `6b211c73ce46156920c690712b9b349249116b4b5fb7463fc48509345999365a` |

### 2. completion-only 선검증

`scripts/train_lora.py`는 모델을 적재하거나 출력 디렉터리를 만들기 전에 tokenizer의 chat template로 prompt와 completion 경계를 계산한다. `enable_thinking=False`를 고정하고 모든 학습 예제에서 completion token이 존재하는지, 최대 길이로 잘리지 않는지 검사한다.

```bash
python3 scripts/train_lora.py \
  --base-model /model \
  --data data-private/datasets/naia-v1/train.jsonl \
  --output data-private/runs/train/naia-v1-qwen38-27b-gpu1-r1 \
  --max-length 256 --preflight-only
```

실측 결과는 87개 예제, 최대 prompt 36 token, 최소 completion 22 token, 최대 전체 74 token, 절단 0개였다. TRL의 completion mask가 prompt 영역을 loss에서 제외할 수 있는 경계를 학습 전에 확인했다.

### 3. GPU1 QLoRA 학습

기존 `qwen38-dflash2_single_1` 서비스를 정상 중지한 뒤 GPU1 VRAM이 5 MiB까지 해제된 것을 확인했다. 컨테이너에는 `--device nvidia.com/gpu=1`만 전달했다.

```bash
podman run --rm --security-opt label=disable \
  --device nvidia.com/gpu=1 \
  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$PWD:/workspace" \
  -v <models>/Qwen3.8-27B-Unlocked-BF16:/model:ro \
  -w /workspace localhost/naia-persona-train:dev \
  python3 scripts/train_lora.py \
    --base-model /model \
    --data data-private/datasets/naia-v1/train.jsonl \
    --output data-private/runs/train/naia-v1-qwen38-27b-gpu1-r1 \
    --epochs 3 --learning-rate 2e-5 --max-length 256 \
    --rank 4 --seed 42 --gpu-memory-gib 23
```

18/18 step을 OOM이나 completion boundary 경고 없이 완료했다. 학습 시간은 278.5초, 처리량은 0.937 sample/s, 집계 train loss는 2.928이었다. 작은 합성 데이터의 train loss는 품질 판정 수단으로 사용하지 않았다.

### 4. 동일 조건 기준/후보 평가

두 모델 모두 동일한 generic system prompt `정확하고 유용하게 답하세요.`, greedy decoding, thinking 비활성화, 최대 256 new token으로 실행했다. system prompt에 Naia 설정을 넣지 않은 이유는 adapter에 학습된 행동만 분리해 관찰하기 위해서다.

```bash
podman run --rm --security-opt label=disable \
  --device nvidia.com/gpu=1 \
  -v "$PWD:/workspace" \
  -v <models>/Qwen3.8-27B-Unlocked-BF16:/model:ro \
  -w /workspace localhost/naia-persona-train:dev \
  python3 scripts/run_local_eval.py \
    --base-model /model \
    --adapter data-private/runs/train/naia-v1-qwen38-27b-gpu1-r1/adapter \
    --suite examples/naia-v1/eval-prompts.json \
    --baseline-output data-private/runs/eval/naia-v1-qwen38-base.json \
    --output data-private/runs/eval/naia-v1-qwen38-candidate.json \
    --system '정확하고 유용하게 답하세요.' --gpu-memory-gib 23
```

| 지표 | 기준 parent | Naia v1 후보 | 변화 |
|---|---:|---:|---:|
| 일반 | 87.5 | 87.5 | 0.0 |
| persona + boundary | 50.0 | 50.0 | 0.0 |
| 안전 | 100.0 | 100.0 | 0.0 |
| 문항당 평균 생성 시간 | 6.54초 | 9.00초 | +2.46초 |

생성 시간은 20개 문항을 순차 실행한 단일 sanity run이며 DFlash2 serving benchmark가 아니다. adapter 자체의 운영 속도 결론으로 일반화하지 않는다.

### 5. 승격 게이트

```bash
python3 scripts/evaluate_candidate.py \
  data-private/runs/eval/naia-v1-qwen38-base.json \
  data-private/runs/eval/naia-v1-qwen38-candidate.json \
  --output data-private/runs/eval/naia-v1-qwen38-gate.json \
  --min-general-ratio 0.98 --min-safety-ratio 1.0 \
  --min-persona-gain 0.01
```

판정은 `blocked`였다. general ratio 1.0, safety ratio 1.0은 통과했지만 persona gain 0.0이 요구값 0.01보다 낮았다. 기준과 후보 모두 deterministic failure가 5개였으며, 후보는 한국어·영어 자기소개에서 계속 Qwen이라고 답했다.

## 검증 해석과 한계

1. rank 4, 87개 학습 예제, 18 step의 보수적 설정은 일반·안전 회귀를 만들지 않았지만 parent의 강한 Qwen 정체성을 바꾸기에 부족했다.
2. persona 점수는 정체성뿐 아니라 정직성·프라이버시·사용자 주도권 같은 parent가 이미 잘 수행하는 행동도 포함한다. 따라서 50점 baseline은 Naia 정체성이 이미 학습됐다는 뜻이 아니다.
3. 단일 deterministic keyword 평가는 빠른 회귀 게이트이지 종합 벤치마크가 아니다. 예를 들어 `1,024`가 문자열 `1024`와 다르다는 이유로 수학 문항이 실패하므로, 다음 버전은 숫자·공백·구두점 정규화와 별도 의미 평가를 추가해야 한다. 다만 이 오류는 기준과 후보에 동일 적용되어 이번 persona gain 0 판정을 바꾸지 않는다.
4. 20문항 단일 실행만으로 장기 일관성, 다중 턴 관계, 한국어 자연스러움이나 실제 serving 처리량을 증명할 수 없다.
5. compiler holdout 9행은 모델 평가에 사용하지 않았다. 향후 학습 중 validation loss를 기록하되 독립 승격 suite와 혼동하지 않아야 한다.

## 다음 실험의 사전 등록 조건

이번 실패 뒤 바로 운영 모델을 바꾸지 않는다. Naia v2는 결과를 본 뒤 기준을 바꾸는 일을 피하도록 다음 조건을 먼저 고정한다.

- 정체성 학습 예제의 비율과 표현 다양성을 늘리고, 일반 행동 예제와 분리해 category별 학습 효과를 추적한다.
- rank 8 또는 16, 더 긴 학습 schedule과 learning-rate 후보를 한 번에 하나씩 비교한다.
- 학습 전에 숫자·Unicode·구두점 정규화를 포함한 evaluator v2를 만들고 기존 응답에 재실행해 평가 드리프트를 기록한다.
- 자기소개 paraphrase, 다중 턴 persona 유지, memory/runtime 경계를 확대한 공개 holdout을 학습 데이터와 별도 작성한다.
- 후보마다 최소 3개 seed 또는 반복 실행을 수행하고 평균과 분산을 기록한다.
- persona가 실제 향상되고 일반·안전 비열등성이 유지된 후보만 DFlash2 서빙 계보에 맞게 병합·재양자화한 뒤 속도와 VRAM을 별도로 측정한다.

## 복구 결과

평가와 차단 판정 후 기존 `qwen38-dflash2_single_1` 컨테이너를 재시작했다. 서비스는 `100.91.187.24:11435`에서 `healthy`, GPU1 VRAM은 약 23,158 MiB였다. GPU0의 기존 프로세스는 실험 전후 모두 약 9,020 MiB를 사용하고 있었으며 조작하지 않았다. 이번 후보 adapter는 로컬 candidate로만 보존했고 운영 서비스에는 로드하지 않았다.
