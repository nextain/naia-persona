# 24GB GPU 학습 운영 절차

## 원칙

- 이 호스트에서는 GPU0를 사용하지 않는다. 예약된 GPU1만 사용한다.
- 27B 추론과 학습을 GPU1에서 동시에 실행하지 않는다.
- 학습 결과는 운영 모델이 아니라 `candidate`이다.
- 실패하더라도 추론 서비스 복구와 실행 기록 저장을 수행한다.
- 모델·데이터·adapter는 `data-private/` 아래에 두고 Git에 커밋하지 않는다.

## 1. 데이터 준비

```bash
python3 scripts/validate_dataset.py data-private/incoming/persona.jsonl
python3 scripts/compile_dataset.py \
  data-private/incoming/persona.jsonl \
  data-private/datasets/persona-v1 \
  --dataset-name persona-v1 --seed 42 --holdout-ratio 0.05
```

`manifest.json`의 입력·분할 SHA-256과 행 수를 실행 기록에 보존한다.

## 2. 이미지와 GPU1 준비

```bash
podman build -f scripts/container/Containerfile -t naia-persona:dev .
podman ps --format '{{.Names}}'
podman stop <inference-container>
nvidia-smi --id=1 --query-compute-apps=pid,used_memory --format=csv
```

GPU1에 예상하지 않은 프로세스가 남아 있으면 학습을 시작하지 않습니다.

## 3. QLoRA 실행

```bash
podman run --rm --device nvidia.com/gpu=1 \
  --security-opt=label=disable --shm-size=16g \
  -v "$PWD:/workspace:Z" \
  -v "$PWD/data-private/hf-cache:/root/.cache/huggingface:Z" \
  naia-persona:dev \
  python3 scripts/train_lora.py \
    --data data-private/datasets/persona-v1/train.jsonl \
    --output data-private/runs/persona-v1 \
    --epochs 3 --learning-rate 2e-5 --max-length 256 \
    --rank 4 --seed 42 --gpu-memory-gib 22
```

OOM이 나면 sequence length를 먼저 낮추고, 다음으로 rank나 target module을 조정합니다. 설정이 다른 실행은 새 run 디렉터리를 사용합니다.

## 4. 평가와 복구

```bash
python3 scripts/evaluate_candidate.py \
  data-private/runs/eval/baseline.json \
  data-private/runs/eval/candidate.json \
  --output data-private/runs/eval/gate.json

podman start <inference-container>
python3 scripts/observe_serving.py --url http://<server>:<port>
```

`eligible_for_manual_review`가 아니면 후보를 격리합니다. 통과해도 자동 배포하지 않습니다. 야간 자동화는 `trap` 또는 동등한 supervisor를 사용해 성공·실패 모두에 복구 단계를 실행해야 합니다. GPU lease, 실행 잠금, 알림이 준비되기 전에는 cron에 직접 연결하지 않습니다.
