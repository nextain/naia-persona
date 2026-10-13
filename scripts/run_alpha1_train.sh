#!/usr/bin/env bash
# 알파 어댑터 학습. 129행을 10 epoch 으로 돌려 90 스텝을 맞춘다.
#
# 데이터는 이 저장소가 아니라 data-private 의 코퍼스에서 온다. 코퍼스가 산출물이고
# 어댑터는 그것이 작동한다는 증명이다. 베이스 모델을 바꿀 때는 코퍼스에서 다시
# 생성해 이 스크립트의 데이터 해시만 갱신하면 된다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="/var/home/luke/alpha-adk/data-private/alpha-corpus"
PARENT="${NAIA_PARENT_MODEL:-/var/mnt/hdd/nextain/naia-persona-models/Qwen3.8-27B-Unlocked-BF16}"
IMAGE="localhost/naia-persona-train:dev"
DATA="$CORPUS/datasets/alpha-v1/train.jsonl"
OUT="$CORPUS/runs/train/alpha-v1-qwen38-27b-gpu1-r128-e10-lr5e5"
DATA_SHA="$(python3 -c "import json;print(json.load(open('$CORPUS/datasets/alpha-v1/manifest.json'))['train_sha256'])")"
GPU_UUID="GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

cd "$REPO"
[ -f "$DATA" ] || { echo "FATAL: 데이터셋이 없습니다: $DATA" >&2; exit 2; }
[ -d "$PARENT" ] || { echo "FATAL: 베이스 체크포인트가 없습니다: $PARENT" >&2; exit 2; }
podman image exists "$IMAGE" || { echo "FATAL: 이미지가 없습니다: $IMAGE" >&2; exit 2; }

actual="$(sha256sum "$DATA" | cut -d' ' -f1)"
[ "$actual" = "$DATA_SHA" ] || { echo "FATAL: 데이터 해시가 어긋납니다: $actual" >&2; exit 2; }

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$used" -lt 500 ] || { echo "FATAL: GPU1 이 ${used} MiB 를 쓰고 있습니다" >&2; exit 2; }

# 시작 시점에만 확인하면 부족하다. 베이스 모델을 디스크에서 읽는 데 8분이 걸리고,
# 그 사이에 다른 학습이 들어와 메모리를 선점하면 적재가 끝나는 순간 OOM 으로
# 죽는다. 실제로 그렇게 한 번 잃었다. 적재 중에도 남의 프로세스가 들어오는지
# 지켜보다가, 들어오면 우리가 GPU 를 잡기 전에 먼저 물러난다.
watch_gpu() {
  local mine="$1"
  while kill -0 "$mine" 2>/dev/null; do
    local others
    others="$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v "^$" || true)"
    for pid in $others; do
      if ! podman top "alpha1-train" pid 2>/dev/null | tail -n +2 | grep -qw "$pid"; then
        if ! grep -q "podman\|conmon" "/proc/$pid/comm" 2>/dev/null; then
          echo "경고: GPU1 에 다른 프로세스가 들어왔습니다 (pid $pid). 학습이 실패할 수 있습니다" >&2
        fi
      fi
    done
    sleep 20
  done
}

if [ -e "$OUT" ]; then
  if [ -d "$OUT/adapter" ] || [ -f "$OUT/run.json" ]; then
    echo "FATAL: $OUT 에 완료된 학습이 있습니다. 지우지 않습니다" >&2; exit 2
  fi
  rm -rf -- "$OUT"
fi

LOG="$(mktemp -t alpha1-train-XXXXXX.log)"
run_container() {
  podman run --rm --replace --name "alpha1-$1" \
    --security-opt label=disable --device nvidia.com/gpu=1 \
    -e HF_DATASETS_CACHE=/tmp/hf-datasets \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO:/workspace" -v "$CORPUS:$CORPUS" -v "$PARENT:/model:ro" \
    -w /workspace "$IMAGE" \
    python3 scripts/train_lora.py \
      --base-model /model --data "$DATA" --output "$OUT" --profile alpha1 \
      --epochs 10 --learning-rate 5e-5 --max-length 256 \
      --rank 128 --seed 42 --gpu-memory-gib 23 \
      --expected-data-sha256 "$DATA_SHA" --expected-gpu-uuid "$GPU_UUID" \
      "${@:2}"
}

set +e
{
  echo "=== alpha1 사전 점검 $(date -Is) ==="
  run_container preflight --preflight-only &&
  echo "=== alpha1 학습 $(date -Is) ===" &&
  run_container train &&
  echo "=== alpha1 완료 $(date -Is) ==="
} 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
echo "EXIT=$status" | tee -a "$LOG"
[ -d "$OUT" ] && cp -- "$LOG" "$OUT/train.log"
exit "$status"
