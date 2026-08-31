#!/usr/bin/env bash
# H23 training run: persona-and-capability-only dataset on physical GPU1.
# Recipe identical to H22; the training rows are the only change.
set -euo pipefail

# 저장소 루트를 스크립트 자신의 위치에서 구한다. 어느 기계에서도 동작한다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
# 학습 대상 원본 체크포인트의 위치. 각자 내려받은 곳을 가리킨다.
PARENT="${NAIA_PARENT_MODEL:-$HOME/models/Qwen3.8-27B-Unlocked-BF16}"
IMAGE="localhost/naia-persona-train:dev"
DATA="data-private/datasets/naia-v12/train.jsonl"
OUT="data-private/runs/train/naia-v12-qwen38-27b-gpu1-r16-e3-lr5e5-h23"
DATA_SHA="4be4cd1208291049c51dbad57a199027a1933799aa8c6ff09e8cb52dee901a39"
GPU_UUID="GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

cd "$REPO"

# Environment gate. A hand-run that skips these is not a verified run.
[ -f "$DATA" ] || { echo "FATAL: dataset missing: $DATA" >&2; exit 2; }
[ -f "data-private/datasets/naia-v12/freeze-manifest.json" ] || { echo "FATAL: freeze manifest missing; freeze before allocating the GPU" >&2; exit 2; }
[ -d "$PARENT" ] || { echo "FATAL: parent checkpoint not mounted: $PARENT" >&2; exit 2; }
podman image exists "$IMAGE" || { echo "FATAL: image missing: $IMAGE" >&2; exit 2; }

actual_sha="$(sha256sum "$DATA" | cut -d' ' -f1)"
[ "$actual_sha" = "$DATA_SHA" ] || { echo "FATAL: dataset digest drifted: $actual_sha" >&2; exit 2; }

# GPU1 must be idle before we claim it, and GPU0 is never touched.
gpu1_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$gpu1_used" -lt 500 ] || { echo "FATAL: GPU1 holds ${gpu1_used} MiB; refusing to share" >&2; exit 2; }

# train_lora.py creates $OUT itself and refuses to run if it already exists, so a
# finished run can never be silently overwritten. Do not pre-create it. A leftover
# directory from an aborted attempt is removed only when it holds nothing but this
# script's own log — never when an adapter or checkpoint is present.
if [ -e "$OUT" ]; then
  leftovers="$(find "$OUT" -mindepth 1 -not -name train.log)"
  if [ -n "$leftovers" ]; then
    echo "FATAL: $OUT holds run artifacts; refusing to remove:" >&2
    echo "$leftovers" >&2
    exit 2
  fi
  echo "removing empty leftover output directory from an aborted attempt: $OUT"
  rm -rf -- "$OUT"
fi

LOG="$(mktemp -t naia-h23-train-XXXXXX.log)"

run_container() {
  podman run --rm \
    --name "naia-persona-h23-$1" \
    --security-opt label=disable \
    --device nvidia.com/gpu=1 \
    -e HF_DATASETS_CACHE=/tmp/hf-datasets \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO:/workspace" \
    -v "$PARENT:/model:ro" \
    -w /workspace "$IMAGE" \
    python3 scripts/train_lora.py \
      --base-model /model \
      --data "$DATA" \
      --output "$OUT" \
      --profile h23 \
      --epochs 3 --learning-rate 5e-5 --max-length 256 \
      --rank 16 --seed 42 --gpu-memory-gib 23 \
      --expected-data-sha256 "$DATA_SHA" \
      --expected-gpu-uuid "$GPU_UUID" \
      "${@:2}"
}

set +e
{
  echo "=== H23 preflight $(date -Is) ==="
  run_container preflight --preflight-only &&
  echo "=== H23 training $(date -Is) ===" &&
  run_container train &&
  echo "=== H23 done $(date -Is) ==="
} 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
echo "EXIT=$status" | tee -a "$LOG"

# Park the log next to the run it describes, now that training has created the
# directory. If training never got that far, leave the log where a reader can
# still find it and say so.
if [ -d "$OUT" ]; then
  cp -- "$LOG" "$OUT/train.log"
  echo "log: $OUT/train.log"
else
  echo "log: $LOG (no output directory was created)"
fi
exit "$status"
