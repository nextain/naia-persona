#!/usr/bin/env bash
# H25 training run: 287 rows for 5 epochs on physical GPU1.
# The 239 inherited rows are untouched; 48 identity-diversity rows are appended.
# Epochs drop 6 -> 5 for one reason only: 287 rows at 5 epochs is again 90
# optimizer steps, so the comparison against H24 is about the added rows.
set -euo pipefail

REPO="<repo>"
PARENT="<models>/Qwen3.8-27B-Unlocked-BF16"
IMAGE="localhost/naia-persona-train:dev"
DATA="data-private/datasets/naia-v13/train.jsonl"
OUT="data-private/runs/train/naia-v13-qwen38-27b-gpu1-r16-e5-lr5e5-h25"
DATA_SHA="570bbc1088a61f76032feefc1ec14d56850c5baf26b9906a1914cbc1619b5fe6"
GPU_UUID="GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

cd "$REPO"

[ -f "$DATA" ] || { echo "FATAL: dataset missing: $DATA" >&2; exit 2; }
[ -f "data-private/datasets/naia-v13/freeze-manifest.json" ] || { echo "FATAL: H25 freeze manifest missing; freeze before allocating the GPU" >&2; exit 2; }
[ -d "$PARENT" ] || { echo "FATAL: parent checkpoint not mounted: $PARENT" >&2; exit 2; }
podman image exists "$IMAGE" || { echo "FATAL: image missing: $IMAGE" >&2; exit 2; }

actual_sha="$(sha256sum "$DATA" | cut -d' ' -f1)"
[ "$actual_sha" = "$DATA_SHA" ] || { echo "FATAL: dataset digest drifted: $actual_sha" >&2; exit 2; }

gpu1_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$gpu1_used" -lt 500 ] || { echo "FATAL: GPU1 holds ${gpu1_used} MiB; refusing to share" >&2; exit 2; }

# train_lora.py creates $OUT and refuses to run if it already exists. Remove a
# leftover only when it holds nothing but this script's own log.
# A finished run always writes both adapter/ and run.json. A directory holding
# neither never produced a result, so it is an aborted attempt and may be
# cleared. Anything with either one is evidence and stops the run instead.
if [ -e "$OUT" ]; then
  if [ -d "$OUT/adapter" ] || [ -f "$OUT/run.json" ]; then
    echo "FATAL: $OUT holds a completed run (adapter/ or run.json present); refusing to remove" >&2
    exit 2
  fi
  echo "clearing an aborted attempt with no adapter and no run.json: $OUT"
  rm -rf -- "$OUT"
fi

LOG="$(mktemp -t naia-h25-train-XXXXXX.log)"

# A crash that leaves podman unable to record an exit code — a full disk, for
# instance — strands the container name and blocks the next attempt. --replace
# clears a stranded corpse by name. It cannot silently kill live work here
# because the run already refuses to start unless GPU1 is idle.
run_container() {
  podman run --rm --replace \
    --name "naia-persona-h25-$1" \
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
      --profile h25 \
      --epochs 5 --learning-rate 5e-5 --max-length 256 \
      --rank 16 --seed 42 --gpu-memory-gib 23 \
      --expected-data-sha256 "$DATA_SHA" \
      --expected-gpu-uuid "$GPU_UUID" \
      "${@:2}"
}

set +e
{
  echo "=== H25 preflight $(date -Is) ==="
  run_container preflight --preflight-only &&
  echo "=== H25 training $(date -Is) ===" &&
  run_container train &&
  echo "=== H25 done $(date -Is) ==="
} 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
echo "EXIT=$status" | tee -a "$LOG"

if [ -d "$OUT" ]; then
  cp -- "$LOG" "$OUT/train.log"
  echo "log: $OUT/train.log"
else
  echo "log: $LOG (no output directory was created)"
fi
exit "$status"
