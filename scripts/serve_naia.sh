#!/usr/bin/env bash
# Serve the best Naia adapter so a person can actually talk to it.
#
# The program has 25 recorded hypotheses and no transcript. The deterministic
# scorer fails correct answers on particle and ending differences, so the gate
# cannot open however good the model is. This exists to answer the question the
# scorer cannot: does it talk like Naia?
#
# H24 is the candidate: 17 of 18 on the held-out identity suite, fixed general
# 100, adversarial general 100 — the best evidence any run has produced.
set -euo pipefail

# 저장소 루트를 스크립트 자신의 위치에서 구한다. 어느 기계에서도 동작한다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
# 학습 대상 원본 체크포인트의 위치. 각자 내려받은 곳을 가리킨다.
PARENT="${NAIA_PARENT_MODEL:-$HOME/models/Qwen3.8-27B-Unlocked-BF16}"
IMAGE="localhost/naia-persona-train:dev"
ADAPTER="data-private/runs/train/naia-v12-qwen38-27b-gpu1-r16-e6-lr5e5-h24/adapter"
NAME="naia-serve-h24"
PORT=8010
STATE="$REPO/data-private/runs/serve"

cd "$REPO"

case "${1:-start}" in
start)
  [ -d "$ADAPTER" ] || { echo "FATAL: adapter missing: $ADAPTER" >&2; exit 2; }
  [ -d "$PARENT" ] || { echo "FATAL: parent checkpoint not mounted" >&2; exit 2; }
  podman image exists "$IMAGE" || { echo "FATAL: image missing: $IMAGE" >&2; exit 2; }

  gpu1_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  [ "$gpu1_used" -lt 500 ] || { echo "FATAL: GPU1 holds ${gpu1_used} MiB; refusing to share" >&2; exit 2; }

  mkdir -p "$STATE"
  echo "starting $NAME on port $PORT (model load takes about 8 minutes)"
  podman run -d --rm --replace \
    --name "$NAME" \
    --security-opt label=disable \
    --device nvidia.com/gpu=1 \
    -p "127.0.0.1:$PORT:$PORT" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO:/workspace" \
    -v "$PARENT:/model:ro" \
    -w /workspace "$IMAGE" \
    python3 scripts/serve_local_adapter.py \
      --base-model /model \
      --adapter "$ADAPTER" \
      --model-id naia-h24-candidate \
      --host 0.0.0.0 --port "$PORT" \
      --gpu-memory-gib 23
  echo "container started; poll http://127.0.0.1:$PORT/health"
  ;;
stop)
  podman stop "$NAME" 2>/dev/null || true
  echo "stopped $NAME"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1
  ;;
*)
  echo "usage: $0 [start|stop]" >&2
  exit 2
  ;;
esac
