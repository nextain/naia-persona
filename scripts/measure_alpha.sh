#!/usr/bin/env bash
# 알파 어댑터를 띄워 120항목으로 측정하고 정리한다.
set -euo pipefail

LABEL="${1:-alpha1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="/var/home/luke/alpha-adk/data-private/alpha-corpus"
PARENT="${NAIA_PARENT_MODEL:-/var/mnt/hdd/nextain/naia-persona-models/Qwen3.8-27B-Unlocked-BF16}"
ADAPTER="$CORPUS/runs/train/alpha-v1-qwen38-27b-gpu1-r128-e10-lr5e5/adapter"
KEYS="${NAIA_JUDGE_KEYS:-/var/home/luke/alpha-adk/data-private/key/llm-key.env}"
NAME="alpha-measure"
PORT=8012

[ -d "$ADAPTER" ] || { echo "FATAL: 어댑터가 없습니다: $ADAPTER" >&2; exit 2; }
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$used" -lt 500 ] || { echo "FATAL: GPU1 이 ${used} MiB 를 쓰고 있습니다" >&2; exit 2; }

cleanup() { podman stop "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

cd "$REPO"
podman run -d --rm --replace --name "$NAME" \
  --security-opt label=disable --device nvidia.com/gpu=1 \
  -p "127.0.0.1:$PORT:$PORT" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$REPO:/workspace" -v "$CORPUS:$CORPUS" -v "$PARENT:/model:ro" \
  -w /workspace localhost/naia-persona-train:dev \
  python3 scripts/serve_local_adapter.py --base-model /model --adapter "$ADAPTER" \
  --model-id "$LABEL" --host 0.0.0.0 --port "$PORT" --gpu-memory-gib 23 >/dev/null

echo "적재 중입니다"
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  podman ps --format '{{.Names}}' | grep -q "$NAME" || {
    echo "FATAL: 서버가 준비 전에 종료했습니다" >&2; podman logs --tail 15 "$NAME" >&2; exit 1; }
  sleep 20
done
echo "준비됐습니다"

set -a; . "$KEYS"; set +a
python3 -u scripts/run_persona_benchmark.py \
  --endpoint "http://127.0.0.1:$PORT" --model "$LABEL" --label "$LABEL" \
  --benchmark "$CORPUS/benchmark/alpha-benchmark-v1.json" \
  --card "$CORPUS/card/alpha-persona-card.md" \
  --out-dir "$CORPUS/runs"
