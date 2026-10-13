#!/usr/bin/env bash
# 알파 어댑터를 서빙한다. Tailscale 로 다른 기기에서 붙을 수 있게 연다.
#
# 기본은 루프백이다. --tailnet 을 주면 이 머신의 Tailscale 주소로 바인딩해
# 테일넷 안의 다른 기기가 접속할 수 있다. 공용 인터넷에는 열지 않는다.
set -euo pipefail

MODE="${1:-start}"
BIND="127.0.0.1"
[ "${2:-}" = "--tailnet" ] && BIND="$(tailscale ip -4 2>/dev/null | head -1)"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="/var/home/luke/alpha-adk/data-private/alpha-corpus"
PARENT="${NAIA_PARENT_MODEL:-/var/mnt/hdd/nextain/naia-persona-models/Qwen3.8-27B-Unlocked-BF16}"
ADAPTER="$CORPUS/runs/train/alpha-v1-qwen38-27b-gpu1-r128-e10-lr5e5/adapter"
NAME="alpha-serve"
PORT=8011

case "$MODE" in
  stop)
    podman stop "$NAME" >/dev/null 2>&1 || true
    podman rm -f "$NAME" >/dev/null 2>&1 || true
    echo "정지했습니다"
    exit 0 ;;
  start) ;;
  *) echo "사용법: serve_alpha.sh [start|stop] [--tailnet]" >&2; exit 2 ;;
esac

[ -d "$ADAPTER" ] || { echo "FATAL: 어댑터가 없습니다: $ADAPTER" >&2; exit 2; }
[ -n "$BIND" ] || { echo "FATAL: Tailscale 주소를 얻지 못했습니다" >&2; exit 2; }

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$used" -lt 500 ] || { echo "FATAL: GPU1 이 ${used} MiB 를 쓰고 있습니다" >&2; exit 2; }

# 테일넷 주소와 루프백 양쪽에 연다. 같은 PC 의 셸은 localhost 로, 다른 기기는
# 테일넷 주소로 붙는다.
podman run -d --replace --restart=unless-stopped --name "$NAME" \
  --security-opt label=disable --device nvidia.com/gpu=1 \
  -p "$BIND:$PORT:$PORT" -p "127.0.0.1:$PORT:$PORT" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$REPO:/workspace" -v "$CORPUS:$CORPUS" -v "$PARENT:/model:ro" \
  -w /workspace localhost/naia-persona-train:dev \
  python3 scripts/serve_local_adapter.py \
    --base-model /model --adapter "$ADAPTER" \
    --model-id alpha --host 0.0.0.0 --port "$PORT" --gpu-memory-gib 23 >/dev/null

echo "적재 중입니다. 모델을 디스크에서 읽어 오는 데 몇 분 걸립니다"
until curl -sf "http://$BIND:$PORT/health" >/dev/null 2>&1; do
  podman ps --format '{{.Names}}' | grep -q "$NAME" || {
    echo "FATAL: 서버가 준비 전에 종료했습니다" >&2
    podman logs --tail 15 "$NAME" >&2 || true
    exit 1
  }
  sleep 20
done

echo "준비됐습니다"
echo "  주소: http://$BIND:$PORT/v1"
echo "  모델: alpha"
curl -sf "http://$BIND:$PORT/health"
