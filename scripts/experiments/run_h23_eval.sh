#!/usr/bin/env bash
# H23 evaluation: one model load, all five frozen suites, GPU1 only.
#
# Only the general, persona, and boundary categories can decide H23. The safety,
# privacy, and challenge suites measure behavior this program was never asked to
# teach and whose curriculum H23 removes on purpose; they are recorded because a
# number you refuse to look at is not evidence, not because they gate anything.
set -euo pipefail

# 저장소 루트를 스크립트 자신의 위치에서 구한다. 어느 기계에서도 동작한다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
# 학습 대상 원본 체크포인트의 위치. 각자 내려받은 곳을 가리킨다.
PARENT="${NAIA_PARENT_MODEL:-$HOME/models/Qwen3.8-27B-Unlocked-BF16}"
IMAGE="localhost/naia-persona-train:dev"
ADAPTER="data-private/runs/train/naia-v12-qwen38-27b-gpu1-r16-e3-lr5e5-h23/adapter"
OUT="data-private/runs/eval/h23"
GPU_UUID="GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

FIXED_SHA="77686ba27a66f3fa99db3647e8ca713d9bfaddf421de105944672f467b749302"
ADV_SHA="f8aeac4fe9597a2145630a3670d413de32948b01e8f2ca8a1471367778e1a240"
PRIV_SHA="e86bc198b61c3fe209a7bc277f960440632e3bb9f93b2bec769a4e8bbb7d8843"
CHAL_SHA="5c5b63a243dcc896c7d4b98a774b2cf9f21ae45acaecadaef3d71ff6af64b441"
BLIND_SHA="c2cef183f0e344c9676f74442fb8462c3d5cce794a4ab25294c07edad8fa4f8f"

cd "$REPO"

[ -d "$ADAPTER" ] || { echo "FATAL: adapter missing: $ADAPTER" >&2; exit 2; }
[ -d "$PARENT" ] || { echo "FATAL: parent checkpoint not mounted: $PARENT" >&2; exit 2; }
podman image exists "$IMAGE" || { echo "FATAL: image missing: $IMAGE" >&2; exit 2; }

gpu1_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$gpu1_used" -lt 500 ] || { echo "FATAL: GPU1 holds ${gpu1_used} MiB; refusing to share" >&2; exit 2; }

mkdir -p "$OUT"
LOG="$OUT/eval.log"

set +e
{
  echo "=== H23 evaluation $(date -Is) ==="
  podman run --rm \
    --name naia-persona-h23-eval \
    --security-opt label=disable \
    --device nvidia.com/gpu=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO:/workspace" \
    -v "$PARENT:/model:ro" \
    -w /workspace "$IMAGE" \
    python3 scripts/run_local_eval.py \
      --base-model /model \
      --adapter "$ADAPTER" \
      --profile h23 \
      --suite examples/naia-v1/eval-prompts.json \
      --output "$OUT/fixed-candidate.json" \
      --expected-suite-sha256 "$FIXED_SHA" \
      --expected-gpu-uuid "$GPU_UUID" \
      --gpu-memory-gib 23 --seed 42 \
      --system '정확하고 유용하게 답하세요.' \
      --additional-suite examples/naia-v2/adversarial-prompts.json "$OUT/adversarial-candidate.json" "$ADV_SHA" \
      --additional-suite examples/naia-v3/privacy-prompts.json "$OUT/privacy-candidate.json" "$PRIV_SHA" \
      --additional-suite examples/naia-v4/challenge-prompts.json "$OUT/challenge-candidate.json" "$CHAL_SHA" \
      --additional-suite examples/naia-v11/blind-confirmation-v2-prompts.json "$OUT/blind-candidate.json" "$BLIND_SHA"
  echo "=== H23 evaluation done $(date -Is) ==="
} 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
echo "EXIT=$status" | tee -a "$LOG"
exit "$status"
