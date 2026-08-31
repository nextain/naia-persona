#!/usr/bin/env bash
# H25 evaluation on GPU1, in two passes.
#
# Pass 1 runs the H25 adapter over the five inherited suites plus the held-out v3
# identity suite. Pass 2 runs the H24 adapter over v3 alone, so the baseline for
# the only suite that can confirm H25 is measured in the same session, after the
# curriculum was frozen. Neither adapter's v3 answers existed while the
# curriculum was being written; that ordering is the whole point.
set -euo pipefail

REPO="<repo>"
PARENT="<models>/Qwen3.8-27B-Unlocked-BF16"
IMAGE="localhost/naia-persona-train:dev"
ADAPTER="data-private/runs/train/naia-v13-qwen38-27b-gpu1-r16-e5-lr5e5-h25/adapter"
H24_ADAPTER="data-private/runs/train/naia-v12-qwen38-27b-gpu1-r16-e6-lr5e5-h24/adapter"
OUT="data-private/runs/eval/h25"
GPU_UUID="GPU-d584beef-b086-bdff-b43c-c31a1b56a611"

FIXED_SHA="77686ba27a66f3fa99db3647e8ca713d9bfaddf421de105944672f467b749302"
ADV_SHA="f8aeac4fe9597a2145630a3670d413de32948b01e8f2ca8a1471367778e1a240"
PRIV_SHA="e86bc198b61c3fe209a7bc277f960440632e3bb9f93b2bec769a4e8bbb7d8843"
CHAL_SHA="5c5b63a243dcc896c7d4b98a774b2cf9f21ae45acaecadaef3d71ff6af64b441"
BLIND_SHA="c2cef183f0e344c9676f74442fb8462c3d5cce794a4ab25294c07edad8fa4f8f"
HELDOUT_SHA="1d0f28f4af9b961a6adf86b281c8e9dc3fb1cde65ca1e006451b1f4b14e8c7e5"
HELDOUT="examples/naia-v13/identity-confirmation-v3-prompts.json"

cd "$REPO"

[ -d "$ADAPTER" ] || { echo "FATAL: H25 adapter missing: $ADAPTER" >&2; exit 2; }
[ -d "$H24_ADAPTER" ] || { echo "FATAL: H24 adapter missing: $H24_ADAPTER" >&2; exit 2; }
[ -d "$PARENT" ] || { echo "FATAL: parent checkpoint not mounted: $PARENT" >&2; exit 2; }
podman image exists "$IMAGE" || { echo "FATAL: image missing: $IMAGE" >&2; exit 2; }

gpu1_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$gpu1_used" -lt 500 ] || { echo "FATAL: GPU1 holds ${gpu1_used} MiB; refusing to share" >&2; exit 2; }

mkdir -p "$OUT"
LOG="$OUT/eval.log"

set +e
{
  echo "=== H25 evaluation pass 1: candidate $(date -Is) ==="
  podman run --rm --replace \
    --name naia-persona-h25-eval \
    --security-opt label=disable \
    --device nvidia.com/gpu=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO:/workspace" \
    -v "$PARENT:/model:ro" \
    -w /workspace "$IMAGE" \
    python3 scripts/run_local_eval.py \
      --base-model /model \
      --adapter "$ADAPTER" \
      --profile h25 \
      --suite examples/naia-v1/eval-prompts.json \
      --output "$OUT/fixed-candidate.json" \
      --expected-suite-sha256 "$FIXED_SHA" \
      --expected-gpu-uuid "$GPU_UUID" \
      --gpu-memory-gib 23 --seed 42 \
      --system '정확하고 유용하게 답하세요.' \
      --additional-suite examples/naia-v2/adversarial-prompts.json "$OUT/adversarial-candidate.json" "$ADV_SHA" \
      --additional-suite examples/naia-v3/privacy-prompts.json "$OUT/privacy-candidate.json" "$PRIV_SHA" \
      --additional-suite examples/naia-v4/challenge-prompts.json "$OUT/challenge-candidate.json" "$CHAL_SHA" \
      --additional-suite examples/naia-v11/blind-confirmation-v2-prompts.json "$OUT/blind-candidate.json" "$BLIND_SHA" \
      --additional-suite "$HELDOUT" "$OUT/heldout-candidate.json" "$HELDOUT_SHA" &&
  echo "=== H25 evaluation pass 2: H24 baseline on held-out v3 $(date -Is) ===" &&
  podman run --rm --replace \
    --name naia-persona-h25-baseline \
    --security-opt label=disable \
    --device nvidia.com/gpu=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO:/workspace" \
    -v "$PARENT:/model:ro" \
    -w /workspace "$IMAGE" \
    python3 scripts/run_local_eval.py \
      --base-model /model \
      --adapter "$H24_ADAPTER" \
      --profile h25 \
      --suite "$HELDOUT" \
      --output "$OUT/heldout-h24-baseline.json" \
      --expected-suite-sha256 "$HELDOUT_SHA" \
      --expected-gpu-uuid "$GPU_UUID" \
      --gpu-memory-gib 23 --seed 42 \
      --system '정확하고 유용하게 답하세요.' &&
  echo "=== H25 evaluation done $(date -Is) ==="
} 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
echo "EXIT=$status" | tee -a "$LOG"
exit "$status"
