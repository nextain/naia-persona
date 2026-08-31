#!/usr/bin/env bash
# Serve a candidate, run the 100-item persona benchmark against it, stop the server.
#
# Until now the serve, the wait, and the measurement were three manual steps, and
# every candidate cost the same hand-holding. This is the whole measure phase of
# the quality loop as one command, so re-measuring is cheap enough that nobody
# skips it.
#
#   measure_candidate.sh h27
#
# The adapter directory and model id are derived from the label, so a new
# candidate needs no new script.
set -euo pipefail

LABEL="${1:-}"
[ -n "$LABEL" ] || { echo "usage: $0 <label>   e.g. $0 h27" >&2; exit 2; }

REPO="<repo>"
PARENT="<models>/Qwen3.8-27B-Unlocked-BF16"
IMAGE="localhost/naia-persona-train:dev"
KEYS="<home>/alpha-adk/data-private/key/llm-key.env"
PORT=8010
NAME="naia-measure-$LABEL"

cd "$REPO"

# The label "parent" measures the un-fine-tuned checkpoint. The benchmark asks
# for this control because an axis the parent already passes is not measuring
# the persona, and no result can be trusted until it has been run once.
CONTROL=""
if [ "$LABEL" = "parent" ]; then
  ADAPTER="none"
  CONTROL="--control"
else
  # One adapter per label. Fail loudly rather than measuring the wrong weights.
  ADAPTER="$(find data-private/runs/train -maxdepth 1 -type d -name "*-${LABEL}" | head -1)/adapter"
  [ -d "$ADAPTER" ] || { echo "FATAL: no adapter found for label '$LABEL'" >&2; exit 2; }
  matches="$(find data-private/runs/train -maxdepth 1 -type d -name "*-${LABEL}" | wc -l)"
  [ "$matches" -eq 1 ] || { echo "FATAL: '$LABEL' matches $matches run directories; be specific" >&2; exit 2; }
fi

[ -d "$PARENT" ] || { echo "FATAL: parent checkpoint not mounted" >&2; exit 2; }
podman image exists "$IMAGE" || { echo "FATAL: image missing: $IMAGE" >&2; exit 2; }
[ -f "$KEYS" ] || { echo "FATAL: judge credentials missing: $KEYS" >&2; exit 2; }

gpu1_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
[ "$gpu1_used" -lt 500 ] || { echo "FATAL: GPU1 holds ${gpu1_used} MiB; refusing to share" >&2; exit 2; }

cleanup() { podman stop "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "serving $ADAPTER as $NAME"
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
    --base-model /model --adapter "$ADAPTER" \
    --model-id "naia-$LABEL" --host 0.0.0.0 --port "$PORT" --gpu-memory-gib 23 >/dev/null

echo "waiting for the model to load (about 8 minutes)"
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  podman ps --format '{{.Names}}' | grep -q "$NAME" || {
    echo "FATAL: server exited before becoming ready" >&2
    podman logs --tail 15 "$NAME" 2>&1 | tail -15 >&2 || true
    exit 1
  }
  sleep 20
done
echo "ready"

set -a; . "$KEYS"; set +a
export GEMINI_API_KEY
python3 -u scripts/run_persona_benchmark.py \
  --endpoint "http://127.0.0.1:$PORT" \
  --model "naia-$LABEL" \
  --label "$LABEL" $CONTROL
status=$?
echo "benchmark exit=$status"
exit "$status"
