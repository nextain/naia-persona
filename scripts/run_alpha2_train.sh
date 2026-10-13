#!/usr/bin/env bash
# 알파 2차 어댑터 학습. 학습 설정은 alpha1 과 같고 데이터만 alpha-v2 로 바꾼다.
#
# alpha-v2 는 기억 층 사실과 평가 문항 근접을 걷어낸 138행이다. 138행도 16행씩
# 9스텝이라 10 epoch 에서 90 스텝으로 alpha1 과 같다. alpha1 이 GPU1 에서 서빙
# 중이므로 GPU 를 인자로 받는다. 학습 GPU 가 alpha1 과 다르다는 것은 run.json 에
# 남는다.
#
# 사용법: run_alpha2_train.sh <gpu-index 0|1> [gpu-memory-gib]
set -euo pipefail

GPU_INDEX="${1:?GPU 번호(0 또는 1)를 주세요}"
GPU_MEM="${2:-21}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="/var/home/luke/alpha-adk/data-private/alpha-corpus"
PARENT="${NAIA_PARENT_MODEL:-/var/mnt/hdd/nextain/naia-persona-models/Qwen3.8-27B-Unlocked-BF16}"
IMAGE="localhost/naia-persona-train:dev"
DATA="$CORPUS/datasets/alpha-v2/train.jsonl"
OUT="$CORPUS/runs/train/alpha-v2-qwen38-27b-gpu${GPU_INDEX}-r128-e10-lr5e5"
DATA_SHA="$(python3 -c "import json;print(json.load(open('$CORPUS/datasets/alpha-v2/manifest.json'))['train_sha256'])")"
GPU_UUID="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
V1_SHA="6e16fc3d9272e69ab9367897bd10112e226a51cf0987615133b8594e90fbaf58"

cd "$REPO"
[ -f "$DATA" ] || { echo "FATAL: 데이터셋이 없습니다: $DATA" >&2; exit 2; }
[ -d "$PARENT" ] || { echo "FATAL: 베이스 체크포인트가 없습니다: $PARENT" >&2; exit 2; }
podman image exists "$IMAGE" || { echo "FATAL: 이미지가 없습니다: $IMAGE" >&2; exit 2; }

actual="$(sha256sum "$DATA" | cut -d' ' -f1)"
[ "$actual" = "$DATA_SHA" ] || { echo "FATAL: 데이터 해시가 어긋납니다: $actual" >&2; exit 2; }
# alpha1 의 데이터는 비교 기준이므로 바이트 단위로 보존되어 있어야 한다.
v1="$(sha256sum "$CORPUS/datasets/alpha-v1/train.jsonl" | cut -d' ' -f1)"
[ "$v1" = "$V1_SHA" ] || { echo "FATAL: alpha-v1 데이터가 바뀌었습니다: $v1" >&2; exit 2; }

# GPU0 에는 ditto 아바타 서버(약 2.6GB)가 상주한다. 우리 것이 아닌 프로세스를
# 세어 여유가 학습 상한보다 작으면 시작하지 않는다.
total="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
used="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
free_gib=$(( (total - used) / 1024 ))
[ "$free_gib" -ge "$GPU_MEM" ] || { echo "FATAL: GPU${GPU_INDEX} 여유 ${free_gib}GiB 가 상한 ${GPU_MEM}GiB 보다 작습니다" >&2; exit 2; }

if [ -e "$OUT" ]; then
  if [ -d "$OUT/adapter" ] || [ -f "$OUT/run.json" ]; then
    echo "FATAL: $OUT 에 완료된 학습이 있습니다. 지우지 않습니다" >&2; exit 2
  fi
  rmdir -- "$OUT" 2>/dev/null || { echo "FATAL: $OUT 에 알 수 없는 파일이 있습니다" >&2; exit 2; }
fi

LOG="$(mktemp -t alpha2-train-XXXXXX.log)"
run_container() {
  podman run --rm --replace --name "alpha2-$1" \
    --security-opt label=disable --device "nvidia.com/gpu=$GPU_INDEX" \
    -e HF_DATASETS_CACHE=/tmp/hf-datasets \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$REPO:/workspace" -v "$CORPUS:$CORPUS" -v "$PARENT:/model:ro" \
    -w /workspace "$IMAGE" \
    python3 scripts/train_lora.py \
      --base-model /model --data "$DATA" --output "$OUT" --profile alpha2 \
      --epochs 10 --learning-rate 5e-5 --max-length 256 \
      --rank 128 --seed 42 --gpu-memory-gib "$GPU_MEM" \
      --expected-data-sha256 "$DATA_SHA" --expected-gpu-uuid "$GPU_UUID" \
      "${@:2}"
}

set +e
{
  echo "=== alpha2 사전 점검 $(date -Is) GPU${GPU_INDEX} ${GPU_UUID} 상한 ${GPU_MEM}GiB ==="
  run_container preflight --preflight-only &&
  echo "=== alpha2 학습 $(date -Is) ===" &&
  run_container train &&
  echo "=== alpha2 완료 $(date -Is) ==="
} 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
echo "EXIT=$status" | tee -a "$LOG"
[ -d "$OUT" ] && cp -- "$LOG" "$OUT/train.log"
exit "$status"
