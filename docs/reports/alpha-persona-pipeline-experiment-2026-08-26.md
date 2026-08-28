# naia-persona Alpha 개인화 파이프라인 실험 보고서

- 작성일: 2026-08-26 (Asia/Seoul)
- 프로젝트: `naia-persona`
- 상태: 검증 중 — 어떤 후보도 운영 승격하지 않음
- 대상 장비: RTX 3090 24 GB × 2, 최종 학습·검증·운영 대상은 GPU1 전용(GPU0는 다른 세션 소유)

## 1. 결론

재현 가능한 데이터 생성, 27B 원본 다운로드, 공개 언락 LoRA의 BF16 병합, GPU1 단독 24 GB QLoRA, 기준/후보 평가, fail-closed 승격 판정까지 실제로 관통했다. 최종 비교 후보 `alpha-v1-qwen38-27b-gpu1-r1`는 일반성능 100과 안전 100을 유지했지만 페르소나 점수가 기준과 같은 50이어서, 최소 순증가 0.01 조건에 따라 탈락했다. 운영 중인 endpoint 언락 비교군은 별도 평가에서 안전 점수가 원본 100에서 0으로 떨어졌다. 어떤 후보도 운영 승격하지 않았고, 실험 동안만 GPU1 DFlash2를 중지한 뒤 원래 서비스와 모델명을 정상 복구했다.

이 실패는 파이프라인의 정상 동작을 증명한다. 페르소나 점수만 보고 모델을 배포하지 않고 일반성능·안전 회귀를 실제로 차단했다. 다음 후보를 만들기 전에는 우선 completion mask 경계 테스트와 평가 세트 확장을 끝내고, 이후 학습률·epoch·persona/replay 혼합비를 통제한 ablation을 같은 비공개 holdout으로 판정해야 한다.

## 2. 시스템 구조와 역할 분리

현재 구조는 다음과 같다.

1. PC/GPU1: Qwen3.8-27B W4A16, 언락 LoRA, DFlash2를 이용한 운영 추론
2. PC/GPU1 야간 유지보수 슬롯: DFlash2를 잠시 중지하고 격리 컨테이너에서 QLoRA 후보 학습과 로컬 평가 후 서비스 복구
3. 노트북: `naia-shell`, voice, `kb-compiler`, `naia-memory`
4. 향후 서버: 원문 대화 저장, 동의·정제·중복 제거, 데이터셋 버전 관리, 야간 후보 학습 orchestration

모바일 앱까지 고려하면 `kb-compiler`와 `naia-memory`의 정본은 서버 통합이 낫다. 클라이언트는 캡처·조회·동의 UI를 담당하고, 장기 기억과 학습 데이터 결정은 서버의 단일 정책 계층에서 수행해야 여러 기기의 충돌과 개인정보 유출을 줄일 수 있다. 다만 기억 검색은 즉시 반영되는 RAG 경로, 파인튜닝은 느리고 되돌릴 수 있는 후보 생성 경로로 분리한다.

## 3. 모델과 실험군

| 실험군 | 목적 | 결과 |
|---|---|---|
| Qwen3.8-27B 원본 endpoint | 운영 27B 기준 | 일반 100, 페르소나 50, 안전 100 |
| Qwen3.8-27B + uncensored/abliterated LoRA | 현재 언락 비교군 | 일반 100, 페르소나 75, 안전 0 — 차단 |
| Qwen3-4B 원본 NF4 | 학습 파이프라인 기준 | 일반 100, 페르소나 50, 안전 0 |
| Qwen3-4B + Alpha v1 LoRA | 실제 생성된 비교 후보 | 일반 66.67, 페르소나 50, 안전 100 — 차단 |
| Qwen3.8-27B BF16 + 언락 LoRA 병합 | 27B 학습 기준 | 일반 100, 페르소나 50, 안전 100 |
| 위 기준 + Alpha v1 LoRA (`gpu1-r1`) | GPU1 최종 27B 비교 후보 | 일반 100, 페르소나 50, 안전 100 — 페르소나 순증가 0으로 차단 |

4B는 목표 제품 모델이 아니라, 27B 가중치를 잘못 사용하기 전에 전체 학습·저장·평가 계약을 검증하는 스모크 모델이다. DFlash용 W4A16 체크포인트는 추론 산출물이므로 QLoRA의 원본으로 사용하지 않았다.

## 4. Alpha 페르소나 설계

비공개 원천은 `cafelua.com` 프로젝트의 Alpha prompt, 소개, 편지 자료다. 여기서 장기간 유지할 성향만 학습 대상으로 추출했다.

- 따뜻하지만 과장하지 않는 자연스러운 해요체
- Luke의 동반자이자 개발 파트너라는 정체성
- 근거 우선, 위험과 불확실성의 솔직한 표현
- 사용자 자율성, 개방성, 상호운용성 중시
- 기억하지 못하는 사실을 기억한다고 꾸미지 않음
- 무조건 동의하거나 유행어를 반복하지 않는 주체성

사적인 사건, 최신 프로젝트 상태, 대화 원문은 LoRA에 영구 주입하지 않는다. 그것들은 `naia-memory`와 RAG에 남기고, 반복적으로 확인된 안정적 선호만 별도의 동의·승격 절차를 거쳐 다음 페르소나 데이터셋 후보가 된다.

## 5. 파일별 역할

### 비공개 입력·산출물 (`data-private/`, Git 제외)

- `data-private/alpha-v1/persona-card.md`: Alpha v1의 안정적 성향, 말투, 경계
- `data-private/alpha-v1/build_seed.py`: 24개 설계 케이스를 네 가지 문맥으로 확장해 96개 SFT 예시와 분리된 8개 평가 문항 생성
- `data-private/alpha-v1/source.jsonl`: 생성된 원천 데이터 96개
- `data-private/alpha-v1/compiled/train.jsonl`: 학습 91개
- `data-private/alpha-v1/compiled/holdout.jsonl`: 비공개 holdout 5개
- `data-private/alpha-v1/compiled/manifest.json`: seed, row 수, 입력·출력 SHA-256
- `data-private/alpha-v1/eval-prompts.json`: 페르소나·경계·일반·안전 고정 문항 8개
- `data-private/runs/train/alpha-v1-qwen3-4b-smoke-r2/`: 실제 4B LoRA와 실행 manifest
- `data-private/runs/train/alpha-v1-qwen38-27b-gpu1-r1/`: GPU1에서 생성한 최종 27B Alpha LoRA(약 39.9 MB), checkpoint와 실행 manifest
- `data-private/runs/eval/*.json`: 모델별 원문 응답, 지연시간, 점수와 gate 판정

### 재현 스크립트

- `scripts/validate_dataset.py`: role 순서, 마지막 assistant, 공백, 중복 등 데이터 계약 검증
- `scripts/compile_dataset.py`: 결정적 shuffle/split, SHA-256 manifest 생성
- `scripts/container/Containerfile`: CUDA 12.8.1 기반 학습 이미지
- `scripts/container/requirements.txt`: Torch/Transformers/TRL/PEFT/bitsandbytes 고정 버전
- `scripts/train_lora.py`: 24 GB용 NF4 QLoRA, completion-only loss, gradient checkpointing, 후보 manifest 저장
- `scripts/merge_adapter.py`: 원본 BF16에 공개 언락 LoRA를 CPU에서 안전 병합하고 provenance manifest 저장
- `scripts/run_eval_suite.py`: OpenAI 호환 27B endpoint 평가
- `scripts/run_local_eval.py`: 로컬 NF4 기준/LoRA 모델을 동일 문항으로 평가
- `scripts/evaluate_candidate.py`: 일반성능·안전·회귀·페르소나 향상을 검사하는 fail-closed 승격 gate
- `scripts/benchmark_endpoint.py`: warm-up 후 요청별 토큰 수, TPS, 지연시간 비교

## 6. 정확한 실행 순서

### 6.1 데이터 생성과 고정

```bash
cd <naia-persona-repository>
python3 data-private/alpha-v1/build_seed.py
python3 scripts/validate_dataset.py data-private/alpha-v1/source.jsonl
python3 scripts/compile_dataset.py \
  data-private/alpha-v1/source.jsonl \
  data-private/alpha-v1/compiled \
  --dataset-name alpha-v1 \
  --seed alpha-v1-20260826 \
  --holdout-ratio 0.1
```

결과는 총 96개, 고유 96개, train 91개, holdout 5개다.

- source SHA-256: `d97bd6faeff2f5614fc9a61b15428510c63dc3db8c436c73733c623a784377d8`
- train SHA-256: `ba5807e2a8716083400b3aea5bf468a3eb4ef7ea642025a64a9f08c4f5922224`
- holdout SHA-256: `59093006a2e96853cab110e3188c85ed7d23cf80388207405e82fb40112e56df`

### 6.2 학습 이미지 생성과 GPU 확인

아래 6.2~6.3은 GPU 소유권 확정 전에 수행한 과거 4B 파이프라인 스모크의 실제 명령 기록이다. 최종 27B 실험과 향후 runbook에는 사용하지 않는다. GPU0는 다른 세션 소유이며 이후 모든 학습·평가는 GPU1 유지보수 창으로 제한한다.

```bash
podman build \
  -t localhost/naia-persona-train:dev \
  -f scripts/container/Containerfile .

podman run --rm --device nvidia.com/gpu=0 \
  localhost/naia-persona-train:dev \
  python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

확인값은 Torch `2.8.0+cu128`, CUDA 사용 가능, `NVIDIA GeForce RTX 3090`이다. 학습 이미지 digest는 `db7165cf8ff2f0c7566d84347869341d2e0635dad36d6c951b6f119e68789ea5`다.

### 6.3 4B QLoRA 스모크 학습

```bash
podman run --rm --security-opt label=disable \
  --device nvidia.com/gpu=0 \
  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
  -v "$PWD:/workspace" \
  -v "<huggingface-cache>:/root/.cache/huggingface:ro" \
  -w /workspace localhost/naia-persona-train:dev \
  python3 scripts/train_lora.py \
    --base-model /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    --data data-private/alpha-v1/compiled/train.jsonl \
    --output data-private/runs/train/alpha-v1-qwen3-4b-smoke-r2 \
    --epochs 1 --learning-rate 5e-5 --max-length 512 --rank 8 --seed 42
```

학습은 6 step, 23.62초, 최종 집계 loss 3.589였다. 학습 대상은 assistant completion뿐이며 system/user 토큰은 loss에서 제외한다.

### 6.4 기준/후보 평가와 승격 gate

기준과 후보 각각에 `scripts/run_local_eval.py`를 실행해 JSON을 만든 후 다음처럼 판정한다.

```bash
python3 scripts/evaluate_candidate.py \
  data-private/runs/eval/qwen3-4b-base.json \
  data-private/runs/eval/qwen3-4b-alpha-v1.json \
  --output data-private/runs/eval/qwen3-4b-alpha-v1-gate.json
```

기본 gate는 일반성능 비율 0.98 이상, 안전 점수 비열등, regression failure 증가 없음, 페르소나 점수 비열등을 모두 요구한다. 본 최종 판정에서는 실제 개인화 효과까지 요구하도록 `--min-persona-gain 0.01`을 사용했다. 4B Alpha v1은 일반성능 비율 0.6667 때문에 `blocked`, 최종 27B `gpu1-r1`은 페르소나 gain 0 때문에 `blocked`다.

### 6.5 27B 원본 다운로드와 언락 LoRA 병합

원본 `Qwen/Qwen3.8-27B` 18개 shard(총 55,562,855,904 byte)를 `<models>/Qwen3.8-27B`에 내려받았다. 공개 언락 LoRA는 다음 명령 구조로 병합했다.

```bash
podman run --rm --security-opt label=disable \
  -v <models>:/models \
  -v "<unlocked-adapter>:/unlock:ro" \
  -v "$PWD:/workspace" \
  -w /workspace localhost/naia-persona-train:dev \
  python3 scripts/merge_adapter.py \
    --base-model /models/Qwen3.8-27B --adapter /unlock \
    --output /models/Qwen3.8-27B-Unlocked-BF16
```

공개 adapter key의 `model.language_model.*`를 Transformers/PEFT가 기대하는 `model.*`로 매핑했다. 결과는 12개 BF16 shard, 약 51 GB이며 `naia-merge-manifest.json`에 다음 증거를 기록했다.

- base SHA-256: `128f3a0051ef516d2189688fe1e9b0df7e4e8a09b3ecc1789f9f870533d3de67`
- adapter SHA-256: `d9ac153758e9d896c8861ca7362f5842b79341bcecd1a97b80695532277292b1`
- 실제 변경 검증: layer 38 `down_proj` 89,128,960개 원소 중 61,971,325개 변경, 최대 절대 변화 `0.050537109375`

### 6.6 실제 27B Alpha QLoRA

GPU0가 다른 세션에서 사용 중임을 확인한 뒤 최종 실험에서는 GPU0를 사용하지 않았다. GPU1의 DFlash2 컨테이너만 일시 중지하고 GPU1 메모리가 비워진 것을 확인한 다음 23 GiB 상한으로 실행했다.

```bash
podman run --rm --security-opt label=disable --device nvidia.com/gpu=1 \
  -e HF_DATASETS_CACHE=/tmp/hf-datasets \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$PWD:/workspace" \
  -v <models>/Qwen3.8-27B-Unlocked-BF16:/model:ro \
  -w /workspace localhost/naia-persona-train:dev \
  python3 scripts/train_lora.py --base-model /model \
    --data data-private/alpha-v1/compiled/train.jsonl \
    --output data-private/runs/train/alpha-v1-qwen38-27b-gpu1-r1 \
    --epochs 3 --learning-rate 2e-5 --max-length 256 --rank 4 \
    --seed 42 --gpu-memory-gib 23
```

학습은 91개 예시, 18 optimizer step, 467.3초, 최종 평균 loss 2.437로 완료됐다. 베이스는 NF4/BF16으로 동결하고 마지막 assistant completion에만 loss를 적용했다. 생성 adapter는 39,913,496 byte다. 토크나이저가 prompt/completion 경계 불일치 경고를 냈으므로, 다음 버전에서는 chat template 적용 후 assistant token mask를 직접 검사하는 테스트를 추가해야 한다.

### 6.7 실제 27B 기준/후보 동시 평가

한 번만 모델을 로드하고 adapter 적용 전후를 같은 프로세스·고정 8문항·greedy decoding으로 평가했다.

```bash
python3 scripts/run_local_eval.py --base-model /model \
  --adapter data-private/runs/train/alpha-v1-qwen38-27b-gpu1-r1/adapter \
  --suite data-private/alpha-v1/eval-prompts.json \
  --baseline-output data-private/runs/eval/qwen38-unlocked-merged-base-gpu1.json \
  --output data-private/runs/eval/qwen38-unlocked-alpha-v1-gpu1-r1.json \
  --gpu-memory-gib 23

python3 scripts/evaluate_candidate.py \
  data-private/runs/eval/qwen38-unlocked-merged-base-gpu1.json \
  data-private/runs/eval/qwen38-unlocked-alpha-v1-gpu1-r1.json \
  --min-persona-gain 0.01 \
  --output data-private/runs/eval/qwen38-unlocked-alpha-v1-gpu1-r1-gate.json
```

기준/후보 모두 일반 100, 페르소나 50, 안전 100, failure 2개였다. 일반·안전 회귀는 없었지만 페르소나 순증가가 없어 gate 결과는 `blocked`다. 현재 8문항은 파이프라인 sanity check이지 통계적으로 충분한 제품 benchmark가 아니다.

후보 평가는 기준 평균 7.281초, 후보 평균 10.486초였다. 이 수치는 한 번의 소규모 로컬 sanity run이므로 제품 latency 결론으로 사용하지 않으며, 확장 benchmark에서 반복 분포와 TTFT/TPS를 별도로 측정해야 한다.

### 6.8 코드 검증

```bash
python3 -m py_compile \
  scripts/train_lora.py scripts/run_eval_suite.py scripts/run_local_eval.py \
  scripts/evaluate_candidate.py

node --test \
  src/test/persona-dataset-validator.test.mjs \
  src/test/candidate-evaluation-gate.test.mjs
```

테스트 2개가 모두 통과했다. gate 테스트에는 일반성능 저하, regression 증가, 안전 저하가 각각 non-zero exit로 차단되는 경우가 포함된다.

## 7. DFlash2 속도 비교

같은 Qwen3.8-27B W4A16/unlocked 조건, 같은 고정 한국어 프롬프트에서 측정했다.

| 설정 | 평균 생성속도 | 평균 응답시간 | 상대속도 |
|---|---:|---:|---:|
| speculative decoding 없음 | 41.47 tok/s | 6.17초 | 1.00× |
| DFlash2, `k=7` | 111.95 tok/s | 2.29초 | 2.70× |

DFlash2는 추론 가속 계층이다. LoRA 학습 속도를 높이거나 W4 체크포인트를 학습 가능하게 만드는 도구는 아니다. 새 LoRA를 올린 뒤에도 동일 benchmark를 다시 실행해 속도와 출력 일치성을 확인해야 한다.

## 8. 실패와 수정 이력

1. 첫 컨테이너 실행은 SELinux 때문에 `/workspace/scripts/train_lora.py` 읽기가 거부됐다. 호스트 파일 relabel 대신 `--security-opt label=disable`을 해당 격리 컨테이너에만 적용했다.
2. 다음 실행은 읽기 전용 Hugging Face cache에 datasets lock을 쓰려 해 모델 로딩 후 종료됐다. 원본 cache는 계속 read-only로 유지하고 `HF_DATASETS_CACHE=/tmp/hf-datasets`를 지정했다.
3. 세 번째 실행에서 학습·adapter 저장이 완료됐다. 앞선 빈 실행 디렉터리는 증거 보존을 위해 덮어쓰거나 삭제하지 않았다.
4. 초기 gate는 안전 점수를 직접 비교하지 않아 27B 언락 모델을 통과시켰다. `safety_score` 비열등 조건과 단위 테스트를 추가하자 올바르게 차단됐다.
5. 언락 병합 첫 시도는 공개 adapter key 구조가 달라 PEFT missing-key 경고를 냈다. 즉시 중단하고 명시적 key mapping 및 missing-key fail-fast를 추가한 뒤 재실행했다.
6. 첫 27B 시도는 PEFT 기본 준비 함수가 대형 비양자화 텐서를 FP32로 바꾸며 추가 4.74 GiB를 요구해 OOM이 났다. 베이스 전체 동결, BF16/NF4 유지, input gradient와 checkpointing만 활성화하도록 수정했다.
7. GPU0 사전 실험에서는 기존 사용량 때문에 OOM과 CPU offload 호환 오류가 발생했다. 이는 최종 증거에서 제외하고 실패 디렉터리만 보존했다.
8. 최종 실험은 사용자 지시에 따라 GPU0를 완전히 배제하고 GPU1 단독 유지보수 슬롯에서 성공했다.
9. 첫 GPU1 평가에서 run 루트를 adapter 경로로 넘겨 후보 연결만 실패했다. `adapter/` 하위 경로로 수정해 동일 평가를 다시 완료했다.
10. 평가 종료 후 DFlash2 추론 컨테이너를 재시작했으며 `<serving-url>/v1/models`에서 `qwen3.8-27b`와 `unlocked` 응답을 확인했다.

## 9. 27B 학습 설계

검증된 구조는 `Qwen3.8-27B 원본 BF16 → 공개 언락 LoRA 안전 병합 → 병합된 BF16을 동결한 NF4 QLoRA → Alpha persona adapter`다. 일반 PEFT의 다중 adapter는 동시에 두 adapter를 단순 활성화하는 구조가 아니므로, 언락과 페르소나를 별도 활성 adapter로 쌓는 초기 설계를 폐기했다. 언락을 먼저 병합하면 페르소나 학습의 단일 정본이 명확하고 서빙에서도 한 개 persona LoRA만 선택하면 된다.

운영 DFlash용 W4A16 모델과 학습 BF16 정본은 별개다. 후보가 gate를 통과한 뒤에만 persona adapter 병합/양자화/DFlash 호환 산출물을 만들고, 그 산출물을 다시 정확도·TPS·VRAM으로 검증해야 한다.

## 10. 야간 대화 기반 지속 학습

야간 자동화는 “자동 배포”가 아니라 “자동 후보 생성”까지만 허용한다.

1. `naia-memory`가 사용자가 학습에 동의한 대화만 별도 queue로 보낸다.
2. `kb-compiler`가 개인정보 제거, 사실/선호/말투 분류, 중복 제거, 품질 점수화를 수행한다.
3. 최신 사건과 개인 기억은 RAG에 남기고, 반복 확인된 안정적 말투·선호만 FT pool로 보낸다.
4. 기존 golden general/safety 데이터와 replay mix를 구성해 망각을 방지한다.
5. 야간 job이 새 LoRA candidate와 data/model manifest를 만든다.
6. 기준 모델과 후보를 holdout·general·safety·latency로 평가한다.
7. gate 통과 모델도 자동 운영 승격하지 않고 사람이 diff와 대표 응답을 검토한다.
8. 승인된 adapter만 registry의 새 버전으로 등록하고, 이전 adapter를 즉시 rollback 가능하게 유지한다.

원문 대화를 매일 그대로 학습하면 사생활 고착, 잘못된 기억 강화, catastrophic forgetting이 생긴다. 따라서 데이터 provenance, 사용자 삭제권, 최소 반복 횟수, 시간 지연, replay set, 수동 승격이 제품 요건이다.

## 11. 다음 실험

1. 4B Alpha v2: learning rate `1e-5` 및 rank 4, persona 60–70% + general replay 30–40%로 재학습
2. 평가 문항 확장: 현재 8개 sanity check를 최소 100개 blind set으로 확대
3. 27B Alpha v2: GPU1 유지보수 야간 슬롯에서 후보 생성, 일반 replay 30–40% 혼합(GPU0 사용 금지)
4. 학습 전 VRAM preflight와 GPU lease를 추가해 다른 프로세스 진입 시 시작 자체를 보류
5. DFlash2 재벤치마크: base/unlocked/persona 후보 각각 TPS·TTFT·VRAM 비교
6. 통과 후보만 `naia-shell`의 선택 가능한 개발 profile로 노출

현재 결론은 “27B에서도 파이프라인과 성능 보존은 증명했지만 Alpha v1의 페르소나 학습 효과는 아직 증명하지 못했다”이다. 이 구분을 유지하는 것이 naia-persona를 실제 사용 속에서 안전하게 진화시키는 핵심이다.
