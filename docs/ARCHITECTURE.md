# naia-persona architecture

## 책임 경계

`naia-memory`는 대화 원문과 장기 기억의 정본이며 실시간 RAG를 제공합니다. `kb-compiler`는 동의된 대화를 학습 가능한 작은 단위로 정제하고 출처와 품질 정보를 유지합니다. `naia-persona`는 그 결과의 검증, 학습, 평가, candidate registry를 담당합니다. 추론 서버는 승인된 adapter만 읽습니다.

노트북이나 향후 모바일 앱은 voice/UI와 임시 로컬 캐시를 맡습니다. memory와 compiler는 장기적으로 GPU PC 또는 별도 서버 계층에 통합해 여러 클라이언트가 동일한 기억과 데이터 정책을 보도록 합니다. 단, memory API와 학습 worker의 권한은 분리합니다. 학습 worker는 compiler가 내보낸 승인 dataset만 읽고 원본 memory 저장소를 임의 탐색하지 않습니다.

## 데이터 흐름

```text
client → memory/RAG → response
   └→ explicit consent → compiler → immutable dataset
                                   → validator → QLoRA run
                                                → evaluation
                                                → candidate registry
                                                → manual promotion
```

각 dataset에는 생성 시각, compiler 버전, source 범위, consent, 제거 규칙, 해시를 기록해야 합니다. 원문 대화와 dataset, Hugging Face cache, adapter와 run 결과는 `data-private/` 아래에 두며 git에 커밋하지 않습니다.

## 모델 계보

학습 parent는 PEFT가 지원하는 Qwen3.8-27B 원본 계열이어야 합니다. 현재 W4A16 AutoRound checkpoint와 DFlash2 drafter는 inference artifact이며 학습 parent가 아닙니다. 기존 unlocked adapter를 계속 학습하려면 base revision, target modules, rank, tokenizer/chat template이 일치하는지 먼저 확인합니다. 불일치하면 unlocked adapter와 persona adapter를 별도로 학습·조합하는 실험 트랙을 만들며 조용히 병합하지 않습니다.

## 24GB 실행 경계

- NF4 double quantization과 BF16 compute
- batch size 1, gradient accumulation 16
- gradient checkpointing, sequence length 512에서 시작
- GPU1 추론 서비스와 학습 worker 동시 실행 금지
- OOM 시 sequence length를 먼저 낮추고 target module/rank를 조정
- 컨테이너는 교체 가능하며 데이터와 결과는 host volume에 유지

## 승격 계약

학습 산출물의 기본 상태는 언제나 `candidate`입니다. 승격에는 dataset validation 통과, base/unlocked/candidate 고정 평가 비교, 개인정보 회귀 검사, 대표 Naia 대화 수동 검토, 처리량과 VRAM 확인이 모두 필요합니다. scheduler는 수집·검증·후보 학습을 자동화할 수 있지만 production adapter 포인터는 변경할 수 없습니다. 직전 adapter와 manifest를 보존해 rollback 가능해야 합니다.

## 현재 검증 상태와 한계

2026-08-26에 GPU1 RTX 3090 24GB에서 Qwen3.8-27B QLoRA의 데이터 컴파일부터 학습·평가·승격 차단까지 한 번 완주했습니다. 같은 GPU에서 DFlash2가 고정 조건의 단일 스트림 생성 속도를 41.47 tok/s에서 111.95 tok/s로 높이는 것도 확인했습니다.

첫 Alpha 후보는 일반·안전 sanity 점수를 유지했지만 persona 점수가 기준보다 향상되지 않아 승격하지 않았습니다. 학습 중 completion boundary mismatch 경고도 있어 다음 실험 전에 masking 검증을 보강해야 합니다. 이 결과는 8개 프롬프트의 파이프라인 확인이며 대규모 제품 벤치마크가 아닙니다. 상세 조건은 [실험 보고서](./reports/alpha-persona-pipeline-experiment-2026-08-26.md)에 있습니다.
