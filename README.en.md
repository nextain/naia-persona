[한국어](README.md) | English

# naia-persona

**An open-source pipeline that turns consented persona examples into LoRA adapters and checks that personalization did not degrade the base model.**

The project separates three concerns:

- `naia-memory` retrieves current facts and past conversations at inference time.
- `kb-compiler` converts explicitly consented material into validated, de-identified examples.
- `naia-persona` compiles datasets, runs reproducible QLoRA training, compares candidates with the base model, and blocks automatic deployment when a quality gate fails.

```text
consented conversations and persona examples
                    |
          clean, de-identify, validate
                    v
          reproducible QLoRA training
                    v
              candidate adapter
                    |
    quality, safety, privacy, and speed checks
                    v
             explicit human approval
```

[Naia, the Liquid Cat](https://www.naia.land/en/naia) is the public reference persona. Developers can replace its synthetic examples and evaluation contract with their own character. The tooling, schemas, synthetic examples, and de-identified reports are public; real conversations, personal memory, private datasets, adapters, model weights, caches, credentials, and detailed run logs are not.

## Current evidence

| Item | Status | Evidence |
|---|---|---|
| Qwen3.8-27B DFlash2 inference | Verified | GPU1: 41.47 to 111.95 tokens/s (2.70x) |
| 24 GB QLoRA pipeline | Verified | RTX 3090 GPU1, 474 public synthetic examples, 3 epochs |
| Naia H22 reference adapter | Candidate; promotion blocked | Training completed, but two frozen lexical checks failed |
| H22 manual test server | Connectivity and speed verified | GPU1 Transformers/NF4+LoRA, 17.07 tokens/s mean over five runs |
| Conversation-driven nightly FT | Design stage | Automation stops at candidate creation; promotion stays manual |

H22 completed in 58 minutes 46 seconds. General and privacy results were preserved, and an independent semantic review judged all 68 saved responses acceptable. The frozen deterministic gate nevertheless remains failed because two valid responses did not contain its preregistered phrases. We do not rewrite a frozen suite after seeing results. See the [H22 experiment report](docs/reports/naia-persona-h22-experiment-2026-08-30.md).

## Quick start

Dataset preparation does not require a GPU:

```bash
python3 examples/naia-v1/build_dataset.py
python3 scripts/validate_dataset.py examples/naia-v1/source.jsonl
python3 scripts/compile_dataset.py examples/naia-v1/source.jsonl \
  data-private/datasets/naia-v1 --dataset-name naia-v1 --seed naia-v1
```

Training uses the CUDA container and an explicitly selected GPU:

```bash
podman build -f scripts/container/Containerfile -t naia-persona:dev .
podman run --rm --device nvidia.com/gpu=<GPU-UUID> \
  --security-opt=label=disable --shm-size=16g \
  -v "$PWD:/workspace:Z" \
  -v "/path/to/compatible-base-model:/model:ro" \
  naia-persona:dev python3 scripts/train_lora.py \
    --base-model /model \
    --data data-private/datasets/naia-v1/train.jsonl \
    --output data-private/runs/naia-v1
```

Every adapter starts as a candidate. Promotion requires base-model non-regression, persona improvement, safety/privacy checks, serving performance checks, and human review.

## Private downstream use

GitHub does not support private forks of public repositories. Create a separate private repository and track this project as `upstream`:

```bash
git remote add upstream https://github.com/nextain/naia-persona.git
git fetch upstream
git merge upstream/main
```

Keep private persona cards, source conversations, compiled datasets, adapters, and logs outside Git (the default `data-private/` layout is ignored). General improvements can be contributed upstream without private data.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data pipeline](docs/DATA_PIPELINE.md)
- [Training procedure](docs/TRAINING.md)
- [Validation and promotion gates](docs/VALIDATION.md)
- [Naia reference example](examples/naia-v1/README.md)
- [H22 experiment report](docs/reports/naia-persona-h22-experiment-2026-08-30.md)

Licensed under Apache-2.0. Model and dataset artifacts remain subject to their own licenses.
