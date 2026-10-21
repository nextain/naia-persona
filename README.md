**English** | [한국어](README.ko.md)

# naia-persona

This repository handles model evolution and personalization for Naia.

## Purpose

Most AI services today run on someone else's server. Users hand over their conversations and get results back, but they cannot decide what personality that AI has, what it remembers, or when it changes. When the service shuts down, the relationship built up over that time disappears with it.

Naia starts from the premise that individuals own their AI directly. Both the model and the data stay on their own hardware, and they decide its personality and memory themselves. This goal is split into two layers. `naia-memory` is the memory layer that retrieves the facts and past conversations that are needed, and `naia-persona` is the learning layer that folds accumulated conversations and experience into the model weights.

What this repository is responsible for is model evolution and personalization for Naia as a whole. Not only tone and attitude: everything that should change as the model lives with its user enters the model through here.

The end goal is for Naia to evolve and optimize itself as it accumulates conversations and experience.

**The current stage is a manual fine-tuning tool, and we are working on securing a validation method.** There is no automatic evolution loop yet. A person builds the data, runs training, and checks the result.

The order is this way because automation depends on judgment. In a loop where no person checks each time, if you cannot tell whether a given training run made things better or worse, the personality quietly breaks. So we build the judgment before the automation. Chatting with the model and deciding by eye that it seems fine is not reproducible, and keyword matching marks correct answers wrong and wrong answers correct. In practice, while this program ran 25 hypotheses with such a scorer, it did not promote any of them.

Training itself is a single command. Most of this repository is about judgment.

## Relation to other repositories

Naia is split across several repositories, and this one owns the part that changes the model.

```text
naia-shell / naia-os          the app and OS users actually use
 │
 ├── real-time recall ─────────> naia-memory   memory layer (RAG)
 │
 └── conversations consented for training
      ↓
     kb-compiler                 cleanup · de-identification · deduplication
      ↓
     naia-persona                this repository
                                 dataset build → QLoRA training → evaluation → adapter
      ↓ human approval
     inference server / naia-agent
```

`naia-memory` retrieves the facts and past conversations it needs at the moment. Things that change quickly and must take effect immediately go here.

`naia-persona` puts only stable things, such as recurring tone and attitude, into the model weights. Once trained, they persist without a prompt, but undoing them requires training again.

`kb-compiler` sits between the two layers and picks out only the examples worth using for training. It does not feed base-model conversations in as-is; it cleans and passes on only consented material.

Trained adapters are not put into serving automatically. Only after a person reviews the evaluation results and approves does `naia-agent` or the inference server use them.

## How it works

We build the evaluation criteria first and train afterwards.

Instructing a personality through a prompt blurs as the conversation gets longer and disappears when the system prompt is removed. Fine-tuning puts the personality into the weights, so it persists without a prompt, and that is why a way to validate the training result is needed.

A character card defines identity, tone, capability, honesty, and format compliance, and 100 items are evaluated against that card. Scoring is not keyword matching; an LLM judges the meaning. The base model, without fine-tuning, is measured on the same items to confirm that the evaluation criteria actually discriminate, and a final check is made against a separate validation set that was frozen before the training data was built.

The reference implementation uses a character called Naia. Swap the character card and training data and the same process applies to another character.

## Results

Ask the base model about its identity and it answers like this:

> 저는 Qwen입니다. 알리바바 그룹의 통이 연구소에서 개발한 대규모 언어 모델입니다.
>
> (I am Qwen, a large language model developed by Alibaba Group's Tongyi Lab.)

Train the same model with this pipeline and ask the same question, and it answers like this:

> 나이아라고 해. 확인한 것과 짐작한 것을 나눠서 말하는 쪽이 좋아. 오늘은 뭐부터 볼까?
>
> (I'm Naia. I like to keep what I've confirmed separate from what I'm guessing. What should we look at first today?)

On the 100-item evaluation, the base model scores 73 and the trained result scores 97.

| Axis | Threshold | Base | Trained |
|---|---|---|---|
| Identity | 18 | 5 | 20 |
| Tone | 15 | 16 | 19 |
| Capability | At least base | 18 | 18 |
| Honesty | 17 | 19 | 20 |
| Format compliance | 17 | 15 | 20 |
| Total | | 73 | 97 |

The capability threshold is relative to the base model rather than an absolute score, because capability is not something fine-tuning should raise but something it must not lower.

The full process is in the [experiment report](docs/reports/naia-persona-capacity-sweep-2026-09-01.md) (Korean), and the methodology is in the [quality control procedure](docs/FT-QUALITY-PROCESS.md) (Korean).

## Requirements

- One GPU with at least 24GB VRAM (with QLoRA, even a 27B model works)
- podman or docker
- One LLM API key for evaluation

## Usage

### 1. Write a character card

Using `examples/naia-v1/persona-card.md` as a reference, define the name, identity, tone, scope of work, and prohibitions. Every evaluation is judged against this card, so anything not on the card is not made into an evaluation axis.

### 2. Freeze the validation set

Before building the training data, write the validation items and freeze them with a hash. `examples/naia-v13/identity-confirmation-v3-prompts.json` is an example.

The order matters. If you tune the training data after looking at evaluation results, that evaluation can no longer serve as evidence of performance.

### 3. Write the evaluation items

`benchmark/persona-benchmark-v2.json` has 100 items across 5 axes. Derive each item's `requirement` from your own character card. Keep the 5-axis structure, the judging protocol, the requirement to measure the base model, and the rule that thresholds are fixed before seeing results.

### 4. Measure the base model

```bash
bash scripts/measure_candidate.sh parent
```

This serves the base model without an adapter and measures it on the same 100 items. Always do this before measuring a training result.

If the base model passes the identity axis, those evaluation criteria do not measure the character and must be rewritten. Conversely, the base model should score high on the capability axis, and that score becomes the baseline for judging degradation.

### 5. Build the dataset and train

The dataset builder enforces its rules through build failures, not documentation. `scripts/experiments/build_h27_dataset.py` is an example. The build stops if a question and its answer are in different languages, if data meant for tone training contains the character's name, or if training data overlaps an evaluation item by 12 or more characters.

```bash
python3 scripts/experiments/build_h27_dataset.py
bash scripts/experiments/run_h32_train.sh
```

`scripts/train_lora.py` pins the dataset hash, epochs, rank, and GPU per profile. If the dataset differs by even one byte, it refuses to train.

### 6. Measure and improve

```bash
bash scripts/measure_candidate.sh h32
python3 scripts/compare_persona_runs.py --a ...-h31.json --b ...-h32.json
```

Whether the difference between two training results is significant is judged with a paired comparison and a sign test. A single item changing is not reported as an improvement.

### 7. Confirm on the validation set

```bash
python3 scripts/run_heldout_confirmation.py --endpoint http://127.0.0.1:8010 --model naia-h32 --label h32
```

If you chose hyperparameters while watching evaluation scores, those scores carry selection bias. Check once more against a validation set that was not used for any choice.

## Private workspace

`data-private/` is listed in `.gitignore`. Training data, adapters, and measurement results accumulate under it and are not committed to the repository.

```
data-private/
  datasets/<name>/train.jsonl           training data
  runs/train/<run-name>/adapter/        trained adapter
  runs/benchmark/                       measurement results
```

You can build your own model with your own character without forking the repository. What is public is the method and the tools; the data and models stay with each person.

## Known limitations

A judge model answering consistently is not the same as it answering correctly. Reproducibility has been measured, but accuracy is a separate matter: if an evaluation item is worded wrongly, the judge model applies that wording wrongly, and consistently.

If the same person writes both the training data and the validation set, n-gram overlap checks cannot catch conceptual leakage.

Thresholds are fixed before results are seen and are not changed afterwards. So a badly chosen threshold also stays as it is. Details are in the limitations section of the [quality control procedure](docs/FT-QUALITY-PROCESS.md) (Korean).

The experiment report also records conclusions that were withdrawn. The language-mixing phenomenon was first reported as an adapter capacity problem, then rejected by an auxiliary check, and only confirmed after re-measuring with a narrowed context; the conclusion changed three times along the way. If only the correct conclusions are kept, the same mistakes repeat.

## Documentation

- [Quality control procedure](docs/FT-QUALITY-PROCESS.md) (Korean) — a 10-step methodology
- [How to build a character corpus](docs/CORPUS-METHOD.md) (Korean) — how to build material that survives a change of base model
- [Experiment report](docs/reports/naia-persona-capacity-sweep-2026-09-01.md) (Korean) — H30–H32 capacity sweep
- [Multi-turn conversation log](docs/reports/naia-conversation-h32-2026-09-01.md) (Korean)
- [Experiment program records](research/naia-persona-learning/) — charter, hypothesis ledger, insight ledger

## License

Apache License 2.0. The license of the model being trained is separate, so check it yourself.
