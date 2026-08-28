# Naia v1 reference experiment

This is the public reference persona for `naia-persona`. It demonstrates how to turn a character specification into synthetic SFT examples, compile immutable splits, train a candidate LoRA, and compare it with its exact parent model.

The character source is [Naia — The Liquid Cat](https://www.naia.land/en/naia). The dataset contains newly written examples rather than copied site text or real conversations. See [persona-card.md](./persona-card.md) for the behavioral contract.

```bash
python3 examples/naia-v1/build_dataset.py
python3 scripts/validate_dataset.py examples/naia-v1/source.jsonl
python3 scripts/compile_dataset.py examples/naia-v1/source.jsonl \
  data-private/datasets/naia-v1 --dataset-name naia-v1 --seed naia-v1
```

`eval-prompts.json` is written independently from the training prompts. It checks persona identity and boundaries alongside arithmetic, factual, instruction-following, uncertainty, and safety cases. The compiler holdout verifies deterministic dataset handling; model promotion uses this independent suite to avoid treating near-duplicate training variants as evidence of generalization.

To adapt this example, copy the directory, replace the character card and synthetic cases, keep evaluation prompts independent from training wording, and require a positive persona gain without general-capability regression. Generated adapters, model caches, responses, and private datasets belong under `data-private/` and must not be committed.
