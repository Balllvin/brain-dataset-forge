# Training Handoff

Brain Dataset Forge is responsible for data generation, audit, and handoff. Training backends are intentionally optional.

## Hugging Face TRL

Use:

- `dataset_sft_train.jsonl`
- `dataset_sft_eval.jsonl`
- `trainer_recipes/trl_sft_config.json`

Recommended smoke model:

- `Qwen/Qwen2.5-0.5B-Instruct`

The dataset uses conversational messages so a TRL trainer can apply the base model's chat template consistently.

## Tinker

Use:

- `trainer_recipes/tinker_supervised_plan.json`
- `dataset_sft_train.jsonl`
- `dataset_sft_eval.jsonl`

Set `TINKER_API_KEY` before running any Tinker recipe. The repo emits a plan rather than executing paid remote training by default.

## Promptfoo

Use:

- `eval_suite.json`
- `trainer_recipes/promptfoo.yaml`

Run Promptfoo after a trained or served model is available, then convert misses into new generation targets.
