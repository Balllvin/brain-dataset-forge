# Research Map

This repo does not vendor external projects. It maps their proven ideas into a small local implementation and provides optional handoff files when a team wants to use the full tool.

## Hugging Face TRL

Relevant idea: `SFTTrainer` accepts conversational `messages` and prompt-completion formats, and can apply chat templates automatically for conversational data.

Implementation:

- `dataset_sft_messages.jsonl` stores system, user, and assistant messages.
- `dataset_prompt_completion.jsonl` stores a fallback prompt/completion view.
- `trainer_recipes/trl_sft_config.json` keeps train/eval split files explicit.

## Hugging Face Datasets

Relevant idea: JSONL is the most efficient JSON shape for row-oriented datasets, and split handling should remain explicit.

Implementation:

- All large row exports are JSONL.
- Train and eval files are separately emitted.
- The audit checks prompt overlap between train and eval records.

## Tinker And Tinker Cookbook

Relevant idea: keep local training logic simple while delegating distributed training to the training backend, and use benchmark/eval recipes as part of the loop.

Implementation:

- `trainer_recipes/tinker_supervised_plan.json` describes a Tinker supervised handoff.
- `eval_suite.json` preserves task-level behavior expectations for post-train evaluation.
- The iteration loop treats training feedback as a deficiency plan rather than a one-shot generation result.

## Distilabel

Relevant idea: synthetic data quality improves when generation and AI feedback are explicit pipeline stages.

Implementation:

- Generation, audit, and targeted regeneration are separate modules.
- Preference pairs are emitted so users can run judge, DPO, or rejection-sampling workflows.
- `scripts/bootstrap_tools.py` can clone Distilabel locally for deeper experimentation.

## Promptfoo

Relevant idea: prompt and model behavior should be tested locally and in CI with declarative assertions.

Implementation:

- `trainer_recipes/promptfoo.yaml` is generated for quick local eval checks.
- The HTML report exposes quality flags before a model is trained.

## OpenAI Evals And AutoEvals

Relevant idea: private evals encode the recurring failure patterns in a workflow.

Implementation:

- `eval_suite.json` is generated from held-out examples and behavior rubrics.
- The deficiency plan names missing or weak behavioral buckets.

## OpenCode Go

Relevant idea: route broad generation through cheaper models and reserve stronger models for judging and planning.

Implementation:

- `light` defaults to `opencode-go/deepseek-v4-flash`.
- `medium` defaults to `opencode-go/deepseek-v4-pro`.
- `high` is configurable for a frontier judge slot, with `deepseek-v4-pro` as the fallback when no compatible high provider is available.

## Additional Training And Evaluation Repos

Relevant idea: keep this repo small while making the full research stack easy to inspect locally.

Implementation:

- `scripts/bootstrap_tools.py` can clone TRL, Alignment Handbook, LM Evaluation Harness, Open Instruct, and LLM Course.
- The generated artifacts stay plain JSON/JSONL/YAML so those tools can consume them without custom adapters.
