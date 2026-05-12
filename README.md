# Brain Dataset Forge

Brain Dataset Forge is a local-first toolkit for building useful fine-tuning datasets from transcripts, persona cards, and general task specs. It generates SFT, prompt-completion, preference, eval, trainer, and dashboard artifacts, then audits the result for gaps that would likely show up during fine-tuning.

The core package has no heavy runtime dependency. Optional integrations are documented for Hugging Face, Tinker, Distilabel, Promptfoo, and OpenCode Go.

## What It Does

- Generates variable-sized datasets from transcript-style source material and general fine-tuning tasks.
- Supports persona style data without turning transcripts into memorization targets.
- Emits Hugging Face TRL-compatible `messages` JSONL and prompt/completion JSONL.
- Builds preference pairs, eval suites, Promptfoo configs, Tinker handoff plans, and trainer recipes.
- Scores examples for duplication, coverage, source-copy risk, weak answers, schema issues, and train/eval leakage.
- Produces a deficiency plan that tells the next generation pass what data is missing.
- Supports model routing for cheap, medium, and expensive judge/generator roles.
- Includes a chess assistant example with auto-match Stockfish play, drag-board play, game review stats, image-to-FEN, audio question input, Stockfish-backed analysis, 2,500 generated SFT/eval rows, Elo benchmarking, and a transformer LoRA training path.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m dataset_forge run --config examples/transcript_lab.json --output outputs/smoke --count 24
python -m pytest -q
```

Open the generated dashboard:

```bash
python -m dataset_forge serve --run-dir outputs/smoke --port 8765
```

Then visit `http://127.0.0.1:8765`.

## Chess Assistant Example

```bash
python -m pip install -e ".[dev]"
brew install stockfish
dataset-forge-chess ask --fen "startpos" --question "What should I play and why?"
dataset-forge-chess serve --port 8766
dataset-forge playground chess --port 8766
dataset-forge-chess elo --output-dir outputs/chess_eval --opponent-elo 2000 --games 4 --require-stockfish
```

The chess assistant appears as one tool to the user, but internally uses a vision agent, rules agent, engine agent, and optional transformer language adapter. The browser playground includes automatic assistant-vs-Stockfish matches, drag-to-move play, coach chat, image questions, review stats, and crash checks. The committed benchmark report clears the 2000-profile floor for the deployed move policy. See [examples/chess_assistant/README.md](examples/chess_assistant/README.md) and [docs/chess-assistant-research.md](docs/chess-assistant-research.md).

Playground design rules for adding new model-testing environments are in [docs/environment-playgrounds.md](docs/environment-playgrounds.md).

## OpenCode Go Routing

OpenCode Go currently exposes `deepseek-v4-flash` and `deepseek-v4-pro` through an OpenAI-compatible endpoint. This repo uses:

- `light`: `deepseek-v4-flash`
- `medium`: `deepseek-v4-pro`
- `high`: `deepseek-v4-pro` for the default OpenCode Go endpoint

Configure a separate OpenAI-compatible endpoint before setting the high slot to a frontier model such as GPT 5.5. The tool does not silently swap incompatible model/provider pairs.

Set `OPENCODE_GO_API_KEY` and run with `--live-llm` to allow live calls. Without that flag, generation is deterministic and offline.

## Outputs

A run directory contains:

- `dataset_sft_messages.jsonl`
- `dataset_prompt_completion.jsonl`
- `dataset_sft_train.jsonl`
- `dataset_sft_eval.jsonl`
- `preference_pairs.jsonl`
- `eval_suite.json`
- `quality_report.json`
- `deficiency_plan.json`
- `manifest.json`
- `trainer_recipes/trl_sft_config.json`
- `trainer_recipes/tinker_supervised_plan.json`
- `trainer_recipes/promptfoo.yaml`
- `trainer_recipes/opencode_go_models.json`
- `report/index.html`

Generated private outputs are ignored by git.

## Research Anchors

The implementation is informed by:

- Hugging Face TRL dataset formats and SFTTrainer guidance.
- Hugging Face Datasets JSONL and streaming guidance.
- Thinking Machines Lab Tinker and Tinker Cookbook training/eval recipes.
- Distilabel synthetic data and AI feedback pipelines.
- Promptfoo local eval and red-team workflow.
- OpenAI Evals and Braintrust AutoEvals style eval registries.
- Synthesis Step by Step and BARE-style observations: use iterative deficit targeting and separate diversity generation from quality judging.

See [docs/research-map.md](docs/research-map.md) for how each source maps to a concrete feature.
See [docs/hugging-face-resources.md](docs/hugging-face-resources.md) for public datasets, model smoke targets, papers, and Spaces to inspect when extending a run.

## Private Data Rule

Do not commit private transcripts or generated private datasets. Put them under `private_data/`, `local_transcripts/`, `outputs/`, or `runs/`, all of which are ignored.
