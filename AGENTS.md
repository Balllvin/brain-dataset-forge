# AGENTS.md

This repository is Brain Dataset Forge: a local-first toolkit for generating, auditing, and iterating fine-tuning datasets from transcripts, seed tasks, and persona cards.

## Project Rules

- Keep the core package installable without heavy ML dependencies.
- Put optional integrations behind explicit commands, extras, or configuration.
- Do not commit private transcripts, generated private datasets, API keys, or model outputs from paid providers.
- Keep JSONL exports compatible with Hugging Face TRL conversational and prompt-completion formats.
- Avoid broad abstractions; keep generation, scoring, export, and iteration behavior explicit.
- Never use the current date in filenames, DB object names, function names, tool names, or generated project identifiers.
- Add or update tests and docs whenever behavior, schema, setup, or command output changes.

## Verification

Before shipping:

- `python -m pytest -q`
- `python -m dataset_forge run --config examples/transcript_lab.json --output outputs/smoke --count 24`
- Inspect `outputs/smoke/report/index.html` in a browser when UI/report output changed.
