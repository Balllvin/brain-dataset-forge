# OpenCode Go Setup

Brain Dataset Forge can call OpenCode Go through its OpenAI-compatible endpoint when live generation is enabled.

## Model Roles

- `light`: `opencode-go/deepseek-v4-flash`
- `medium`: `opencode-go/deepseek-v4-pro`
- `high`: `opencode-go/deepseek-v4-pro`
- `fallback_high`: `opencode-go/deepseek-v4-pro`

OpenCode Go's public model list includes the DeepSeek models but not every frontier model a user may have through another provider. Use a separate compatible endpoint before setting `high` to a model outside OpenCode Go.

## Environment

```bash
export OPENCODE_GO_API_KEY="your-key"
python -m dataset_forge run --config examples/transcript_lab.json --output outputs/live --count 60 --live-llm
```

## Cost Control

Use the light model for broad candidate generation, the medium model for rubric and deficit analysis, and the high slot only for final judge passes or hard failure triage.
