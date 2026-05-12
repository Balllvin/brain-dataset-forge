# Chess Assistant Example

This example turns Brain Dataset Forge into a small, open-source chess assistant that works as one user-facing assistant while using multiple internal agents:

- Vision agent: converts a clean top-down board image into FEN.
- Rules agent: validates FEN, legal moves, checks, game state, and material.
- Engine agent: uses Stockfish when installed, with a lightweight legal fallback.
- Language agent: can run a small transformer adapter for final wording, with deterministic engine-grounded fallback when model weights are absent.

The default small transformer target is `HuggingFaceTB/SmolLM2-135M-Instruct`. This repo includes a compact LoRA adapter trained on the generated engine-grounded chess conversations at `examples/chess_assistant/adapters/smollm2_chess_lora`.

## Install

```bash
python -m pip install -e ".[dev]"
brew install stockfish
```

Training the optional adapter needs the heavier training extra:

```bash
python -m pip install -e ".[chess-train]"
```

## Ask From FEN

```bash
dataset-forge-chess ask \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --question "What should I play and why?"
```

## Ask From An Image

Render a board image:

```bash
dataset-forge-chess render \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --output outputs/chess/start.png
```

Ask from that image:

```bash
dataset-forge-chess ask \
  --image outputs/chess/start.png \
  --question "What is this position and what should I play?"
```

For photos or arbitrary screenshots, use a clean top-down crop. Image-only chess cannot infer castling rights, en passant, or move counters, so pass FEN when those details matter.

## Browser App

```bash
dataset-forge-chess serve --port 8766
```

Open `http://127.0.0.1:8766`, paste a FEN or upload a board image, and ask a question.

## Generate Data

```bash
dataset-forge-chess make-data --output-dir outputs/chess_data --count 128
```

Outputs:

- `chess_assistant_sft.jsonl`
- `chess_assistant_train.jsonl`
- `chess_assistant_eval.jsonl`
- `chess_eval_suite.json`

## Train The Small Transformer Adapter

```bash
dataset-forge-chess plan-train \
  --dataset outputs/chess_data/chess_assistant_sft.jsonl \
  --output outputs/chess_adapter

dataset-forge-chess train \
  --dataset outputs/chess_data/chess_assistant_sft.jsonl \
  --output outputs/chess_adapter \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --max-steps 80
```

The chess policy remains engine-grounded for legality and strength. The transformer is responsible for fluent, user-facing explanations from the engine/rules context.

To use the included adapter in a Python integration, pass `LanguageConfig(adapter_path=Path("examples/chess_assistant/adapters/smollm2_chess_lora"))` and set `use_transformer=True`. The CLI and browser app keep deterministic wording by default so they work without downloading the base model.

## Evaluate

```bash
dataset-forge-chess eval --output-dir outputs/chess_eval
```

The eval checks legal move recommendations, move mention rate, rendered image roundtrip, and a basic match against a simple opponent.
