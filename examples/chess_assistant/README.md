# Chess Assistant Example

This example turns Brain Dataset Forge into an open-source chess assistant that works as one user-facing assistant while using multiple internal agents:

- Vision agent: converts a clean top-down board image into FEN.
- Rules agent: checks FEN, legal moves, checks, game state, and material.
- Engine agent: uses Stockfish when installed, with a lightweight legal fallback for clone-and-run demos.
- Language agent: can run a transformer LoRA adapter for final wording, with deterministic engine-grounded fallback when model weights are absent.

The default transformer target is `HuggingFaceTB/SmolLM2-360M-Instruct`, chosen because it is substantially stronger than the 135M smoke target while still fitting a 4 GB runtime budget when used as an adapter-backed wording model. Chess strength does not depend on raw language-model move guessing; the move policy is Stockfish/rules-grounded.

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

Open `http://127.0.0.1:8766`. The browser environment includes:

- clickable chess board with SAN/UCI move entry
- assistant reply move after the user plays
- chat tied to the current board state
- image upload that turns a clean board image into FEN
- microphone button that uses browser speech recognition for audio questions when the browser supports it

## Generate Data

```bash
dataset-forge-chess make-data --output-dir outputs/chess_data --count 2500 --require-stockfish --engine-depth 3
```

Outputs:

- `chess_assistant_sft.jsonl`
- `chess_assistant_train.jsonl`
- `chess_assistant_eval.jsonl`
- `chess_eval_suite.json`
- `chess_dataset_manifest.json`

The committed dataset in `examples/chess_assistant/data` contains 2,500 Stockfish-grounded rows, 2,250 train rows, 250 eval rows, 2,500 unique FENs, and 20 question types.

## Train The Small Transformer Adapter

```bash
dataset-forge-chess plan-train \
  --dataset outputs/chess_data/chess_assistant_sft.jsonl \
  --output outputs/chess_adapter

dataset-forge-chess train \
  --dataset outputs/chess_data/chess_assistant_sft.jsonl \
  --output outputs/chess_adapter \
  --base-model HuggingFaceTB/SmolLM2-360M-Instruct \
  --max-steps 80
```

The chess policy remains engine-grounded for legality and strength. The transformer is responsible for fluent, user-facing explanations from the engine/rules context.

To use the included adapter in a Python integration, pass `LanguageConfig(adapter_path=Path("examples/chess_assistant/adapters/smollm2_360m_chess_lora"))` and set `use_transformer=True`. The CLI and browser app keep deterministic wording by default so they work without downloading the base model.

## Evaluate

```bash
dataset-forge-chess eval --output-dir outputs/chess_eval --require-stockfish --engine-depth 3
dataset-forge-chess elo --output-dir outputs/chess_eval --opponent-elo 2000 --games 4 --plies 120 --engine-time 0.04 --require-stockfish
```

The committed eval artifacts are in `examples/chess_assistant/evals`:

- legal move rate: 1.0
- move mention rate: 1.0
- rendered image roundtrip rate: 1.0
- Elo benchmark: 3.0/4 against a Stockfish 2000 profile, estimated rating floor 2191, pass true

The Elo number is for the deployed assistant move policy: Stockfish-grounded move selection with a transformer wording layer.
