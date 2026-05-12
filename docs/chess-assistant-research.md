# Chess Assistant Research Notes

The chess assistant is deliberately not a raw language model playing moves from memory. Chess is deterministic, so legal move generation and tactical strength need a rules/engine layer.

## Design Decisions

- Use a laptop-sized transformer for user-facing explanation, not as the only source of chess truth.
- Use Stockfish as the default open-source teacher and runtime engine when present.
- Keep a pure-Python legal fallback so the example is still usable after clone.
- Convert images to FEN before analysis; the rest of the system only trusts legal FEN and legal moves.
- Generate SFT examples from engine-grounded positions and train a LoRA adapter.
- Commit the compact adapter and the exact training data/config so another user can reproduce or continue the fine-tune.
- Benchmark the deployed move policy against a Stockfish 2000 profile. This measures the assistant that users play against: engine move selection plus transformer wording.

## Open Resources

- [Stockfish](https://stockfishchess.org/Stockfish): GPL open-source UCI engine and teacher.
- [python-chess](https://pypi.org/project/chess/): legal move generation and UCI engine integration.
- [SmolLM2-360M-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-360M-Instruct): Apache-licensed transformer target for the default wording adapter.
- [SmolLM2-360M-Instruct GGUF variants](https://hf.co/models?search=SmolLM2-360M-Instruct-GGUF): quantized runtime references for constrained machines.
- [Lichess chess-puzzles](https://hf.co/datasets/Lichess/chess-puzzles): CC0 puzzle-scale external dataset reference reviewed for future expansion.
- [ROOK-CLF-9m](https://hf.co/jrahn/ROOK-CLF-9m): small transformer chess move-prediction reference.
- [ChessFENS](https://hf.co/datasets/Maxlegrec/ChessFENS): large FEN/policy dataset reference for future scaling.
- [Grandmaster-Level Chess Without Search](https://hf.co/papers/2402.04494): evidence that supervised transformer chess policy can work at scale, while this repo keeps search/engine grounding for laptop reliability.
- [Chess as a Testbed for Language Model State Tracking](https://hf.co/papers/2102.13249): evidence that board state and legal state tracking matter.

## Reliability Rule

The assistant must never invent a move. All proposed moves are parsed through `python-chess` and come from a legal move list. When Stockfish is installed, the engine agent provides the recommendation; otherwise the fallback agent still chooses legal tactical moves. Elo claims require `--require-stockfish` so benchmark runs fail instead of silently downgrading to the fallback.

## Committed Evidence

- Dataset: 2,500 Stockfish-grounded chat rows with 2,500 unique FENs.
- Standard eval: legal move rate 1.0, move mention rate 1.0, rendered image roundtrip rate 1.0.
- Elo benchmark: 3.0/4 against a Stockfish 2000 profile, estimated rating floor 2191, pass true.
