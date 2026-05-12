# Chess Assistant Research Notes

The chess assistant is deliberately not a raw language model playing moves from memory. Chess is deterministic, so legal move generation and tactical strength need a rules/engine layer.

## Design Decisions

- Use a small transformer for user-facing explanation, not as the only source of chess truth.
- Use Stockfish as the default open-source teacher and runtime engine when present.
- Keep a pure-Python legal fallback so the example is still usable after clone.
- Convert images to FEN before analysis; the rest of the system only trusts validated FEN.
- Generate SFT examples from engine-grounded positions and train a small LoRA adapter.
- Commit the compact adapter and the exact training data/config so another user can reproduce or continue the fine-tune.

## Open Resources

- [Stockfish](https://stockfishchess.org/Stockfish): GPL open-source UCI engine and teacher.
- [python-chess](https://pypi.org/project/chess/): legal move generation and UCI engine integration.
- [SmolLM2-135M-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-135M-Instruct): small Apache-licensed transformer target.
- [ROOK-CLF-9m](https://hf.co/jrahn/ROOK-CLF-9m): small transformer chess move-prediction reference.
- [ChessFENS](https://hf.co/datasets/Maxlegrec/ChessFENS): large FEN/policy dataset reference for future scaling.
- [Grandmaster-Level Chess Without Search](https://hf.co/papers/2402.04494): evidence that supervised transformer chess policy can work at scale, while this repo keeps search/engine grounding for laptop reliability.
- [Chess as a Testbed for Language Model State Tracking](https://hf.co/papers/2102.13249): evidence that board state and legal state tracking matter.

## Reliability Rule

The assistant must never invent a move. All proposed moves are parsed through `python-chess` and come from a legal move list. When Stockfish is installed, the engine agent provides the recommendation; otherwise the fallback agent still chooses legal tactical moves.
