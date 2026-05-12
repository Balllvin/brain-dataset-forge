# Chess Assistant Eval Artifacts

These artifacts verify the committed chess assistant example.

- `chess_eval_report.json`: legality, answer grounding, image roundtrip, and a basic match.
- `chess_elo_benchmark.json`: deployed move-policy match against a Stockfish 2000 profile.
- `images/`: rendered board fixtures used by the image roundtrip checks.

Current committed results:

- legal move rate: 1.0
- move mention rate: 1.0
- image roundtrip rate: 1.0
- Elo benchmark score: 3.0/4
- estimated rating floor: 2191
- passed 2000 floor: true

The Elo report measures the assistant users play against: Stockfish-grounded move selection with the transformer responsible for wording.
