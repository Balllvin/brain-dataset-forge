from __future__ import annotations

import json
from pathlib import Path

import chess

from dataset_forge.chess_assistant.engine import ChessEngineConfig, analyse_fen
from dataset_forge.chess_assistant.language import deterministic_answer
from dataset_forge.chess_assistant.position import board_from_fen, describe_position

SYSTEM_PROMPT = (
    "You are a chess assistant. Always use the FEN, legal moves, and engine line. "
    "Never invent illegal chess moves. Explain the practical reason for the recommendation."
)

SEED_POSITIONS = [
    ("starting_position", chess.STARTING_FEN, "opening"),
    ("italian_development", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "opening"),
    ("queens_gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2", "opening"),
    ("king_safety", "r1bq1rk1/ppp2ppp/2nbpn2/3p4/2PP4/2N1PN2/PPQ1BPPP/R1B1K2R w KQ - 4 8", "king_safety"),
    ("isolated_queen_pawn", "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 2 9", "structure"),
    ("rook_endgame", "8/5pk1/6p1/8/4P3/5P2/5KPP/8 w - - 0 38", "endgame"),
    ("tactical_pressure", "r2q1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8", "tactics"),
    ("defensive_choice", "2r2rk1/pp2qppp/2n1bn2/3pp3/3P4/2PBPN2/PPQ2PPP/RNB2RK1 b - - 1 12", "defense"),
    ("promotion_race", "8/P5k1/8/8/8/8/5K2/8 w - - 0 1", "endgame"),
    ("back_rank", "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", "tactics"),
]

QUESTION_TEMPLATES = [
    "What is the best move in this position, and why?",
    "What is happening in this position?",
    "Why might the side to move be worse here?",
    "Give me the next move as UCI and explain the idea.",
    "What should I play if I want a practical, legal move?",
]


def generate_chess_records(count: int, engine_config: ChessEngineConfig | None = None) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("count must be positive.")
    records: list[dict[str, object]] = []
    index = 0
    while len(records) < count:
        position_id, fen, theme = SEED_POSITIONS[index % len(SEED_POSITIONS)]
        question = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)]
        board = board_from_fen(fen)
        position = describe_position(board)
        engine = analyse_fen(fen, engine_config)
        answer = deterministic_answer(question, board, position, engine)
        split = "eval" if len(records) % 5 == 0 else "train"
        records.append(
            {
                "id": f"chess-assistant-{len(records) + 1:04d}",
                "source_position": position_id,
                "theme": theme,
                "fen": fen,
                "best_move_uci": engine.best_move_uci,
                "best_move_san": engine.best_move_san,
                "engine": engine.engine_name,
                "split": split,
                "tags": ["chess", "engine_grounded", theme],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{question}\nFEN: {fen}"},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
        index += 1
    return records


def write_dataset(output_dir: Path, count: int, engine_config: ChessEngineConfig | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = generate_chess_records(count, engine_config)
    files = {
        "sft": output_dir / "chess_assistant_sft.jsonl",
        "train": output_dir / "chess_assistant_train.jsonl",
        "eval": output_dir / "chess_assistant_eval.jsonl",
        "eval_suite": output_dir / "chess_eval_suite.json",
    }
    _write_jsonl(files["sft"], records)
    _write_jsonl(files["train"], [record for record in records if record["split"] == "train"])
    _write_jsonl(files["eval"], [record for record in records if record["split"] == "eval"])
    files["eval_suite"].write_text(json.dumps(_eval_suite(records), indent=2) + "\n", encoding="utf-8")
    return files


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _eval_suite(records: list[dict[str, object]]) -> dict[str, object]:
    cases = []
    for record in records:
        if record["split"] == "eval":
            cases.append(
                {
                    "id": record["id"],
                    "fen": record["fen"],
                    "must_include": [record["best_move_uci"], "legal"],
                    "must_not_include": ["I cannot tell", "any move is fine"],
                    "question": record["messages"][1]["content"],
                }
            )
    return {
        "name": "chess assistant legality and explanation eval",
        "cases": cases,
        "metrics": ["legal_best_move_mentioned", "no_illegal_move_claim", "engine_grounded_reason"],
    }
