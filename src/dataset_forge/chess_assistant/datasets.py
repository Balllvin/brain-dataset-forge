from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import chess

from dataset_forge.chess_assistant.engine import ChessEngineConfig, StockfishBatchAnalyzer
from dataset_forge.chess_assistant.language import deterministic_answer
from dataset_forge.chess_assistant.position import board_from_fen, describe_position, material_balance

MIN_DATASET_COUNT = 1_000
DEFAULT_DATASET_COUNT = 2_500

SYSTEM_PROMPT = (
    "You are a chess assistant. Treat the FEN, legal moves, and engine line as the source of truth. "
    "Give one concrete legal recommendation and explain the practical chess idea."
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
    ("sicilian_center", "rnbqkb1r/pp3ppp/2p2n2/3pp3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq d6 0 5", "opening"),
    ("carlsbad_structure", "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1NBPPP/R2Q1RK1 w - - 4 10", "structure"),
    ("king_attack", "r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 w - - 0 8", "attack"),
    ("minor_piece_endgame", "8/5pk1/5np1/4p3/4P3/5PN1/5KPP/8 w - - 2 35", "endgame"),
]

QUESTION_TEMPLATES = [
    ("best_move", "What is the best move in this position, and why?"),
    ("position_read", "What is happening in this position?"),
    ("losing_reason", "Why might the side to move be worse here?"),
    ("uci", "Give me the next move as UCI and explain the idea."),
    ("practical", "What should I play if I want a practical, legal move?"),
    ("coach", "Coach me through the position. What matters most right now?"),
    ("threat", "What is the main threat and the move that handles it?"),
    ("plan", "What plan should the side to move follow from here?"),
    ("mistake", "What common mistake should I avoid in this position?"),
    ("eval", "Who is better, by how much, and what move proves it?"),
    ("tactics", "Is there a tactic here? Give the move and the reason."),
    ("endgame", "What is the cleanest endgame move and why does it work?"),
    ("defense", "How should I defend this position without collapsing?"),
    ("conversion", "How should I convert the advantage from this FEN?"),
    ("opening", "What opening move is most reliable here?"),
    ("human", "Explain the best move like you are talking to me during a game."),
    ("blunder_check", "What move should I not miss in this exact position?"),
    ("candidate", "Compare the candidate move to the engine move and choose one."),
    ("short", "Answer with the move first, then one sentence of reasoning."),
    ("followup", "I uploaded or described this board. What should I do next?"),
]


def generate_chess_records(
    count: int,
    engine_config: ChessEngineConfig | None = None,
    seed: int = 41,
) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("count must be positive.")

    config = engine_config or ChessEngineConfig()
    records: list[dict[str, object]] = []
    with StockfishBatchAnalyzer(config) as analyzer:
        for index, (position_id, board, theme, source_kind) in enumerate(_candidate_positions(count, seed=seed)):
            question_type, question = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)]
            engine = analyzer.analyse(board)
            position = describe_position(board)
            answer = deterministic_answer(question, board, position, engine)
            split = "eval" if len(records) % 10 == 0 else "train"
            records.append(
                {
                    "id": f"chess-assistant-{len(records) + 1:05d}",
                    "source_position": position_id,
                    "source_kind": source_kind,
                    "question_type": question_type,
                    "theme": theme,
                    "fen": position.fen,
                    "best_move_uci": engine.best_move_uci,
                    "best_move_san": engine.best_move_san,
                    "engine": engine.engine_name,
                    "split": split,
                    "tags": ["chess", "engine_grounded", "assistant_chat", theme, question_type],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{question}\nFEN: {position.fen}"},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "legal_move_count": len(position.legal_moves),
                        "material_balance_cp": position.material_balance,
                        "side_to_move": position.side_to_move,
                        "teacher_principal_variation": engine.principal_variation,
                    },
                }
            )
    return records


def write_dataset(
    output_dir: Path,
    count: int,
    engine_config: ChessEngineConfig | None = None,
    seed: int = 41,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = generate_chess_records(count, engine_config, seed=seed)
    files = {
        "sft": output_dir / "chess_assistant_sft.jsonl",
        "train": output_dir / "chess_assistant_train.jsonl",
        "eval": output_dir / "chess_assistant_eval.jsonl",
        "eval_suite": output_dir / "chess_eval_suite.json",
        "manifest": output_dir / "chess_dataset_manifest.json",
    }
    _write_jsonl(files["sft"], records)
    _write_jsonl(files["train"], [record for record in records if record["split"] == "train"])
    _write_jsonl(files["eval"], [record for record in records if record["split"] == "eval"])
    files["eval_suite"].write_text(json.dumps(_eval_suite(records), indent=2) + "\n", encoding="utf-8")
    files["manifest"].write_text(json.dumps(_manifest(records, count), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return files


def _candidate_positions(count: int, seed: int) -> Iterator[tuple[str, chess.Board, str, str]]:
    rng = random.Random(seed)
    seen: set[str] = set()

    for position_id, fen, theme in SEED_POSITIONS:
        board = board_from_fen(fen)
        seen.add(board.board_fen() + " " + str(board.turn))
        yield position_id, board, theme, "curated"
        if len(seen) >= count:
            return

    game_index = 0
    while len(seen) < count:
        board = chess.Board()
        target_plies = rng.randint(4, 76)
        for ply in range(target_plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(_sample_move(board, rng))
            if ply >= 3 and (ply % 3 == 0 or rng.random() < 0.18):
                key = board.board_fen() + " " + str(board.turn)
                if key in seen or board.is_game_over(claim_draw=True):
                    continue
                seen.add(key)
                yield f"synthetic_game_{game_index:04d}_ply_{ply + 1:02d}", board.copy(stack=False), _theme_for_board(board), "self_play"
                if len(seen) >= count:
                    return
        game_index += 1


def _sample_move(board: chess.Board, rng: random.Random) -> chess.Move:
    legal = list(board.legal_moves)
    captures = [move for move in legal if board.is_capture(move)]
    promotions = [move for move in legal if move.promotion]
    checks = []
    for move in legal:
        probe = board.copy(stack=False)
        probe.push(move)
        if probe.is_check():
            checks.append(move)
    if promotions and rng.random() < 0.75:
        return rng.choice(promotions)
    if captures and rng.random() < 0.42:
        return rng.choice(captures)
    if checks and rng.random() < 0.30:
        return rng.choice(checks)
    return rng.choice(legal)


def _theme_for_board(board: chess.Board) -> str:
    if board.is_check():
        return "defense"
    if any(move.promotion for move in board.legal_moves):
        return "endgame"
    if board.fullmove_number <= 12 and len(board.piece_map()) > 24:
        return "opening"
    if len(board.piece_map()) <= 12:
        return "endgame"
    if _has_tactical_candidate(board):
        return "tactics"
    if abs(material_balance(board)) >= 250:
        return "conversion"
    return "middlegame"


def _has_tactical_candidate(board: chess.Board) -> bool:
    for move in board.legal_moves:
        if board.is_capture(move):
            return True
        probe = board.copy(stack=False)
        probe.push(move)
        if probe.is_check():
            return True
    return False


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
                    "must_not_include": ["I cannot tell", "any move is fine", "illegal"],
                    "question": record["messages"][1]["content"],
                }
            )
    return {
        "name": "chess assistant legality and explanation eval",
        "cases": cases,
        "metrics": ["legal_best_move_mentioned", "no_illegal_move_claim", "engine_grounded_reason"],
    }


def _manifest(records: list[dict[str, object]], requested_count: int) -> dict[str, object]:
    themes = Counter(str(record["theme"]) for record in records)
    question_types = Counter(str(record["question_type"]) for record in records)
    engines = Counter(str(record["engine"]) for record in records)
    splits = Counter(str(record["split"]) for record in records)
    return {
        "requested_rows": requested_count,
        "rows": len(records),
        "minimum_required_rows": MIN_DATASET_COUNT,
        "meets_minimum": len(records) >= MIN_DATASET_COUNT,
        "unique_fens": len({str(record["fen"]) for record in records}),
        "splits": dict(sorted(splits.items())),
        "themes": dict(sorted(themes.items())),
        "question_types": dict(sorted(question_types.items())),
        "engines": dict(sorted(engines.items())),
        "teacher": "Stockfish when available, otherwise explicit simple-tactical fallback",
        "recommended_base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "dataset_sources": [
            "curated FEN seed positions",
            "deterministic python-chess self-play positions",
            "Stockfish principal variations",
            "Hugging Face Lichess/chess-puzzles was reviewed as the larger external puzzle source",
        ],
    }
