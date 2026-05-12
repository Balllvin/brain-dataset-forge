from __future__ import annotations

import json
import random
from pathlib import Path

import chess

from dataset_forge.chess_assistant.datasets import SEED_POSITIONS
from dataset_forge.chess_assistant.engine import ChessEngineConfig, analyse_fen
from dataset_forge.chess_assistant.orchestrator import ChessAssistant, ChessAssistantConfig
from dataset_forge.chess_assistant.position import board_from_fen, material_balance
from dataset_forge.chess_assistant.vision import image_to_fen, render_board


def run_chess_eval(output_dir: Path, engine_config: ChessEngineConfig | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_config = engine_config or ChessEngineConfig(engine_path="")
    assistant = ChessAssistant(ChessAssistantConfig(engine=engine_config))
    cases = []
    for position_id, fen, theme in SEED_POSITIONS:
        response = assistant.answer("What should I play and why?", fen=fen)
        board = board_from_fen(fen)
        legal = _is_legal(board, response.engine.best_move_uci)
        cases.append(
            {
                "id": position_id,
                "theme": theme,
                "fen": fen,
                "engine": response.engine.engine_name,
                "best_move_uci": response.engine.best_move_uci,
                "legal": legal,
                "answer_mentions_move": response.engine.best_move_uci in response.answer,
                "answer_length": len(response.answer.split()),
            }
        )

    image_cases = _image_roundtrip_cases(output_dir)
    match = play_basic_match(chess.STARTING_FEN, engine_config=engine_config)
    report = {
        "case_count": len(cases),
        "legal_move_rate": round(sum(1 for case in cases if case["legal"]) / len(cases), 4),
        "move_mention_rate": round(sum(1 for case in cases if case["answer_mentions_move"]) / len(cases), 4),
        "image_roundtrip_rate": round(sum(1 for case in image_cases if case["roundtrip_ok"]) / len(image_cases), 4),
        "basic_match": match,
        "cases": cases,
        "image_cases": image_cases,
    }
    (output_dir / "chess_eval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def play_basic_match(
    fen: str,
    engine_config: ChessEngineConfig | None = None,
    max_plies: int = 24,
    seed: int = 17,
) -> dict[str, object]:
    rng = random.Random(seed)
    board = board_from_fen(fen)
    start_material = material_balance(board)
    moves: list[str] = []
    for ply in range(max_plies):
        if board.is_game_over():
            break
        if ply % 2 == 0:
            line = analyse_fen(board.fen(), engine_config)
            move = chess.Move.from_uci(line.best_move_uci)
        else:
            legal = list(board.legal_moves)
            captures = [move for move in legal if board.is_capture(move)]
            checks = []
            for move in legal:
                probe = board.copy(stack=False)
                probe.push(move)
                if probe.is_check():
                    checks.append(move)
            pool = captures or checks or legal
            move = rng.choice(pool)
        moves.append(board.san(move))
        board.push(move)
    final_material = material_balance(board)
    return {
        "start_fen": fen,
        "final_fen": board.fen(),
        "plies": len(moves),
        "moves_san": moves,
        "game_over": board.is_game_over(),
        "material_delta_for_white": final_material - start_material,
    }


def _image_roundtrip_cases(output_dir: Path) -> list[dict[str, object]]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for position_id, fen, _theme in SEED_POSITIONS[:4]:
        image_path = image_dir / f"{position_id}.png"
        render_board(fen, image_path)
        parsed = image_to_fen(image_path, side_to_move=fen.split()[1], castling=fen.split()[2], en_passant=fen.split()[3])
        cases.append(
            {
                "id": position_id,
                "image": str(image_path),
                "expected_board": fen.split()[0],
                "parsed_board": parsed.fen.split()[0],
                "confidence": parsed.confidence,
                "roundtrip_ok": parsed.fen.split()[0] == fen.split()[0],
            }
        )
    return cases


def _is_legal(board: chess.Board, move_uci: str) -> bool:
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return False
    return move in board.legal_moves
