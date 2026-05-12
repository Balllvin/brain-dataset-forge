from __future__ import annotations

import json
import math
import random
from pathlib import Path

import chess

from dataset_forge.chess_assistant.datasets import SEED_POSITIONS
from dataset_forge.chess_assistant.engine import ChessEngineConfig, StockfishBatchAnalyzer, analyse_fen
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


def run_elo_benchmark(
    output_dir: Path,
    assistant_config: ChessEngineConfig | None = None,
    opponent_elo: int = 2000,
    games: int = 4,
    max_plies: int = 160,
) -> dict[str, object]:
    if games < 2:
        raise ValueError("games must be at least 2 so the assistant plays both colors.")
    output_dir.mkdir(parents=True, exist_ok=True)

    assistant_config = assistant_config or ChessEngineConfig(depth=6, hash_mb=96, threads=1, require_engine=True)
    opponent_config = ChessEngineConfig(
        engine_path=assistant_config.engine_path,
        time_limit=assistant_config.time_limit,
        depth=assistant_config.depth,
        hash_mb=assistant_config.hash_mb,
        threads=assistant_config.threads,
        require_engine=True,
        limit_strength=True,
        uci_elo=opponent_elo,
    )

    games_out: list[dict[str, object]] = []
    with StockfishBatchAnalyzer(assistant_config) as assistant_engine, StockfishBatchAnalyzer(opponent_config) as opponent_engine:
        for game_index in range(games):
            assistant_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
            games_out.append(
                _play_engine_game(
                    assistant_engine=assistant_engine,
                    opponent_engine=opponent_engine,
                    assistant_color=assistant_color,
                    max_plies=max_plies,
                    game_index=game_index,
                    opponent_elo=opponent_elo,
                )
            )

    score = sum(float(game["assistant_score"]) for game in games_out)
    score_rate = score / games
    estimated_rating = _elo_from_score(opponent_elo, score_rate)
    report = {
        "opponent_elo": opponent_elo,
        "games": games,
        "assistant_score": score,
        "score_rate": round(score_rate, 4),
        "estimated_rating_floor": estimated_rating,
        "passed_elo_floor": estimated_rating >= opponent_elo and score_rate >= 0.5,
        "assistant_policy": "Stockfish-grounded move selection with transformer wording layer",
        "opponent_policy": "Stockfish UCI_LimitStrength profile when supported by the engine",
        "games_detail": games_out,
    }
    (output_dir / "chess_elo_benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
            move = _sample_basic_opponent_move(board, rng)
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


def _play_engine_game(
    assistant_engine: StockfishBatchAnalyzer,
    opponent_engine: StockfishBatchAnalyzer,
    assistant_color: chess.Color,
    max_plies: int,
    game_index: int,
    opponent_elo: int,
) -> dict[str, object]:
    board = chess.Board()
    moves: list[dict[str, object]] = []
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = assistant_engine if board.turn == assistant_color else opponent_engine
        line = engine.analyse(board)
        move = chess.Move.from_uci(line.best_move_uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"Engine returned illegal move {line.best_move_uci} for {board.fen()}")
        moves.append(
            {
                "ply": ply + 1,
                "side": "assistant" if board.turn == assistant_color else f"stockfish_{opponent_elo}",
                "move_san": board.san(move),
                "move_uci": move.uci(),
                "score_cp": line.score_cp,
            }
        )
        board.push(move)

    result = board.result(claim_draw=True)
    adjudication = None
    if result == "*":
        assistant_score, adjudication = _adjudicate_unfinished_game(board, assistant_color, assistant_engine)
    else:
        assistant_score = _assistant_score(result, assistant_color)
    return {
        "game": game_index + 1,
        "assistant_color": "white" if assistant_color == chess.WHITE else "black",
        "result": result,
        "adjudication": adjudication,
        "assistant_score": assistant_score,
        "plies": len(moves),
        "final_fen": board.fen(),
        "moves": moves,
    }


def _assistant_score(result: str, assistant_color: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if assistant_color == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if assistant_color == chess.BLACK else 0.0
    return 0.5


def _adjudicate_unfinished_game(
    board: chess.Board,
    assistant_color: chess.Color,
    engine: StockfishBatchAnalyzer,
) -> tuple[float, dict[str, object]]:
    line = engine.analyse(board)
    score_cp = line.score_cp or 0
    white_cp = score_cp if board.turn == chess.WHITE else -score_cp
    assistant_cp = white_cp if assistant_color == chess.WHITE else -white_cp
    if assistant_cp >= 200:
        score = 1.0
    elif assistant_cp <= -200:
        score = 0.0
    else:
        score = 0.5
    return score, {
        "reason": "max_plies_reached_engine_score",
        "assistant_score_cp": assistant_cp,
        "threshold_cp": 200,
        "best_move_uci": line.best_move_uci,
    }


def _elo_from_score(opponent_elo: int, score_rate: float) -> int:
    if score_rate <= 0:
        return opponent_elo - 800
    if score_rate >= 1:
        return opponent_elo + 800
    delta = -400 * math.log10((1 / score_rate) - 1)
    return int(round(opponent_elo + delta))


def _sample_basic_opponent_move(board: chess.Board, rng: random.Random) -> chess.Move:
    legal = list(board.legal_moves)
    captures = [move for move in legal if board.is_capture(move)]
    checks = []
    for move in legal:
        probe = board.copy(stack=False)
        probe.push(move)
        if probe.is_check():
            checks.append(move)
    pool = captures or checks or legal
    return rng.choice(pool)


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
