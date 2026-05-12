from __future__ import annotations

import math
import shutil
from dataclasses import dataclass

import chess
import chess.engine

from dataset_forge.chess_assistant.position import PIECE_VALUES, board_from_fen
from dataset_forge.chess_assistant.types import EngineLine

CHECKMATE_SCORE = 100_000

PIECE_SQUARE_BONUS = {
    chess.PAWN: [0, 5, 8, 12, 18, 22, 28, 0],
    chess.KNIGHT: [-12, -4, 4, 8, 8, 4, -4, -12],
    chess.BISHOP: [-6, 2, 6, 8, 8, 6, 2, -6],
    chess.ROOK: [0, 2, 4, 6, 6, 4, 2, 0],
    chess.QUEEN: [-4, 0, 4, 6, 6, 4, 0, -4],
    chess.KING: [8, 4, 0, -4, -4, 0, 4, 8],
}


@dataclass(frozen=True, slots=True)
class ChessEngineConfig:
    engine_path: str | None = None
    time_limit: float = 0.08
    depth: int | None = None
    hash_mb: int = 64
    threads: int = 1


def analyse_fen(fen: str, config: ChessEngineConfig | None = None) -> EngineLine:
    board = board_from_fen(fen)
    if board.is_game_over():
        return _game_over_line(board)

    config = config or ChessEngineConfig()
    engine_path = None if config.engine_path == "" else (config.engine_path or shutil.which("stockfish"))
    if engine_path:
        try:
            return _stockfish_line(board, engine_path, config)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError):
            pass
    return _fallback_line(board, depth=config.depth or 2)


def _stockfish_line(board: chess.Board, engine_path: str, config: ChessEngineConfig) -> EngineLine:
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        _configure_stockfish(engine, config)
        limit = chess.engine.Limit(depth=config.depth) if config.depth else chess.engine.Limit(time=config.time_limit)
        info = engine.analyse(board, limit)
        pv = info.get("pv", [])
        best_move = pv[0] if pv else engine.play(board, limit).move
    if best_move is None:
        return _fallback_line(board, depth=2)
    pv_moves = [move.uci() for move in pv] if pv else [best_move.uci()]
    score = info.get("score")
    score_cp: int | None = None
    mate_in: int | None = None
    if score is not None:
        pov_score = score.pov(board.turn)
        mate_in = pov_score.mate()
        score_cp = pov_score.score(mate_score=CHECKMATE_SCORE)
    return EngineLine(
        best_move_uci=best_move.uci(),
        best_move_san=board.san(best_move),
        score_cp=score_cp,
        mate_in=mate_in,
        principal_variation=pv_moves,
        engine_name="stockfish",
    )


def _configure_stockfish(engine: chess.engine.SimpleEngine, config: ChessEngineConfig) -> None:
    options: dict[str, int] = {}
    if "Threads" in engine.options:
        options["Threads"] = config.threads
    if "Hash" in engine.options:
        options["Hash"] = config.hash_mb
    if options:
        engine.configure(options)


def _fallback_line(board: chess.Board, depth: int) -> EngineLine:
    best_move = max(board.legal_moves, key=lambda move: _move_score(board, move, depth))
    board_after = board.copy(stack=False)
    board_after.push(best_move)
    pv = [best_move.uci()]
    reply = _best_reply(board_after)
    if reply is not None:
        pv.append(reply.uci())
    return EngineLine(
        best_move_uci=best_move.uci(),
        best_move_san=board.san(best_move),
        score_cp=_evaluate(board_after) * (1 if board.turn == chess.WHITE else -1),
        mate_in=1 if board_after.is_checkmate() else None,
        principal_variation=pv,
        engine_name="simple-tactical",
    )


def _move_score(board: chess.Board, move: chess.Move, depth: int) -> int:
    board_after = board.copy(stack=False)
    board_after.push(move)
    if board_after.is_checkmate():
        return CHECKMATE_SCORE
    if depth <= 1:
        return _evaluate_for_side(board_after, board.turn)
    opponent_reply = _best_reply(board_after)
    if opponent_reply is None:
        return _evaluate_for_side(board_after, board.turn)
    board_after.push(opponent_reply)
    return _evaluate_for_side(board_after, board.turn)


def _best_reply(board: chess.Board) -> chess.Move | None:
    if board.is_game_over():
        return None
    return max(board.legal_moves, key=lambda move: _move_score(board, move, 1))


def _evaluate_for_side(board: chess.Board, color: chess.Color) -> int:
    score = _evaluate(board)
    return score if color == chess.WHITE else -score


def _evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -CHECKMATE_SCORE if board.turn == chess.WHITE else CHECKMATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        rank_index = rank if piece.color == chess.WHITE else 7 - rank
        value = PIECE_VALUES[piece.piece_type] + PIECE_SQUARE_BONUS[piece.piece_type][rank_index]
        score += value if piece.color == chess.WHITE else -value

    if board.is_check():
        score += -35 if board.turn == chess.WHITE else 35
    mobility = len(list(board.legal_moves))
    score += int(math.copysign(min(mobility, 40), 1 if board.turn == chess.WHITE else -1))
    return score


def _game_over_line(board: chess.Board) -> EngineLine:
    return EngineLine(
        best_move_uci="",
        best_move_san="",
        score_cp=0,
        mate_in=0 if board.is_checkmate() else None,
        principal_variation=[],
        engine_name="game-over",
    )
