from __future__ import annotations

import chess

from dataset_forge.chess_assistant.types import PositionContext

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class ChessInputError(ValueError):
    """Raised when a chess position or move cannot be used safely."""


def board_from_fen(fen: str) -> chess.Board:
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise ChessInputError(f"Invalid FEN: {fen}") from error
    if board.status() != chess.STATUS_VALID:
        raise ChessInputError(f"FEN is not a valid chess position: {fen}")
    return board


def apply_user_move(fen: str, move_text: str) -> chess.Board:
    board = board_from_fen(fen)
    move_text = move_text.strip()
    try:
        move = board.parse_san(move_text)
    except ValueError:
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError as error:
            raise ChessInputError(f"Move is not SAN or UCI: {move_text}") from error
    if move not in board.legal_moves:
        raise ChessInputError(f"Illegal move for position: {move_text}")
    board.push(move)
    return board


def describe_position(board: chess.Board) -> PositionContext:
    legal_moves = [move.uci() for move in board.legal_moves]
    if board.is_checkmate():
        status = "checkmate"
    elif board.is_stalemate():
        status = "stalemate"
    elif board.is_insufficient_material():
        status = "draw by insufficient material"
    elif board.can_claim_draw():
        status = "draw claim available"
    else:
        status = "active"

    return PositionContext(
        fen=board.fen(),
        legal_moves=legal_moves,
        side_to_move="white" if board.turn == chess.WHITE else "black",
        is_check=board.is_check(),
        status=status,
        material_balance=material_balance(board),
        board_ascii=str(board),
    )


def material_balance(board: chess.Board) -> int:
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


def board_summary(board: chess.Board) -> str:
    context = describe_position(board)
    material = context.material_balance
    if material > 0:
        material_text = f"White is ahead by about {material / 100:.1f} pawns of material."
    elif material < 0:
        material_text = f"Black is ahead by about {abs(material) / 100:.1f} pawns of material."
    else:
        material_text = "Material is balanced."
    check_text = " The side to move is in check." if context.is_check else ""
    return f"{context.side_to_move.capitalize()} to move. {material_text}{check_text} Status: {context.status}."
