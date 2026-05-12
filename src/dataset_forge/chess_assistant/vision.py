from __future__ import annotations

import io
from pathlib import Path

import chess
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

from dataset_forge.chess_assistant.position import board_from_fen
from dataset_forge.chess_assistant.types import VisionResult

LIGHT_SQUARE = (238, 238, 210)
DARK_SQUARE = (118, 150, 86)
WHITE_PIECE = (245, 245, 239)
BLACK_PIECE = (34, 35, 36)
WHITE_TEXT = (18, 18, 18)
BLACK_TEXT = (242, 242, 242)
TEMPLATE_SIZE = 64

PIECE_SYMBOLS = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


class VisionInputError(ValueError):
    """Raised when a chessboard image cannot be interpreted."""


def render_board(fen: str, output: Path | None = None, size: int = 512, orientation: str = "white") -> Image.Image:
    board = board_from_fen(fen)
    image = Image.new("RGB", (size, size), LIGHT_SQUARE)
    draw = ImageDraw.Draw(image)
    square = size // 8
    font = _load_font(max(20, int(square * 0.55)))
    for row in range(8):
        for col in range(8):
            square_name = _square_name(row, col, orientation)
            chess_square = chess.parse_square(square_name)
            x0 = col * square
            y0 = row * square
            is_light = (row + col) % 2 == 0
            draw.rectangle((x0, y0, x0 + square, y0 + square), fill=LIGHT_SQUARE if is_light else DARK_SQUARE)
            piece = board.piece_at(chess_square)
            if piece:
                _draw_piece(draw, piece.symbol(), x0, y0, square, font)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    return image


def image_to_fen(
    image_path: Path | str,
    side_to_move: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove_clock: int = 0,
    fullmove_number: int = 1,
    orientation: str = "white",
) -> VisionResult:
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as error:
        raise VisionInputError(f"Could not read chessboard image: {image_path}") from error
    return image_to_fen_from_image(image, side_to_move, castling, en_passant, halfmove_clock, fullmove_number, orientation)


def image_bytes_to_fen(
    image_bytes: bytes,
    side_to_move: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove_clock: int = 0,
    fullmove_number: int = 1,
    orientation: str = "white",
) -> VisionResult:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except OSError as error:
        raise VisionInputError("Uploaded image is not a readable board image.") from error
    return image_to_fen_from_image(image, side_to_move, castling, en_passant, halfmove_clock, fullmove_number, orientation)


def image_to_fen_from_image(
    image: Image.Image,
    side_to_move: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove_clock: int = 0,
    fullmove_number: int = 1,
    orientation: str = "white",
) -> VisionResult:
    if side_to_move not in {"w", "b"}:
        raise VisionInputError("side_to_move must be 'w' or 'b'.")
    normalized = _normalize_board_image(image)
    templates = _templates(TEMPLATE_SIZE)
    rows: list[list[str]] = [["" for _ in range(8)] for _ in range(8)]
    square_confidence: dict[str, float] = {}

    for row in range(8):
        for col in range(8):
            crop = normalized.crop((col * TEMPLATE_SIZE, row * TEMPLATE_SIZE, (col + 1) * TEMPLATE_SIZE, (row + 1) * TEMPLATE_SIZE))
            symbol, confidence = _classify_square(crop, templates[(row + col) % 2])
            square_name = _square_name(row, col, orientation)
            square_confidence[square_name] = round(confidence, 4)
            fen_rank = 8 - chess.square_rank(chess.parse_square(square_name))
            fen_file = chess.square_file(chess.parse_square(square_name))
            rows[fen_rank - 1][fen_file] = symbol

    board_part = _rows_to_fen(rows)
    fen = f"{board_part} {side_to_move} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
    board_from_fen(fen)
    confidence = round(sum(square_confidence.values()) / len(square_confidence), 4)
    warnings = []
    if castling == "-":
        warnings.append("Image input cannot infer castling rights; pass castling data when it matters.")
    if confidence < 0.78:
        warnings.append("Low image confidence; provide a FEN or a cleaner top-down board image.")
    return VisionResult(fen=fen, confidence=confidence, square_confidence=square_confidence, warnings=warnings)


def _normalize_board_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((TEMPLATE_SIZE * 8, TEMPLATE_SIZE * 8), Image.Resampling.BICUBIC)


def _templates(square_size: int) -> dict[int, dict[str, Image.Image]]:
    result: dict[int, dict[str, Image.Image]] = {}
    for parity, background in ((0, LIGHT_SQUARE), (1, DARK_SQUARE)):
        result[parity] = {}
        for symbol in PIECE_SYMBOLS:
            result[parity][symbol] = _square_template(symbol, square_size, background)
    return result


def _square_template(symbol: str, square_size: int, background: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (square_size, square_size), background)
    if symbol:
        draw = ImageDraw.Draw(image)
        font = _load_font(max(20, int(square_size * 0.55)))
        _draw_piece(draw, symbol, 0, 0, square_size, font)
    return image


def _classify_square(crop: Image.Image, templates: dict[str, Image.Image]) -> tuple[str, float]:
    scores = {
        symbol: ImageStat.Stat(ImageChops.difference(crop, template)).mean
        for symbol, template in templates.items()
    }
    symbol, mean_diff = min(scores.items(), key=lambda item: sum(item[1]))
    confidence = max(0.0, min(1.0, 1.0 - (sum(mean_diff) / (3 * 255))))
    return symbol, confidence


def _draw_piece(draw: ImageDraw.ImageDraw, symbol: str, x0: int, y0: int, square: int, font: ImageFont.ImageFont) -> None:
    is_white = symbol.isupper()
    piece_fill = WHITE_PIECE if is_white else BLACK_PIECE
    text_fill = WHITE_TEXT if is_white else BLACK_TEXT
    margin = int(square * 0.14)
    draw.ellipse((x0 + margin, y0 + margin, x0 + square - margin, y0 + square - margin), fill=piece_fill)
    label = symbol.upper()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        (x0 + (square - text_width) / 2, y0 + (square - text_height) / 2 - int(square * 0.04)),
        label,
        font=font,
        fill=text_fill,
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _square_name(row: int, col: int, orientation: str) -> str:
    if orientation not in {"white", "black"}:
        raise VisionInputError("orientation must be 'white' or 'black'.")
    if orientation == "white":
        file_index = col
        rank_index = 7 - row
    else:
        file_index = 7 - col
        rank_index = row
    return chess.square_name(chess.square(file_index, rank_index))


def _rows_to_fen(rows: list[list[str]]) -> str:
    fen_rows = []
    for row in rows:
        empty = 0
        parts: list[str] = []
        for symbol in row:
            if not symbol:
                empty += 1
            else:
                if empty:
                    parts.append(str(empty))
                    empty = 0
                parts.append(symbol)
        if empty:
            parts.append(str(empty))
        fen_rows.append("".join(parts))
    return "/".join(fen_rows)
