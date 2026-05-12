from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_forge.chess_assistant.engine import ChessEngineConfig, analyse_fen
from dataset_forge.chess_assistant.language import (
    LanguageConfig,
    TransformerResponder,
    acceptable_transformer_answer,
    build_prompt,
    deterministic_answer,
)
from dataset_forge.chess_assistant.position import apply_user_move, board_from_fen, describe_position
from dataset_forge.chess_assistant.types import ChessAssistantResponse, VisionResult
from dataset_forge.chess_assistant.vision import image_to_fen


@dataclass(frozen=True, slots=True)
class ChessAssistantConfig:
    engine: ChessEngineConfig = ChessEngineConfig()
    language: LanguageConfig = LanguageConfig()
    use_transformer: bool = False
    image_side_to_move: str = "w"
    image_castling: str = "-"
    image_orientation: str = "white"


class ChessAssistant:
    """Multi-agent chess assistant facade.

    The public API looks like one assistant. Internally, it runs a vision agent
    for images, a rules agent for legality, an engine agent for chess strength,
    and an optional small-transformer language agent for final wording.
    """

    def __init__(self, config: ChessAssistantConfig | None = None) -> None:
        self.config = config or ChessAssistantConfig()
        self.transformer = TransformerResponder(self.config.language)

    def answer(
        self,
        question: str,
        fen: str | None = None,
        image_path: Path | str | None = None,
        move: str | None = None,
    ) -> ChessAssistantResponse:
        if not question.strip():
            question = "What should I play and why?"
        vision: VisionResult | None = None
        if image_path is not None:
            vision = image_to_fen(
                image_path,
                side_to_move=self.config.image_side_to_move,
                castling=self.config.image_castling,
                orientation=self.config.image_orientation,
            )
            fen = vision.fen
        if fen is None:
            raise ValueError("Provide either a FEN string or an image path.")

        return self._answer_from_fen(question, fen, move=move, vision=vision, image_path=image_path)

    def answer_vision_result(
        self,
        question: str,
        vision: VisionResult,
        move: str | None = None,
    ) -> ChessAssistantResponse:
        return self._answer_from_fen(question, vision.fen, move=move, vision=vision)

    def _answer_from_fen(
        self,
        question: str,
        fen: str,
        move: str | None = None,
        vision: VisionResult | None = None,
        image_path: Path | str | None = None,
    ) -> ChessAssistantResponse:
        board = apply_user_move(fen, move) if move else board_from_fen(fen)
        position = describe_position(board)
        engine = analyse_fen(position.fen, self.config.engine)
        prompt = build_prompt(question, board, position, engine)

        generated = None
        if self.config.use_transformer:
            generated = self.transformer.generate(prompt)
            if generated and not acceptable_transformer_answer(generated, engine):
                generated = None
        answer = generated or deterministic_answer(question, board, position, engine, vision)
        return ChessAssistantResponse(
            question=question,
            answer=answer,
            fen=position.fen,
            position=position,
            engine=engine,
            vision=vision,
            image_path=None if image_path is None else Path(image_path),
            used_transformer=generated is not None,
        )
