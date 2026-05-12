from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EngineLine:
    best_move_uci: str
    best_move_san: str
    score_cp: int | None
    mate_in: int | None
    principal_variation: list[str]
    engine_name: str


@dataclass(frozen=True, slots=True)
class PositionContext:
    fen: str
    legal_moves: list[str]
    side_to_move: str
    is_check: bool
    status: str
    material_balance: int
    board_ascii: str


@dataclass(frozen=True, slots=True)
class VisionResult:
    fen: str
    confidence: float
    square_confidence: dict[str, float]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChessAssistantResponse:
    question: str
    answer: str
    fen: str
    position: PositionContext
    engine: EngineLine
    vision: VisionResult | None = None
    image_path: Path | None = None
    used_transformer: bool = False

    def to_record(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "fen": self.fen,
            "position": {
                "legal_moves": self.position.legal_moves,
                "side_to_move": self.position.side_to_move,
                "is_check": self.position.is_check,
                "status": self.position.status,
                "material_balance": self.position.material_balance,
                "board_ascii": self.position.board_ascii,
            },
            "engine": {
                "best_move_uci": self.engine.best_move_uci,
                "best_move_san": self.engine.best_move_san,
                "score_cp": self.engine.score_cp,
                "mate_in": self.engine.mate_in,
                "principal_variation": self.engine.principal_variation,
                "engine_name": self.engine.engine_name,
            },
            "vision": None
            if self.vision is None
            else {
                "fen": self.vision.fen,
                "confidence": self.vision.confidence,
                "warnings": self.vision.warnings,
            },
            "image_path": None if self.image_path is None else str(self.image_path),
            "used_transformer": self.used_transformer,
        }
