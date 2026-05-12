from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess

from dataset_forge.chess_assistant.position import board_summary
from dataset_forge.chess_assistant.types import EngineLine, PositionContext, VisionResult


@dataclass(frozen=True, slots=True)
class LanguageConfig:
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    adapter_path: Path | None = None
    max_new_tokens: int = 180
    temperature: float = 0.2


def build_prompt(question: str, board: chess.Board, position: PositionContext, engine: EngineLine) -> str:
    return (
        "You are a concise chess assistant. Use legal chess only. "
        "Ground every recommendation in the supplied FEN and engine line.\n\n"
        f"Question: {question}\n"
        f"FEN: {position.fen}\n"
        f"Board summary: {board_summary(board)}\n"
        f"Legal moves: {', '.join(position.legal_moves[:80])}\n"
        f"Engine best move: {engine.best_move_san} ({engine.best_move_uci})\n"
        f"Engine score centipawns from side to move: {engine.score_cp}\n"
        f"Principal variation: {' '.join(engine.principal_variation)}\n"
        "Answer:"
    )


class TransformerResponder:
    """Optional small-transformer responder.

    The dependency is loaded lazily so the chess assistant remains usable in a
    clean clone without downloading model weights. Use the training script to
    produce a local adapter, then pass its path here.
    """

    def __init__(self, config: LanguageConfig | None = None) -> None:
        self.config = config or LanguageConfig()
        self._pipeline = None

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, prompt: str) -> str | None:
        if not self.available():
            return None
        try:
            pipe = self._load_pipeline()
            result = pipe(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            return None
        if not result:
            return None
        text = str(result[0].get("generated_text", ""))
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip() or None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        if self.config.adapter_path:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            base = AutoModelForCausalLM.from_pretrained(self.config.base_model, low_cpu_mem_usage=True)
            model = PeftModel.from_pretrained(base, self.config.adapter_path)
            self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        else:
            self._pipeline = pipeline("text-generation", model=self.config.base_model, device=-1)
        return self._pipeline


def deterministic_answer(
    question: str,
    board: chess.Board,
    position: PositionContext,
    engine: EngineLine,
    vision: VisionResult | None = None,
) -> str:
    question_lower = question.lower()
    intro = board_summary(board)
    if board.is_game_over():
        return f"{intro} The game is already over, so there is no legal next move."

    move_clause = f"The move I recommend is {engine.best_move_san} ({engine.best_move_uci})."
    eval_clause = _score_text(engine)
    pv_clause = ""
    if engine.principal_variation:
        pv_clause = f" Main line: {' '.join(engine.principal_variation[:6])}."

    if "why" in question_lower or "losing" in question_lower:
        reason = _position_reason(board, position, engine)
        answer = f"{intro} {eval_clause} {reason} {move_clause}{pv_clause}"
    elif "move" in question_lower or "play" in question_lower or "next" in question_lower:
        answer = f"{intro} {move_clause} {eval_clause}{pv_clause} The point is {_move_point(board, engine)}"
    elif "position" in question_lower or "what is" in question_lower or "fen" in question_lower:
        answer = f"{intro} FEN: {position.fen}. {move_clause} {eval_clause}{pv_clause}"
    else:
        answer = f"{intro} {move_clause} {eval_clause}{pv_clause} {_move_point(board, engine)}"

    if vision is not None:
        answer += f" I read the board image with confidence {vision.confidence:.2f}."
        if vision.warnings:
            answer += " " + " ".join(vision.warnings)
    return " ".join(answer.split())


def acceptable_transformer_answer(text: str, engine: EngineLine) -> bool:
    normalized = " ".join(text.split())
    if len(normalized.split()) < 12:
        return False
    if engine.best_move_uci and engine.best_move_uci not in normalized and engine.best_move_san not in normalized:
        return False
    words = normalized.lower().split()
    if len(words) >= 24:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.38:
            return False
    fourgrams = [tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3))]
    if fourgrams and len(set(fourgrams)) / len(fourgrams) < 0.72:
        return False
    return True


def _score_text(engine: EngineLine) -> str:
    if engine.mate_in:
        return f"The engine sees mate in {abs(engine.mate_in)}."
    if engine.score_cp is None:
        return "The evaluation is unavailable, so rely on the legal move recommendation."
    pawns = engine.score_cp / 100
    if abs(pawns) < 0.25:
        return "The position is roughly balanced."
    side = "side to move" if pawns > 0 else "opponent"
    return f"The engine evaluation is about {abs(pawns):.1f} pawns for the {side}."


def _position_reason(board: chess.Board, position: PositionContext, engine: EngineLine) -> str:
    if board.is_check():
        return "The first issue is immediate king safety: you must answer the check legally."
    if position.material_balance > 250 and board.turn == chess.BLACK:
        return "White has a material cushion, so Black needs active counterplay rather than slow moves."
    if position.material_balance < -250 and board.turn == chess.WHITE:
        return "Black has a material cushion, so White needs forcing moves or simplification avoidance."
    if engine.score_cp is not None and engine.score_cp < -120:
        return "The tactical balance is bad for the side to move, so quiet improving moves are probably too slow."
    return "The main risk is letting the opponent improve for free; choose a move that creates a concrete threat or fixes the worst weakness."


def _move_point(board: chess.Board, engine: EngineLine) -> str:
    try:
        move = chess.Move.from_uci(engine.best_move_uci)
    except ValueError:
        return "it keeps the position legal and avoids inventing a move."
    if board.is_capture(move):
        return "it wins or removes material while staying tactically sound."
    board_after = board.copy(stack=False)
    board_after.push(move)
    if board_after.is_checkmate():
        return "it checkmates."
    if board_after.is_check():
        return "it gives check and forces the opponent to respond."
    if move.promotion:
        return "it promotes and changes the material balance immediately."
    return "it improves the position without stepping outside the legal move set."
