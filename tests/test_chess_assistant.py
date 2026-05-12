from __future__ import annotations

import json
from pathlib import Path

import chess

from dataset_forge.chess_assistant.cli import main as chess_main
from dataset_forge.chess_assistant.datasets import SEED_POSITIONS, write_dataset
from dataset_forge.chess_assistant.engine import ChessEngineConfig, analyse_fen
from dataset_forge.chess_assistant.eval import run_chess_eval
from dataset_forge.chess_assistant.language import acceptable_transformer_answer
from dataset_forge.chess_assistant.orchestrator import ChessAssistant, ChessAssistantConfig
from dataset_forge.chess_assistant.vision import image_to_fen, render_board


def test_seed_positions_are_valid() -> None:
    for _position_id, fen, _theme in SEED_POSITIONS:
        board = chess.Board(fen)
        assert board.status() == chess.STATUS_VALID


def test_engine_returns_legal_move_with_fallback() -> None:
    fen = chess.STARTING_FEN
    line = analyse_fen(fen, ChessEngineConfig(engine_path="", depth=2))
    assert chess.Move.from_uci(line.best_move_uci) in chess.Board(fen).legal_moves
    assert line.engine_name == "simple-tactical"


def test_assistant_answers_with_legal_engine_grounding() -> None:
    assistant = ChessAssistant(ChessAssistantConfig(engine=ChessEngineConfig(engine_path="", depth=2)))
    response = assistant.answer("What should I play and why?", fen=chess.STARTING_FEN)
    assert response.engine.best_move_uci in response.answer
    assert chess.Move.from_uci(response.engine.best_move_uci) in chess.Board(chess.STARTING_FEN).legal_moves
    assert response.position.status == "active"


def test_rendered_board_image_roundtrips_to_fen(tmp_path: Path) -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    image_path = tmp_path / "board.png"
    render_board(fen, image_path)
    parsed = image_to_fen(image_path, side_to_move="b", castling="KQkq")
    assert parsed.fen.split()[0] == fen.split()[0]
    assert parsed.confidence > 0.9


def test_dataset_and_eval_generation(tmp_path: Path) -> None:
    files = write_dataset(tmp_path / "data", count=12, engine_config=ChessEngineConfig(engine_path="", depth=2))
    records = [json.loads(line) for line in files["sft"].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 12
    assert records[0]["messages"][0]["role"] == "system"
    assert files["eval_suite"].exists()

    report = run_chess_eval(tmp_path / "eval", engine_config=ChessEngineConfig(engine_path="", depth=2))
    assert report["legal_move_rate"] == 1.0
    assert report["image_roundtrip_rate"] == 1.0


def test_chess_cli_ask_outputs_json(capsys) -> None:
    result = chess_main(["ask", "--fen", "startpos", "--engine-path", "", "--question", "What is the best move?"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["engine"]["best_move_uci"]
    assert payload["fen"] == chess.STARTING_FEN


def test_transformer_answer_gate_rejects_repetitive_ungrounded_text() -> None:
    line = analyse_fen(chess.STARTING_FEN, ChessEngineConfig(engine_path="", depth=2))
    bad = "The pawn is best because the pawn is best because the pawn is best because the pawn is best."
    good = f"The best move is {line.best_move_san} ({line.best_move_uci}) because it develops a piece and keeps every move legal."
    assert acceptable_transformer_answer(bad, line) is False
    assert acceptable_transformer_answer(good, line) is True
