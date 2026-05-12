from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import chess

from dataset_forge.chess_assistant.engine import ChessEngineConfig, analyse_fen
from dataset_forge.chess_assistant.language import LanguageConfig
from dataset_forge.chess_assistant.orchestrator import ChessAssistant, ChessAssistantConfig
from dataset_forge.chess_assistant.position import ChessInputError, board_from_fen, describe_position
from dataset_forge.chess_assistant.vision import VisionInputError, image_bytes_to_fen

START_FEN = chess.STARTING_FEN


def serve_chess_assistant(
    host: str = "127.0.0.1",
    port: int = 8766,
    engine_config: ChessEngineConfig | None = None,
    language_config: LanguageConfig | None = None,
    use_transformer: bool = False,
) -> None:
    assistant = ChessAssistant(
        ChessAssistantConfig(
            engine=engine_config or ChessEngineConfig(),
            language=language_config or LanguageConfig(),
            use_transformer=use_transformer,
        )
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            body = CHESS_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                payload = self._json_payload()
                if self.path == "/api/ask":
                    response = _handle_ask(payload, assistant)
                elif self.path == "/api/move":
                    response = _handle_move(payload, assistant)
                elif self.path == "/api/engine-step":
                    response = _handle_engine_step(payload, assistant)
                elif self.path == "/api/review-game":
                    response = _handle_review_game(payload)
                elif self.path == "/api/reset":
                    response = _state_payload(START_FEN)
                else:
                    self.send_error(404)
                    return
                self._json_response(200, response)
            except (ChessInputError, VisionInputError, ValueError) as error:
                self._json_response(400, {"error": str(error)})
            except Exception as error:
                self._json_response(500, {"error": f"Chess assistant failed: {error}"})

        def _json_payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            return payload

        def _json_response(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving chess assistant at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Chess assistant server stopped")
    finally:
        server.server_close()


def _handle_ask(payload: dict[str, Any], assistant: ChessAssistant) -> dict[str, Any]:
    question = str(payload.get("question") or "What should I play and why?")
    fen = str(payload.get("fen") or START_FEN).strip()
    image_data = str(payload.get("image_base64") or "").strip()
    if image_data:
        image_bytes = _decode_image_data(image_data)
        vision = image_bytes_to_fen(
            image_bytes,
            side_to_move=str(payload.get("side_to_move") or "w"),
            castling=str(payload.get("castling") or "-"),
            en_passant=str(payload.get("en_passant") or "-"),
            orientation=str(payload.get("orientation") or "white"),
        )
        response = assistant.answer_vision_result(question, vision)
    else:
        response = assistant.answer(question, fen=fen)
    record = response.to_record()
    record.update(_state_payload(response.fen))
    return record


def _handle_move(payload: dict[str, Any], _assistant: ChessAssistant) -> dict[str, Any]:
    fen = str(payload.get("fen") or START_FEN).strip()
    move_text = str(payload.get("move") or "").strip()
    if not move_text:
        raise ValueError("Provide a move in SAN or UCI.")

    board = board_from_fen(fen)
    user_move = _parse_move(board, move_text)
    user_san = board.san(user_move)
    board.push(user_move)
    assistant_san = None
    assistant_uci = None

    if not board.is_game_over(claim_draw=True):
        assistant_move = _choose_assistant_policy_move(board)
        assistant_san = board.san(assistant_move)
        assistant_uci = assistant_move.uci()
        board.push(assistant_move)

    record = _state_payload(board.fen())
    record["answer"] = _play_mode_comment(board, user_san, assistant_san)
    record["played"] = {
        "user_move_san": user_san,
        "user_move_uci": user_move.uci(),
        "assistant_move_san": assistant_san,
        "assistant_move_uci": assistant_uci,
    }
    return record


def _handle_engine_step(payload: dict[str, Any], assistant: ChessAssistant) -> dict[str, Any]:
    fen = str(payload.get("fen") or START_FEN).strip()
    board = board_from_fen(fen)
    if board.is_game_over(claim_draw=True):
        return _state_payload(board.fen()) | {"game_over": True}

    actor = str(payload.get("actor") or ("assistant" if board.turn == chess.WHITE else "stockfish"))
    opponent_elo = int(payload.get("opponent_elo") or 2000)
    if actor == "assistant":
        move = _choose_assistant_policy_move(board)
        score_cp = None
    else:
        config = _actor_config(actor, assistant.config.engine, opponent_elo)
        line = analyse_fen(board.fen(), config)
        move = chess.Move.from_uci(line.best_move_uci)
        score_cp = line.score_cp
    if move not in board.legal_moves:
        raise RuntimeError(f"{actor} returned illegal move {move.uci()}")
    played = {
        "actor": actor,
        "move_san": board.san(move),
        "move_uci": move.uci(),
        "score_cp": score_cp,
        "best_move_uci": move.uci(),
        "best_move_san": board.san(move),
    }
    board.push(move)
    state = _state_payload(board.fen())
    state["played"] = played
    state["game_over"] = board.is_game_over(claim_draw=True)
    state["result"] = board.result(claim_draw=True)
    return state


def _handle_review_game(payload: dict[str, Any]) -> dict[str, Any]:
    moves = payload.get("moves") or []
    illegal_attempts = payload.get("illegal_attempts") or []
    if not isinstance(moves, list):
        raise ValueError("moves must be a list.")
    if not isinstance(illegal_attempts, list):
        raise ValueError("illegal_attempts must be a list.")

    board = chess.Board()
    captures = 0
    checks = 0
    replayed: list[dict[str, object]] = []
    for index, item in enumerate(moves, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"move {index} must be an object.")
        move_uci = str(item.get("move_uci") or "")
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ChessInputError(f"Move {index} is illegal for replay: {move_uci}")
        san = board.san(move)
        capture = board.is_capture(move)
        board.push(move)
        captures += 1 if capture else 0
        checks += 1 if board.is_check() else 0
        replayed.append(
            {
                "ply": index,
                "actor": str(item.get("actor") or _side_name(not board.turn)),
                "move_san": san,
                "move_uci": move_uci,
                "score_cp": item.get("score_cp"),
            }
        )

    result = board.result(claim_draw=True)
    assistant_color = chess.WHITE if str(payload.get("assistant_color") or "white") == "white" else chess.BLACK
    score = _assistant_score_from_result(result, assistant_color)
    return {
        "result": result,
        "final_fen": board.fen(),
        "plies": len(replayed),
        "moves": replayed,
        "captures": captures,
        "checks": checks,
        "illegal_attempt_count": len(illegal_attempts),
        "banned_moves": illegal_attempts,
        "assistant_score": score,
        "estimated_rating": _single_game_rating(score, int(payload.get("opponent_elo") or 2000)),
        "summary": _review_summary(result, len(replayed), captures, checks, len(illegal_attempts), score),
    }


def _actor_config(actor: str, base: ChessEngineConfig, opponent_elo: int) -> ChessEngineConfig:
    if actor == "stockfish":
        return ChessEngineConfig(
            engine_path=base.engine_path,
            time_limit=base.time_limit,
            depth=base.depth,
            hash_mb=base.hash_mb,
            threads=base.threads,
            require_engine=base.require_engine,
            limit_strength=True,
            uci_elo=opponent_elo,
        )
    return base


def _choose_assistant_policy_move(board: chess.Board) -> chess.Move:
    ranked = sorted(board.legal_moves, key=lambda move: move.uci())
    if not ranked:
        raise ChessInputError("No legal moves are available.")
    return max(ranked, key=lambda move: _policy_move_score(board, move))


def _policy_move_score(board: chess.Board, move: chess.Move) -> int:
    score = 0
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    if captured_piece and moving_piece:
        score += 10 * _piece_value(captured_piece.piece_type) - _piece_value(moving_piece.piece_type)
    if move.promotion:
        score += 80 + _piece_value(move.promotion)
    if board.gives_check(move):
        score += 25
    if board.is_castling(move):
        score += 18
    if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}:
        score += 12
    if moving_piece and moving_piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
        if chess.square_rank(move.from_square) in {0, 7}:
            score += 9

    board.push(move)
    if board.is_checkmate():
        score += 10000
    if board.is_stalemate():
        score -= 200
    board.pop()
    return score


def _piece_value(piece_type: chess.PieceType) -> int:
    return {
        chess.PAWN: 10,
        chess.KNIGHT: 30,
        chess.BISHOP: 32,
        chess.ROOK: 50,
        chess.QUEEN: 90,
        chess.KING: 0,
    }[piece_type]


def _play_mode_comment(board: chess.Board, user_san: str, assistant_san: str | None) -> str:
    position = describe_position(board)
    if assistant_san:
        return (
            f"You played {user_san}. I replied {assistant_san}. "
            f"{position.side_to_move.capitalize()} to move. Status: {position.status}. "
            "I will only reject moves that are illegal in the current position; legal moves are played first, then reviewed."
        )
    return (
        f"You played {user_san}. The game is {position.status}. "
        "I will only reject moves that are illegal in the current position; legal moves are played first, then reviewed."
    )


def _state_payload(fen: str) -> dict[str, Any]:
    board = board_from_fen(fen)
    position = describe_position(board)
    return {
        "fen": fen,
        "turn": position.side_to_move,
        "status": position.status,
        "legal_moves": position.legal_moves,
        "result": board.result(claim_draw=True),
    }


def _parse_move(board: chess.Board, move_text: str) -> chess.Move:
    try:
        move = board.parse_san(move_text)
    except ValueError:
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError as error:
            raise ChessInputError(f"Move is not SAN or UCI: {move_text}") from error
    if move not in board.legal_moves:
        raise ChessInputError(f"Illegal move for position: {move_text}")
    return move


def _assistant_score_from_result(result: str, assistant_color: chess.Color) -> float:
    if result == "1-0":
        return 1.0 if assistant_color == chess.WHITE else 0.0
    if result == "1/2-1/2":
        return 0.5
    if result == "0-1":
        return 1.0 if assistant_color == chess.BLACK else 0.0
    return 0.5


def _single_game_rating(score: float, opponent_elo: int) -> int:
    if score >= 1:
        return opponent_elo + 200
    if score <= 0:
        return opponent_elo - 200
    return opponent_elo


def _review_summary(result: str, plies: int, captures: int, checks: int, banned: int, score: float) -> str:
    rating_text = "won" if score == 1 else "drew" if score == 0.5 else "lost"
    return f"Assistant {rating_text} ({result}) over {plies} plies with {captures} captures, {checks} checks, and {banned} banned move attempts."


def _side_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _decode_image_data(value: str) -> bytes:
    if "," in value and value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=True)


CHESS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chess Playground</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #151515;
      --muted: #665f55;
      --line: #d6cbbb;
      --green: #1f6658;
      --red: #9d1c1c;
      --light-square: #efe3cb;
      --dark-square: #8fa37d;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    main { min-height: 100dvh; padding: 16px clamp(14px, 2vw, 28px); display: grid; gap: 12px; align-content: start; }
    nav { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 2px; }
    nav button { white-space: nowrap; background: #e5dccd; color: #171410; border: 1px solid #c9bca9; }
    nav button.active { background: var(--green); color: white; border-color: var(--green); }
    button { appearance: none; border: 0; border-radius: 6px; background: var(--green); color: white; padding: 10px 12px; font-weight: 800; cursor: pointer; }
    button.secondary { background: #e5dccd; color: #171410; border: 1px solid #c9bca9; }
    button.danger { background: var(--red); }
    button:disabled { opacity: 0.55; cursor: wait; }
    input, textarea, select { width: 100%; border: 1px solid #b9ad9a; border-radius: 6px; padding: 9px; font: inherit; background: white; color: var(--ink); }
    textarea { min-height: 110px; resize: vertical; }
    label { display: grid; gap: 6px; font-size: 12px; font-weight: 800; color: #312c25; text-transform: uppercase; letter-spacing: 0; }
    .page { display: none; }
    .page.active { display: grid; gap: 12px; }
    .arena { display: grid; grid-template-columns: minmax(540px, 1fr) minmax(340px, 390px); gap: 16px; align-items: start; }
    .two { display: grid; grid-template-columns: minmax(330px, 0.9fr) minmax(360px, 1fr); gap: 12px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 10px 28px rgba(22, 18, 14, 0.07); }
    .board-stage { display: grid; gap: 12px; justify-items: start; }
    .board-panel { background: transparent; border: 0; border-radius: 0; padding: 0; box-shadow: none; }
    .board { display: grid; grid-template-columns: repeat(8, 1fr); width: min(calc(100dvh - 174px), calc(100vw - 470px), 760px); aspect-ratio: 1; border: 2px solid #24211c; border-radius: 8px; overflow: hidden; background: #24211c; box-shadow: 0 18px 44px rgba(22, 18, 14, 0.16); }
    .board-controls { width: min(calc(100dvh - 174px), calc(100vw - 470px), 760px); background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; box-shadow: 0 10px 28px rgba(22, 18, 14, 0.07); }
    .side-panel { max-height: calc(100dvh - 92px); overflow: auto; }
    .square { position: relative; display: grid; place-items: center; border: 0; padding: 0; min-width: 0; aspect-ratio: 1; font-size: clamp(34px, 7dvh, 70px); line-height: 1; cursor: default; touch-action: manipulation; }
    .square[draggable="true"] { cursor: grab; }
    .square.light { background: var(--light-square); }
    .square.dark { background: var(--dark-square); }
    .square.last { box-shadow: inset 0 0 0 4px rgba(31, 102, 88, 0.38); }
    .square.dragging { outline: 4px solid #c9902e; outline-offset: -4px; }
    .piece { z-index: 1; user-select: none; }
    .piece.white { color: #fff8e8; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.7); -webkit-text-stroke: 0.7px rgba(0, 0, 0, 0.55); }
    .piece.black { color: #11100d; text-shadow: 0 1px 1px rgba(255, 255, 255, 0.24); }
    .square.selected { outline: 4px solid #c9902e; outline-offset: -4px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .metric { background: #f0e8da; border: 1px solid #d9cfbe; border-radius: 6px; padding: 9px; }
    .metric strong { display: block; font-size: 20px; line-height: 1.1; }
    .list { max-height: calc(100dvh - 130px); overflow: auto; display: grid; gap: 7px; }
    .move { padding: 8px; background: #f7f1e7; border: 1px solid #dfd4c2; border-radius: 6px; font-size: 13px; }
    .log { min-height: 260px; max-height: 520px; overflow: auto; display: grid; align-content: start; gap: 9px; }
    .message { border-radius: 7px; padding: 10px; line-height: 1.45; white-space: pre-wrap; overflow-wrap: anywhere; }
    .message.user { background: #e7f0e8; border: 1px solid #c7d9c7; }
    .message.assistant { background: #181816; color: #f8f3e8; }
    .message.system { background: #f2eadb; border: 1px solid #dacfbf; }
    .small { color: var(--muted); font-size: 12px; line-height: 1.35; }
    .hidden { display: none !important; }
    @media (max-width: 980px) {
      .arena, .two { grid-template-columns: 1fr; }
      .board, .board-controls { width: min(92vw, calc(100dvh - 180px), 760px); }
      .side-panel, .list { max-height: none; }
    }
  </style>
</head>
<body>
<main>
  <nav aria-label="Playgrounds">
    <button data-page-button="auto" class="active">Auto match</button>
    <button data-page-button="play">Play Stockfish</button>
    <button data-page-button="coach">Coach</button>
    <button data-page-button="image">Image</button>
    <button data-page-button="review">Review</button>
    <button data-page-button="crash">Crash lab</button>
  </nav>

  <section id="page-auto" class="page active">
    <div class="arena">
      <div class="board-stage">
        <div class="board-panel">
          <div id="auto-board" class="board" aria-label="Auto match board"></div>
        </div>
        <div class="board-controls">
        <div class="actions">
          <button id="auto-start">Start</button>
          <button id="auto-pause" class="secondary">Pause</button>
          <button id="auto-reset" class="secondary">Reset</button>
        </div>
        <div class="row" style="margin-top: 10px;">
          <label>Stockfish Elo <input id="auto-elo" type="text" inputmode="numeric" value="2000"></label>
          <label>Seconds per move <input id="auto-interval" type="text" inputmode="decimal" value="2"></label>
        </div>
        <div class="metrics" id="auto-metrics" style="margin-top: 10px;"></div>
        <p class="small" id="auto-summary" style="margin-top: 10px;">Press Start to let the assistant play Stockfish automatically.</p>
        </div>
      </div>
      <div class="panel side-panel">
        <div id="auto-moves" class="list"></div>
      </div>
    </div>
  </section>

  <section id="page-play" class="page">
    <div class="arena">
      <div class="board-stage">
        <div class="board-panel"><div id="play-board" class="board" aria-label="Drag board"></div></div>
        <div class="board-controls">
        <div class="actions">
          <button id="play-reset" class="secondary">Reset</button>
          <button id="play-review">Review game</button>
        </div>
        <p class="small" style="margin-top: 10px;">Drag a piece to move. Legal moves are played and reviewed; illegal attempts are tracked separately.</p>
        <div class="metrics" id="play-metrics" style="margin-top: 10px;"></div>
        </div>
      </div>
      <div class="panel side-panel"><div id="play-log" class="log"></div></div>
    </div>
  </section>

  <section id="page-coach" class="page">
    <div class="two">
      <div class="panel">
        <label>Question <textarea id="coach-question">I am watching this game. What matters now?</textarea></label>
        <div class="actions" style="margin-top: 10px;">
          <button id="coach-ask">Ask</button>
          <button id="coach-mic" class="secondary">Mic</button>
        </div>
      </div>
      <div class="panel"><div id="coach-log" class="log"></div></div>
    </div>
  </section>

  <section id="page-image" class="page">
    <div class="two">
      <div class="panel">
        <label>Board image <input id="image-file" type="file" accept="image/*"></label>
        <div class="row" style="margin-top: 10px;">
          <label>Side <select id="image-side"><option value="w">White</option><option value="b">Black</option></select></label>
          <label>Castling <input id="image-castling" value="-"></label>
        </div>
        <label style="margin-top: 10px;">Question <textarea id="image-question">What should I do from this board?</textarea></label>
        <button id="image-ask" style="margin-top: 10px;">Read image</button>
      </div>
      <div class="panel"><div id="image-log" class="log"></div></div>
    </div>
  </section>

  <section id="page-review" class="page">
    <div class="two">
      <div class="panel">
        <div class="actions">
          <button id="review-current">Review current game</button>
          <button id="review-sample" class="secondary">Load sample</button>
        </div>
        <div class="metrics" id="review-metrics" style="margin-top: 10px;"></div>
      </div>
      <div class="panel"><div id="review-log" class="log"></div></div>
    </div>
  </section>

  <section id="page-crash" class="page">
    <div class="two">
      <div class="panel">
        <div class="actions">
          <button id="crash-bad-fen" class="danger">Bad FEN</button>
          <button id="crash-illegal" class="danger">Illegal move</button>
          <button id="crash-recover">Recover</button>
        </div>
      </div>
      <div class="panel"><div id="crash-log" class="log"></div></div>
    </div>
  </section>
</main>

<script>
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const PIECES = { p:"♟", r:"♜", n:"♞", b:"♝", q:"♛", k:"♚", P:"♙", R:"♖", N:"♘", B:"♗", Q:"♕", K:"♔" };
const files = "abcdefgh";
const state = {
  auto: freshGame(),
  play: freshGame(),
  lastFen: START_FEN,
  autoTimer: null,
  dragFrom: null
};

function freshGame() {
  return { fen: START_FEN, moves: [], illegal: [], last: [], running: false, review: null };
}

document.querySelectorAll("[data-page-button]").forEach(button => {
  button.addEventListener("click", () => showPage(button.dataset.pageButton));
});
document.getElementById("auto-start").addEventListener("click", startAuto);
document.getElementById("auto-pause").addEventListener("click", pauseAuto);
document.getElementById("auto-reset").addEventListener("click", resetAuto);
document.getElementById("play-reset").addEventListener("click", resetPlay);
document.getElementById("play-review").addEventListener("click", () => reviewGame("play"));
document.getElementById("coach-ask").addEventListener("click", askCoach);
document.getElementById("coach-mic").addEventListener("click", recordCoach);
document.getElementById("image-ask").addEventListener("click", askImage);
document.getElementById("review-current").addEventListener("click", () => reviewGame("auto"));
document.getElementById("review-sample").addEventListener("click", loadSampleReview);
document.getElementById("crash-bad-fen").addEventListener("click", crashBadFen);
document.getElementById("crash-illegal").addEventListener("click", crashIllegal);
document.getElementById("crash-recover").addEventListener("click", crashRecover);

renderAll();

function showPage(name) {
  document.querySelectorAll("[data-page-button]").forEach(button => button.classList.toggle("active", button.dataset.pageButton === name));
  document.querySelectorAll(".page").forEach(page => page.classList.toggle("active", page.id === `page-${name}`));
}

function renderAll() {
  renderBoard("auto-board", state.auto);
  renderBoard("play-board", state.play, true);
  renderMoves("auto-moves", state.auto.moves);
  renderMetrics("auto-metrics", state.auto.review || liveStats(state.auto));
  renderMetrics("play-metrics", state.play.review || liveStats(state.play));
}

function renderBoard(id, game, draggable = false) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  const rows = game.fen.split(" ")[0].split("/");
  for (let rank = 8; rank >= 1; rank--) {
    const row = rows[8 - rank];
    let fileIndex = 0;
    for (const char of row) {
      if (/^\\d$/.test(char)) {
        for (let i = 0; i < Number(char); i++) addSquare(el, game, fileIndex++, rank, "", "", draggable);
      } else {
        addSquare(el, game, fileIndex++, rank, PIECES[char] || "", char, draggable);
      }
    }
  }
}

function addSquare(parent, game, fileIndex, rank, piece, pieceCode, draggable) {
  const square = `${files[fileIndex]}${rank}`;
  const button = document.createElement("button");
  button.type = "button";
  button.className = `square ${((fileIndex + rank) % 2 === 0) ? "dark" : "light"}`;
  if (game.last.includes(square)) button.classList.add("last");
  if (state.dragFrom === square) button.classList.add("selected");
  button.dataset.square = square;
  button.dataset.hasPiece = piece ? "true" : "false";
  button.draggable = draggable && Boolean(piece);
  button.setAttribute("aria-label", piece ? `${piece} on ${square}` : square);
  const pieceColor = pieceCode && pieceCode === pieceCode.toUpperCase() ? "white" : "black";
  button.innerHTML = `<span class="piece ${piece ? pieceColor : ""}" aria-hidden="true">${piece}</span>`;
  if (draggable) {
    button.addEventListener("dragstart", event => {
      state.dragFrom = square;
      event.dataTransfer.setData("text/plain", square);
      button.classList.add("dragging");
    });
    button.addEventListener("dragend", () => button.classList.remove("dragging"));
    button.addEventListener("dragover", event => event.preventDefault());
    button.addEventListener("drop", event => {
      event.preventDefault();
      playHumanMove(`${state.dragFrom || event.dataTransfer.getData("text/plain")}${square}`);
      state.dragFrom = null;
    });
    button.addEventListener("click", () => clickMove(square, Boolean(piece)));
  }
  parent.appendChild(button);
}

function clickMove(square, hasPiece) {
  if (!state.dragFrom) {
    if (!hasPiece) return;
    state.dragFrom = square;
    renderAll();
    return;
  }
  if (state.dragFrom === square) {
    state.dragFrom = null;
    renderAll();
    return;
  }
  const move = maybePromote(`${state.dragFrom}${square}`);
  state.dragFrom = null;
  playHumanMove(move);
}

async function playHumanMove(move) {
  const submittedMove = maybePromote(move);
  try {
    const data = await postJson("/api/move", { fen: state.play.fen, move: submittedMove });
    state.play.fen = data.fen;
    state.lastFen = data.fen;
    addPlayedPair(state.play, data);
    renderAll();
    addMessage("play-log", "assistant", summarizePair(data));
  } catch (error) {
    if (error.kind === "network") {
      addMessage("play-log", "system", error.message);
      return;
    }
    state.play.illegal.push({ move: submittedMove, reason: friendlyMoveError(error.message) });
    renderMetrics("play-metrics", liveStats(state.play));
    addMessage("play-log", "system", `Illegal attempt: ${submittedMove}. ${friendlyMoveError(error.message)}`);
  }
}

function maybePromote(move) {
  if (!move || move.length !== 4) return move;
  return ((move[1] === "7" && move[3] === "8") || (move[1] === "2" && move[3] === "1")) ? `${move}q` : move;
}

async function startAuto() {
  if (state.auto.running) return;
  state.auto.running = true;
  await autoStep();
}

function pauseAuto() {
  state.auto.running = false;
  if (state.autoTimer) clearTimeout(state.autoTimer);
  state.autoTimer = null;
}

function resetAuto() {
  pauseAuto();
  state.auto = freshGame();
  renderAll();
  document.getElementById("auto-summary").textContent = "Ready.";
}

function resetPlay() {
  state.play = freshGame();
  state.lastFen = START_FEN;
  document.getElementById("play-log").innerHTML = "";
  renderAll();
}

async function autoStep() {
  if (!state.auto.running) return;
  const turn = state.auto.fen.split(" ")[1];
  const actor = turn === "w" ? "assistant" : "stockfish";
  let data;
  try {
    data = await postJson("/api/engine-step", { fen: state.auto.fen, actor, opponent_elo: readElo() });
  } catch (error) {
    pauseAuto();
    document.getElementById("auto-summary").textContent = error.message;
    return;
  }
  if (data.played) applyAutoMove(data);
  renderAll();
  if (data.game_over || state.auto.moves.length >= 160) {
    state.auto.running = false;
    await reviewGame("auto");
    return;
  }
  state.autoTimer = setTimeout(autoStep, readMoveDelayMs());
}

function applyAutoMove(data) {
  state.auto.fen = data.fen;
  state.lastFen = data.fen;
  state.auto.moves.push(data.played);
  state.auto.last = [data.played.move_uci.slice(0, 2), data.played.move_uci.slice(2, 4)];
}

async function reviewGame(kind) {
  const game = state[kind];
  let review;
  try {
    review = await postJson("/api/review-game", {
      moves: game.moves,
      illegal_attempts: game.illegal,
      opponent_elo: readElo(),
      assistant_color: kind === "play" ? "black" : "white"
    });
  } catch (error) {
    if (kind === "auto") document.getElementById("auto-summary").textContent = error.message;
    addMessage(kind === "auto" ? "review-log" : "play-log", "system", error.message);
    return;
  }
  game.review = review;
  renderMetrics(kind === "auto" ? "auto-metrics" : "play-metrics", review);
  if (kind === "auto") document.getElementById("auto-summary").textContent = review.summary;
  renderReview(review);
}

async function askCoach() {
  const question = document.getElementById("coach-question").value || "What matters now?";
  addMessage("coach-log", "user", question);
  try {
    const data = await postJson("/api/ask", { fen: state.lastFen, question });
    addMessage("coach-log", "assistant", data.answer);
  } catch (error) {
    addMessage("coach-log", "system", error.message);
  }
}

function recordCoach() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    addMessage("coach-log", "system", "Audio input is not supported by this browser.");
    return;
  }
  const recognition = new Recognition();
  recognition.lang = "en-US";
  recognition.onresult = event => {
    document.getElementById("coach-question").value = event.results[0][0].transcript;
    askCoach();
  };
  recognition.onerror = event => addMessage("coach-log", "system", `Audio failed: ${event.error}`);
  recognition.start();
}

async function askImage() {
  const file = document.getElementById("image-file").files[0];
  if (!file) {
    addMessage("image-log", "system", "Choose an image first.");
    return;
  }
  try {
    const data = await postJson("/api/ask", {
      question: document.getElementById("image-question").value || "What should I do?",
      image_base64: await fileToDataUrl(file),
      side_to_move: document.getElementById("image-side").value,
      castling: document.getElementById("image-castling").value || "-"
    });
    state.lastFen = data.fen;
    addMessage("image-log", "assistant", data.answer);
  } catch (error) {
    addMessage("image-log", "system", error.message);
  }
}

async function crashBadFen() {
  try {
    await postJson("/api/ask", { fen: "bad fen", question: "Crash?" });
  } catch (error) {
    addMessage("crash-log", "system", `Handled bad FEN: ${error.message}`);
  }
}

async function crashIllegal() {
  try {
    await postJson("/api/move", { fen: START_FEN, move: "e2e5" });
  } catch (error) {
    addMessage("crash-log", "system", `Handled illegal move: ${error.message}`);
  }
}

function crashRecover() {
  state.lastFen = START_FEN;
  addMessage("crash-log", "assistant", "Recovered. The app is still responsive.");
}

function loadSampleReview() {
  const review = {
    result: "sample",
    plies: state.auto.moves.length,
    captures: liveStats(state.auto).captures,
    checks: liveStats(state.auto).checks,
    illegal_attempt_count: state.auto.illegal.length,
    estimated_rating: readElo(),
    summary: "Sample review loaded from the current auto-match moves.",
    moves: state.auto.moves,
    banned_moves: state.auto.illegal
  };
  renderReview(review);
}

function addPlayedPair(game, data) {
  game.moves.push({ actor: "friend", move_san: data.played.user_move_san, move_uci: data.played.user_move_uci, score_cp: null });
  if (data.played.assistant_move_uci) {
    game.moves.push({ actor: "assistant", move_san: data.played.assistant_move_san, move_uci: data.played.assistant_move_uci, score_cp: data.engine?.score_cp ?? null });
    game.last = [data.played.assistant_move_uci.slice(0, 2), data.played.assistant_move_uci.slice(2, 4)];
  } else {
    game.last = [data.played.user_move_uci.slice(0, 2), data.played.user_move_uci.slice(2, 4)];
  }
}

function summarizePair(data) {
  return data.answer;
}

function renderMoves(id, moves) {
  const el = document.getElementById(id);
  el.innerHTML = moves.map((move, index) => `<div class="move">${index + 1}. ${escapeHtml(move.actor)} ${escapeHtml(move.move_san)} <span class="small">${escapeHtml(move.move_uci)}</span></div>`).join("");
}

function liveStats(game) {
  return {
    result: game.moves.length ? "playing" : "ready",
    plies: game.moves.length,
    captures: game.moves.filter(move => String(move.move_san).includes("x")).length,
    checks: game.moves.filter(move => String(move.move_san).includes("+") || String(move.move_san).includes("#")).length,
    illegal_attempt_count: game.illegal.length,
    estimated_rating: readElo(),
    summary: ""
  };
}

function renderMetrics(id, data) {
  const el = document.getElementById(id);
  el.innerHTML = [
    ["Result", data.result || "ready"],
    ["Plies", data.plies ?? 0],
    ["Illegal", data.illegal_attempt_count ?? 0],
    ["Rating", data.estimated_rating || "-"]
  ].map(([label, value]) => `<div class="metric"><span class="small">${label}</span><strong>${value}</strong></div>`).join("");
}

function renderReview(review) {
  renderMetrics("review-metrics", review);
  const banned = (review.banned_moves || []).map(item => `${item.move}: ${item.reason}`).join("\\n") || "None";
  const moves = (review.moves || []).map(move => `${move.ply || ""}. ${move.actor} ${move.move_san}`).join("\\n");
  document.getElementById("review-log").innerHTML = "";
  addMessage("review-log", "assistant", `${review.summary}\\n\\nMoves\\n${moves}\\n\\nBanned moves\\n${banned}`);
}

function addMessage(id, role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  document.getElementById(id).appendChild(div);
  document.getElementById(id).scrollTop = document.getElementById(id).scrollHeight;
}

async function postJson(url, payload) {
  let response;
  try {
    response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  } catch (error) {
    console.error("Playground request could not reach the server.", { url, error });
    throw appError("network", "The local playground server is not reachable. The move was not submitted; restart the app server and try again.");
  }
  let data;
  try {
    data = await response.json();
  } catch (error) {
    console.error("Playground response was not JSON.", { url, status: response.status, error });
    throw appError("server", "The server returned an unreadable response. Refresh the playground and try again.");
  }
  if (!response.ok) throw appError("api", data.error || "The request failed. Check the position and try again.");
  return data;
}

function appError(kind, message) {
  const error = new Error(message);
  error.kind = kind;
  return error;
}

function friendlyMoveError(message) {
  if (message.includes("Illegal move")) return "That move is not legal from the current position.";
  if (message.includes("SAN or UCI")) return "That drop did not form a chess move. Pick one piece, then a destination square.";
  return message || "That move could not be played.";
}

function readElo() {
  return Math.max(1320, Math.min(3190, parseInt(document.getElementById("auto-elo")?.value || "2000", 10) || 2000));
}

function readMoveDelayMs() {
  const raw = Number.parseFloat(document.getElementById("auto-interval")?.value || "2");
  const seconds = Number.isFinite(raw) && raw > 0 ? raw : 2;
  return Math.max(200, seconds * 1000);
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}
</script>
</body>
</html>
"""
