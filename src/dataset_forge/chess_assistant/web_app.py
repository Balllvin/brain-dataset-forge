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


def _handle_move(payload: dict[str, Any], assistant: ChessAssistant) -> dict[str, Any]:
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
        line = analyse_fen(board.fen(), assistant.config.engine)
        assistant_move = chess.Move.from_uci(line.best_move_uci)
        if assistant_move not in board.legal_moves:
            raise RuntimeError(f"Engine returned illegal move {line.best_move_uci}")
        assistant_san = board.san(assistant_move)
        assistant_uci = assistant_move.uci()
        board.push(assistant_move)

    question = "Explain the current position after the last move."
    response = assistant.answer(question, fen=board.fen())
    record = response.to_record()
    record.update(_state_payload(board.fen()))
    record["played"] = {
        "user_move_san": user_san,
        "user_move_uci": user_move.uci(),
        "assistant_move_san": assistant_san,
        "assistant_move_uci": assistant_uci,
    }
    return record


def _state_payload(fen: str) -> dict[str, Any]:
    board = board_from_fen(fen)
    position = describe_position(board)
    return {
        "fen": fen,
        "turn": position.side_to_move,
        "status": position.status,
        "legal_moves": position.legal_moves,
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


def _decode_image_data(value: str) -> bytes:
    if "," in value and value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=True)


CHESS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chess Assistant</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f2eb;
      --panel: #fffdf8;
      --ink: #141414;
      --muted: #686157;
      --line: #d9d0c1;
      --green: #1e5b4f;
      --green-2: #2f7d6c;
      --amber: #c9902e;
      --dark-square: #8fa37d;
      --light-square: #ece2ce;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    main { max-width: 1360px; margin: 0 auto; padding: 22px; display: grid; gap: 16px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 14px; border-bottom: 1px solid var(--line); padding-bottom: 14px; }
    h1 { margin: 0; font-size: 30px; line-height: 1.05; letter-spacing: 0; }
    .status { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    .chip { border: 1px solid var(--line); background: #f2eadb; border-radius: 999px; padding: 6px 9px; font-size: 12px; font-weight: 800; }
    .layout { display: grid; grid-template-columns: minmax(310px, 0.86fr) minmax(390px, 1.12fr) minmax(330px, 0.9fr); gap: 16px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 12px 34px rgba(23, 20, 17, 0.08); }
    .board-panel { padding: 14px; }
    .board-shell { width: min(100%, 620px); margin: 0 auto; display: grid; gap: 10px; }
    .board { display: grid; grid-template-columns: repeat(8, 1fr); aspect-ratio: 1; border: 2px solid #332f2a; border-radius: 6px; overflow: hidden; background: #332f2a; }
    .square { position: relative; display: grid; place-items: center; min-width: 0; border: 0; padding: 0; color: #151515; font-size: clamp(26px, 5.2vw, 56px); line-height: 1; cursor: pointer; }
    .piece { position: relative; z-index: 1; }
    .piece.white { color: #fff8e8; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.65); -webkit-text-stroke: 0.7px rgba(0, 0, 0, 0.55); }
    .piece.black { color: #11100d; text-shadow: 0 1px 1px rgba(255, 255, 255, 0.24); }
    .square.light { background: var(--light-square); }
    .square.dark { background: var(--dark-square); }
    .square.selected { outline: 4px solid var(--amber); outline-offset: -4px; }
    .square.last { box-shadow: inset 0 0 0 4px rgba(30, 91, 79, 0.35); }
    .coord { position: absolute; left: 5px; bottom: 4px; color: rgba(20, 20, 20, 0.48); font-size: 10px; font-weight: 800; }
    .controls, .chat, .uploads { padding: 16px; display: grid; gap: 13px; }
    label { display: grid; gap: 7px; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0; color: #302c27; }
    textarea, input, select { width: 100%; border: 1px solid #bdb4a4; border-radius: 6px; padding: 10px; font: inherit; background: #fff; color: var(--ink); }
    textarea { min-height: 82px; resize: vertical; }
    button { appearance: none; border: 0; border-radius: 6px; background: var(--green); color: white; padding: 11px 13px; font-weight: 850; cursor: pointer; }
    button.secondary { background: #e7decd; color: #181511; border: 1px solid #cabfac; }
    button.icon { width: 42px; height: 42px; padding: 0; display: grid; place-items: center; font-size: 18px; }
    button:disabled { opacity: 0.58; cursor: wait; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .command-row { display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: end; }
    .chat-log { min-height: 420px; max-height: 62vh; overflow: auto; display: grid; align-content: start; gap: 10px; padding: 16px; border-bottom: 1px solid var(--line); }
    .message { border-radius: 8px; padding: 11px 12px; line-height: 1.45; white-space: pre-wrap; overflow-wrap: anywhere; }
    .message.user { background: #e9f0e8; border: 1px solid #c6d6c0; }
    .message.assistant { background: #181816; color: #f8f3e8; }
    .message.system { background: #f3ead9; border: 1px solid #d8cfbd; color: #332f2a; font-weight: 750; }
    .error { color: #9d1c1c; font-weight: 800; }
    .fen-line { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
    @media (max-width: 1120px) {
      .layout { grid-template-columns: minmax(310px, 1fr) minmax(330px, 1fr); }
      .chat-panel { grid-column: 1 / -1; }
    }
    @media (max-width: 760px) {
      main { padding: 14px; }
      header { align-items: start; flex-direction: column; }
      .status { justify-content: flex-start; }
      .layout, .row, .command-row { grid-template-columns: 1fr; }
      h1 { font-size: 26px; }
      .chat-log { min-height: 300px; max-height: 52vh; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Chess Assistant</h1>
    <div class="status" id="status">
      <span class="chip">white to move</span>
      <span class="chip">ready</span>
    </div>
  </header>
  <section class="layout">
    <section class="panel board-panel">
      <div class="board-shell">
        <div id="board" class="board" aria-label="Chess board"></div>
        <div class="command-row">
          <label>Move
            <input id="move" placeholder="e2e4 or Nf3" autocomplete="off">
          </label>
          <button id="play" type="button">Play</button>
          <button id="reset" type="button" class="secondary">Reset</button>
        </div>
        <div class="fen-line" id="fen-line"></div>
      </div>
    </section>
    <section class="panel controls">
      <label>Question
        <textarea id="question">What should I play and why?</textarea>
      </label>
      <div class="row">
        <label>Image side
          <select id="side">
            <option value="w">White</option>
            <option value="b">Black</option>
          </select>
        </label>
        <label>Castling
          <input id="castling" value="-">
        </label>
      </div>
      <label>Board image
        <input id="image" type="file" accept="image/*">
      </label>
      <div class="command-row">
        <button id="ask" type="button">Ask</button>
        <button id="mic" type="button" class="secondary icon" aria-label="Record audio question" title="Record audio question">Mic</button>
        <button id="clear-image" type="button" class="secondary">Clear image</button>
      </div>
    </section>
    <section class="panel chat-panel">
      <div id="chat-log" class="chat-log">
        <div class="message system">Ready.</div>
      </div>
      <div class="chat">
        <div class="fen-line" id="engine-line"></div>
      </div>
    </section>
  </section>
</main>
<script>
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const PIECES = { p: "♟", r: "♜", n: "♞", b: "♝", q: "♛", k: "♚", P: "♙", R: "♖", N: "♘", B: "♗", Q: "♕", K: "♔" };
let currentFen = START_FEN;
let selectedSquare = null;
let lastMove = [];
let busy = false;

const boardEl = document.getElementById("board");
const moveEl = document.getElementById("move");
const questionEl = document.getElementById("question");
const statusEl = document.getElementById("status");
const fenLineEl = document.getElementById("fen-line");
const engineLineEl = document.getElementById("engine-line");
const chatLogEl = document.getElementById("chat-log");

document.getElementById("play").addEventListener("click", playMove);
document.getElementById("ask").addEventListener("click", askQuestion);
document.getElementById("reset").addEventListener("click", resetBoard);
document.getElementById("clear-image").addEventListener("click", () => { document.getElementById("image").value = ""; });
document.getElementById("mic").addEventListener("click", recordQuestion);
moveEl.addEventListener("keydown", event => { if (event.key === "Enter") playMove(); });

renderBoard(currentFen);
updateStatus({ fen: currentFen, turn: "white", status: "active", legal_moves: [] });

async function playMove() {
  const move = moveEl.value.trim();
  if (!move || busy) return;
  setBusy(true);
  addMessage("user", move);
  try {
    const data = await postJson("/api/move", { fen: currentFen, move });
    currentFen = data.fen;
    lastMove = [data.played.user_move_uci?.slice(0, 2), data.played.user_move_uci?.slice(2, 4), data.played.assistant_move_uci?.slice(0, 2), data.played.assistant_move_uci?.slice(2, 4)].filter(Boolean);
    moveEl.value = "";
    renderBoard(currentFen);
    updateStatus(data);
    addMessage("assistant", moveSummary(data));
    setEngineLine(data);
  } catch (error) {
    addMessage("system", error.message, true);
  } finally {
    setBusy(false);
  }
}

async function askQuestion() {
  if (busy) return;
  setBusy(true);
  const question = questionEl.value.trim() || "What should I play and why?";
  addMessage("user", question);
  try {
    const file = document.getElementById("image").files[0];
    const payload = {
      question,
      fen: currentFen,
      side_to_move: document.getElementById("side").value,
      castling: document.getElementById("castling").value || "-"
    };
    if (file) payload.image_base64 = await fileToDataUrl(file);
    const data = await postJson("/api/ask", payload);
    currentFen = data.fen;
    renderBoard(currentFen);
    updateStatus(data);
    addMessage("assistant", data.answer);
    setEngineLine(data);
  } catch (error) {
    addMessage("system", error.message, true);
  } finally {
    setBusy(false);
  }
}

async function resetBoard() {
  if (busy) return;
  setBusy(true);
  try {
    const data = await postJson("/api/reset", {});
    currentFen = data.fen;
    selectedSquare = null;
    lastMove = [];
    moveEl.value = "";
    renderBoard(currentFen);
    updateStatus(data);
    addMessage("system", "Board reset.");
  } catch (error) {
    addMessage("system", error.message, true);
  } finally {
    setBusy(false);
  }
}

function recordQuestion() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    addMessage("system", "Audio input is not supported by this browser.", true);
    return;
  }
  const recognition = new Recognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onresult = event => {
    questionEl.value = event.results[0][0].transcript;
    askQuestion();
  };
  recognition.onerror = event => addMessage("system", `Audio input failed: ${event.error}`, true);
  recognition.start();
}

function renderBoard(fen) {
  boardEl.innerHTML = "";
  const rows = fen.split(" ")[0].split("/");
  for (let rank = 8; rank >= 1; rank--) {
    const row = rows[8 - rank];
    let file = 0;
    for (const char of row) {
      if (Number.isInteger(Number(char))) {
        const empties = Number(char);
        for (let i = 0; i < empties; i++) addSquare(file++, rank, "", "");
      } else {
        addSquare(file++, rank, PIECES[char] || "", char);
      }
    }
  }
}

function addSquare(file, rank, piece, pieceCode) {
  const files = "abcdefgh";
  const square = `${files[file]}${rank}`;
  const button = document.createElement("button");
  button.type = "button";
  button.className = `square ${((file + rank) % 2 === 0) ? "dark" : "light"}`;
  if (selectedSquare === square) button.classList.add("selected");
  if (lastMove.includes(square)) button.classList.add("last");
  button.dataset.square = square;
  const pieceColor = pieceCode && pieceCode === pieceCode.toUpperCase() ? "white" : "black";
  button.innerHTML = `<span class="piece ${piece ? pieceColor : ""}">${piece}</span><span class="coord">${square}</span>`;
  button.addEventListener("click", () => chooseSquare(square));
  boardEl.appendChild(button);
}

function chooseSquare(square) {
  if (!selectedSquare) {
    selectedSquare = square;
    renderBoard(currentFen);
    return;
  }
  let move = `${selectedSquare}${square}`;
  if (needsPromotion(move)) move += "q";
  selectedSquare = null;
  moveEl.value = move;
  renderBoard(currentFen);
  playMove();
}

function needsPromotion(move) {
  const fromRank = move[1];
  const toRank = move[3];
  return (fromRank === "7" && toRank === "8") || (fromRank === "2" && toRank === "1");
}

function moveSummary(data) {
  const played = data.played || {};
  const assistantMove = played.assistant_move_san ? ` I replied ${played.assistant_move_san}.` : "";
  return `You played ${played.user_move_san}.${assistantMove}\n\n${data.answer}`;
}

function updateStatus(data) {
  const best = data.engine?.best_move_san || data.engine?.best_move_uci || "ready";
  statusEl.innerHTML = [
    `${data.turn || "white"} to move`,
    data.status || "active",
    `best ${best}`
  ].map(text => `<span class="chip">${escapeHtml(text)}</span>`).join("");
  fenLineEl.textContent = data.fen || currentFen;
}

function setEngineLine(data) {
  const engine = data.engine || {};
  const pv = (engine.principal_variation || []).join(" ");
  engineLineEl.textContent = `Engine: ${engine.engine_name || "ready"} | score: ${engine.score_cp ?? "n/a"} | pv: ${pv}`;
}

async function postJson(url, payload) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function addMessage(role, text, isError = false) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  if (isError) div.classList.add("error");
  div.textContent = text;
  chatLogEl.appendChild(div);
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function setBusy(value) {
  busy = value;
  for (const button of document.querySelectorAll("button")) button.disabled = value;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}
</script>
</body>
</html>
"""
