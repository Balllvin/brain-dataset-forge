from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dataset_forge.chess_assistant.engine import ChessEngineConfig
from dataset_forge.chess_assistant.language import LanguageConfig
from dataset_forge.chess_assistant.orchestrator import ChessAssistant, ChessAssistantConfig
from dataset_forge.chess_assistant.position import ChessInputError
from dataset_forge.chess_assistant.vision import VisionInputError, image_bytes_to_fen


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
            if self.path != "/api/ask":
                self.send_error(404)
                return
            try:
                payload = self._json_payload()
                response = _handle_ask(payload, assistant)
                self._json_response(200, response)
            except (ChessInputError, VisionInputError, ValueError) as error:
                self._json_response(400, {"error": str(error)})
            except Exception as error:
                self._json_response(500, {"error": f"Chess assistant failed: {error}"})

        def _json_payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
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
    fen = str(payload.get("fen") or "").strip()
    move = str(payload.get("move") or "").strip() or None
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
        response = assistant.answer_vision_result(question, vision, move=move)
    else:
        if not fen:
            raise ValueError("Provide a FEN or upload a board image.")
        response = assistant.answer(question, fen=fen, move=move)
    return response.to_record()


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
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f4ed; color: #141414; }
    main { max-width: 1120px; margin: 0 auto; padding: 28px; display: grid; gap: 18px; }
    header { display: grid; gap: 8px; }
    h1 { margin: 0; font-size: 34px; line-height: 1.05; letter-spacing: 0; }
    p { margin: 0; line-height: 1.5; }
    .layout { display: grid; grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.1fr); gap: 18px; align-items: start; }
    form, .answer, .metrics { background: #fffdf8; border: 1px solid #ded7c9; border-radius: 8px; padding: 18px; box-shadow: 0 12px 34px rgba(23, 20, 17, 0.08); }
    label { display: grid; gap: 7px; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
    textarea, input, select { width: 100%; box-sizing: border-box; border: 1px solid #bdb4a4; border-radius: 6px; padding: 10px; font: inherit; background: #fff; color: #141414; }
    textarea { min-height: 96px; resize: vertical; }
    .stack { display: grid; gap: 14px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    button { appearance: none; border: 0; border-radius: 6px; background: #1e5b4f; color: white; padding: 11px 14px; font-weight: 800; cursor: pointer; }
    button:disabled { opacity: 0.58; cursor: wait; }
    .answer { min-height: 280px; display: grid; gap: 14px; align-content: start; }
    .answer pre { white-space: pre-wrap; word-break: break-word; background: #181816; color: #f8f3e8; border-radius: 8px; padding: 14px; overflow-x: auto; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { border: 1px solid #d8cfbd; background: #f3ead9; border-radius: 999px; padding: 6px 9px; font-size: 12px; font-weight: 700; }
    .error { color: #9d1c1c; font-weight: 700; }
    @media (max-width: 780px) {
      main { padding: 18px; }
      .layout, .row { grid-template-columns: 1fr; }
      h1 { font-size: 28px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Chess Assistant</h1>
    <p>Ask from a FEN or upload a clean top-down board image. The assistant validates legality, analyzes the position, and explains the move.</p>
  </header>
  <section class="layout">
    <form id="ask-form" class="stack">
      <label>Question
        <textarea id="question">What should I play and why?</textarea>
      </label>
      <label>FEN
        <textarea id="fen">rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1</textarea>
      </label>
      <div class="row">
        <label>Move just played
          <input id="move" placeholder="Optional SAN or UCI">
        </label>
        <label>Side to move for image
          <select id="side">
            <option value="w">White</option>
            <option value="b">Black</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>Castling for image
          <input id="castling" value="-">
        </label>
        <label>Board image
          <input id="image" type="file" accept="image/*">
        </label>
      </div>
      <button id="submit" type="submit">Analyze</button>
    </form>
    <section class="answer">
      <div class="chips" id="chips"></div>
      <p id="answer-text">Ready.</p>
      <pre id="fen-out"></pre>
    </section>
  </section>
</main>
<script>
const form = document.getElementById('ask-form');
const submit = document.getElementById('submit');
const answerText = document.getElementById('answer-text');
const fenOut = document.getElementById('fen-out');
const chips = document.getElementById('chips');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submit.disabled = true;
  answerText.textContent = 'Analyzing...';
  fenOut.textContent = '';
  chips.innerHTML = '';
  try {
    const file = document.getElementById('image').files[0];
    const payload = {
      question: document.getElementById('question').value,
      fen: document.getElementById('fen').value,
      move: document.getElementById('move').value,
      side_to_move: document.getElementById('side').value,
      castling: document.getElementById('castling').value || '-'
    };
    if (file) payload.image_base64 = await fileToDataUrl(file);
    const response = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    answerText.textContent = data.answer;
    fenOut.textContent = data.fen;
    chips.innerHTML = [
      `best ${data.engine.best_move_san || data.engine.best_move_uci}`,
      `engine ${data.engine.engine_name}`,
      `score ${data.engine.score_cp}`,
      data.vision ? `image ${data.vision.confidence}` : 'fen input'
    ].map(text => `<span class="chip">${escapeHtml(text)}</span>`).join('');
  } catch (error) {
    answerText.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
  } finally {
    submit.disabled = false;
  }
});

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
</script>
</body>
</html>
"""
