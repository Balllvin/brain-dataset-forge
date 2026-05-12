const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

async function main() {
  const targetUrl = process.argv[2] || "http://127.0.0.1:8766";
  const outputDir = process.argv[3] || "outputs/chess_web_qa";
  const imagePath = process.argv[4] || "examples/chess_assistant/assets/startpos.png";
  fs.mkdirSync(outputDir, { recursive: true });

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const events = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) events.push({ type: message.type(), text: message.text() });
  });
  page.on("pageerror", (error) => events.push({ type: "pageerror", text: error.message }));

  await page.goto(targetUrl, { waitUntil: "networkidle" });
  const h1 = await page.locator("h1").innerText();
  const metrics = await page.evaluate(() => ({
    title: document.title,
    h1: document.querySelector("h1")?.textContent,
    canScrollX: document.documentElement.scrollWidth > window.innerWidth,
    formCount: document.querySelectorAll("form").length,
  }));

  const fenResponse = await page.evaluate(async () => {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: "What should I play and why?",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      })
    });
    return response.json();
  });

  const imageBase64 = fs.readFileSync(imagePath).toString("base64");
  const imageResponse = await page.evaluate(async (imageBase64Value) => {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: "Read this board and recommend a move.",
        image_base64: imageBase64Value,
        side_to_move: "w",
        castling: "KQkq"
      })
    });
    return response.json();
  }, imageBase64);

  await page.screenshot({ path: path.join(outputDir, "chess-assistant.png"), fullPage: true });
  await browser.close();

  const report = {
    targetUrl,
    h1,
    metrics,
    fenAnswerHasMove: Boolean(fenResponse.engine?.best_move_uci && fenResponse.answer?.includes(fenResponse.engine.best_move_uci)),
    imageConfidence: imageResponse.vision?.confidence,
    imageFen: imageResponse.fen,
    events,
  };
  console.log(JSON.stringify(report, null, 2));
  if (events.length || metrics.canScrollX || !report.fenAnswerHasMove || report.imageConfidence < 0.9) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
