const { chromium } = require("playwright");

const targetUrl = process.argv[2] || "http://127.0.0.1:8765";
const outputDir = process.argv[3] || "outputs/smoke/report";

async function checkViewport(label, viewport, options = {}) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport, ...options });
  const events = [];
  page.on("console", (message) => events.push({ type: message.type(), text: message.text() }));
  page.on("pageerror", (error) => events.push({ type: "pageerror", text: error.message }));
  await page.goto(targetUrl, { waitUntil: "networkidle" });
  const metrics = await page.evaluate(() => ({
    title: document.title,
    h1: document.querySelector("h1")?.textContent,
    score: document.querySelector(".score span")?.textContent,
    sampleCount: document.querySelectorAll("article").length,
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
    canScrollX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    canScrollY: document.documentElement.scrollHeight > document.documentElement.clientHeight,
  }));
  await page.screenshot({ path: `${outputDir}/playwright-${label}.png`, fullPage: false });
  await browser.close();
  return { label, metrics, events };
}

(async () => {
  const desktop = await checkViewport("desktop", { width: 1440, height: 1000 });
  const mobile = await checkViewport("mobile", { width: 390, height: 844 }, { isMobile: true });
  const result = { targetUrl, checks: [desktop, mobile] };
  console.log(JSON.stringify(result, null, 2));
  const hasErrors = result.checks.some((check) => check.events.length || check.metrics.canScrollX);
  process.exit(hasErrors ? 1 : 0);
})();
