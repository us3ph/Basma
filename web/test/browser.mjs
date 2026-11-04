// Real-browser test of the page with headless Chromium (Playwright): load, wait for the model, drop two images
// (one real, one AI from the local corpus), read the verdicts, screenshot.
//   PW_CHROME=... node test/browser.mjs [url] [screenshot.png]   (SKIP_MODEL=1 for a design-only run)
import { readFileSync, readdirSync } from "node:fs";
import { chromium } from "playwright-core";
const url = process.argv[2] || "https://x-hunter123345-basma-detector.static.hf.space/";
const shot = process.argv[3] || "/tmp/basma_browser.png";
const root = new URL("../..", import.meta.url).pathname;
const manifest = readFileSync(root + "data/mini/manifest_mini.jsonl", "utf8").trim().split("\n").map(l => JSON.parse(l));
const pick = (label) => manifest.find(r => r.label === label && r.split === "train");
const samples = [pick(0), pick(1)].filter(Boolean);
const browser = await chromium.launch({ headless: true, executablePath: process.env.PW_CHROME || undefined, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(`[${m.type()}] ${m.text().slice(0, 300)}`); });
page.on("pageerror", (e) => errors.push(`[pageerror] ${String(e).slice(0, 300)}`));
if (process.env.SKIP_MODEL) await page.route("**/resolve/main/onnx/**", (r) => r.abort());
const t0 = Date.now();
await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120000 });
if (process.env.SKIP_MODEL) { await page.waitForTimeout(4000); await page.screenshot({ path: shot }); console.log("design screenshot:", shot); if (errors.length) console.log(errors.slice(0, 6).join("\n")); await browser.close(); process.exit(0); }
let ready = false;
for (let i = 0; i < 180; i++) {
  const s = await page.locator("#status").textContent();
  if (/Ready/.test(s || "")) { ready = true; break; }
  if (/couldn't load|went wrong/.test(s || "")) { console.log("STATUS:", s); break; }
  if (i % 12 === 0) console.log(`  ${((Date.now() - t0) / 1000).toFixed(0)}s  ${s}`);
  await page.waitForTimeout(/MB/.test(s || "") ? 1500 : 5000);
}
console.log(`ready=${ready} after ${((Date.now() - t0) / 1000).toFixed(0)}s`);
let ok = ready;
if (ready) {
  for (const r of samples) {
    await page.setInputFiles("#file", root + "data/img/" + r.member);
    await page.waitForTimeout(800);
    await page.click("#send");
    await page.waitForFunction(() => document.querySelector("#verdict .v")?.textContent, null, { timeout: 180000 });
    const verdict = await page.locator("#verdict .v").textContent(); const sub = await page.locator("#verdict .s").textContent(); const count = await page.locator("#count").textContent();
    console.log(`${r.source.padEnd(12)} label=${r.label} -> ${verdict} (${sub}) | ${count}`);
  }
}
await page.screenshot({ path: shot }); console.log("screenshot:", shot);
if (errors.length) console.log("console messages:\n  " + errors.slice(0, 8).join("\n  "));
await browser.close(); console.log(ok ? "BROWSER OK" : "BROWSER FAILED"); process.exit(ok ? 0 : 1);
