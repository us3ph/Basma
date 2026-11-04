// Basma in the browser: decode the image -> deterministic base/views (resample.js, views.js) -> ONNX detector in a
// worker -> calibrated fusion (fusion.js). Nothing leaves the browser.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.mjs";
import { toBase } from "./resample.js";
import { inferenceViews } from "./views.js";
import { createFusion } from "./fusion.js";

const MODEL_REPO = "x-hunter123345/basma-base";
const MODEL_CANDIDATES = [
  { url: `https://huggingface.co/${MODEL_REPO}/resolve/main/onnx/model_w8.onnx`, label: "Downloading the detection model" },
  { url: `https://huggingface.co/${MODEL_REPO}/resolve/main/onnx/model.onnx`, label: "Downloading the detection model" },
];
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
ort.env.wasm.numThreads = (self.crossOriginIsolated && navigator.hardwareConcurrency) ? Math.min(4, navigator.hardwareConcurrency) : 1;

const $ = (id) => document.getElementById(id);
const status = (msg) => { const el = $("status"); el.classList.remove("off"); if (/\d+ \/ \d+ MB|Analyzing/.test(msg)) { el.classList.remove("stagger"); el.textContent = msg; } else window.staggerText(el, msg); };
function showResult(r) { $("status").classList.add("off"); $("verdict").innerHTML = verdictHTML(r); $("count").textContent = `${r.w} × ${r.h} px`; }
function clearResult() { $("verdict").innerHTML = ""; $("count").textContent = ""; }
const progress = (frac) => { $("barfill").style.width = `${Math.round(100 * frac)}%`; };
function checking() { const el = $("status"); el.classList.remove("stagger", "off"); el.innerHTML = '<span class="spin" aria-hidden="true"></span>Checking'; }

async function fetchWithProgress(url, label) {
  let cache = null;
  try { cache = await caches.open("basma-v1"); const hit = await cache.match(url); if (hit) { status("Loading the detection model…"); return await hit.arrayBuffer(); } } catch (_) { /* no Cache API */ }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader(); const chunks = []; let got = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total) { progress(got / total); status(`${label} · ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`); }
  }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { if (cache) await cache.put(url, new Response(buf.buffer.slice(0), { headers: { "content-type": "application/octet-stream" } })); } catch (_) { /* quota */ }
  return buf.buffer;
}
async function loadJSON(p) { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.json(); }

let fusion = null, session = null, worker = null, seq = 0, current = null;
const pending = new Map();
function startWorker() {
  try {
    worker = new Worker(new URL("./ort-worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => { const p = pending.get(e.data.id); if (!p) return; pending.delete(e.data.id); e.data.ok ? p.res(e.data) : p.rej(new Error(e.data.error)); };
    worker.onerror = (e) => { for (const p of pending.values()) p.rej(new Error(e.message || "worker error")); pending.clear(); };
  } catch (_) { worker = null; }
}
const call = (msg, transfer = []) => new Promise((res, rej) => { const id = ++seq; pending.set(id, { res, rej }); worker.postMessage({ ...msg, id }, transfer); });
async function createSession(buf) {
  if (worker) {
    try { await call({ type: "init", buffer: buf }, [buf]); return "worker"; }
    catch (e) { console.warn("worker session failed:", e); worker.terminate(); worker = null; throw e; }
  }
  session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return "main";
}
async function runModel(views, n) {
  if (worker) { const r = await call({ type: "run", views, n }, [views.buffer]); return r; }
  const out = await session.run({ pixel_values: new ort.Tensor("float32", views, [n, 3, 224, 224]) });
  return { logit: Array.from(out.logit.data), patch: Array.from(out.patch.data) };
}

async function init() {
  try {
    status("Preparing the detector…"); progress(0.03);
    fusion = createFusion(await loadJSON("./assets/fusion.json"));
    let lastErr = null, ready = null;
    startWorker();
    for (const cand of MODEL_CANDIDATES) {
      for (const attempt of [0, 1]) {
        try { const buf = await fetchWithProgress(cand.url, cand.label); status("Almost ready…"); ready = await createSession(buf); break; }
        catch (e) { console.warn("model candidate failed:", cand.url, e); lastErr = e; if (worker) break; }
      }
      if (ready) break;
    }
    if (!ready) throw lastErr || new Error("no model could be loaded");
    console.log("basma model running on", ready === "worker" ? "a background worker" : "the main thread");
    progress(1); status("Ready to check"); setTimeout(() => $("bar").classList.add("done"), 700);
    $("send").disabled = !current; window.basmaBlob?.setState("idle");
  } catch (e) { console.error(e); status("The detector couldn't load. Check your connection and refresh the page."); }
}

// ---- image input: file picker, drag & drop, paste
async function decode(file) {
  const bmp = await createImageBitmap(file, { colorSpaceConversion: "none", imageOrientation: "from-image", premultiplyAlpha: "none" });
  const c = document.createElement("canvas"); c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true, alpha: false, colorSpace: "srgb" });
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height); ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data; bmp.close();
  const rgb = new Uint8Array(c.width * c.height * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2]; }
  return { data: rgb, h: c.height, w: c.width, canvas: c };
}
async function setImage(file) {
  if (!file || !file.type.startsWith("image/")) { status("Please choose an image file."); return; }
  try {
    const img = await decode(file);
    current = img;
    const prev = $("preview"); prev.src = img.canvas.toDataURL("image/jpeg", 0.85); prev.alt = file.name;
    $("drop").classList.add("has"); $("ph").classList.remove("show");
    $("send").disabled = !fusion || !(worker || session);
    clearResult();
    if (fusion && (worker || session)) status("Ready to check");
  } catch (e) { console.error(e); status("That image couldn't be read."); }
}

function verdictHTML(r) {
  const color = { AI: "var(--ai)", Real: "var(--human)", Uncertain: "var(--mixed)" }[r.verdict];
  const sub = { AI: `${r.percent}% likely AI-generated`, Real: `${r.percent}% likely a real photo or artwork`, Uncertain: `${Math.round(100 * r.p)}% AI score — not enough evidence either way` }[r.verdict];
  return `<div class="v" style="color:${color}">${r.verdict}</div><div class="s">${sub}</div>`;
}

async function detect() {
  if (!current) { status("Drop an image first."); return; }
  clearResult();
  $("send").disabled = true; window.basmaBlob?.setState("thinking"); checking();
  try {
    const t0 = performance.now();
    const base = toBase(current);
    const { views, n, info } = inferenceViews(base);
    const out = await runModel(views, n);
    const row = { lg: out.logit[0], lc: out.logit.slice(1, n), h: info.h, w: info.w };
    const p = fusion.prob(row), v = fusion.verdict(p);
    const r = { verdict: v, p, percent: v === "Real" ? Math.round(100 * (1 - p)) : Math.round(100 * p), w: current.w, h: current.h, views: out.logit, ms: performance.now() - t0 };
    showResult(r); window.basmaBlob?.speak(1600);
    console.log("basma", r);
  } catch (e) { console.error(e); status("Something went wrong while analyzing. Please try again."); }
  finally { $("send").disabled = false; window.basmaBlob?.setState("idle"); }
}

$("send").addEventListener("click", detect);
$("file").addEventListener("change", (e) => setImage(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("click", (e) => { if (e.target.id !== "send") $("file").click(); });
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); setImage(e.dataTransfer.files[0]); });
window.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/")); if (f) setImage(f.getAsFile()); });
window.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) detect(); });
init();
