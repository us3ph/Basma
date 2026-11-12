// Views of one image: global (short side -> 224, centre crop of the 512-px base) + up to 3 texture-rich 224 crops taken from the
// image's own pixels, never resized (v6). Same as basma/views.py.
import { VIEW, resizeArea, fitMinSide } from "./resample.js";
const MEAN = [0.485, 0.456, 0.406].map(Math.fround), STD = [0.229, 0.224, 0.225].map(Math.fround), K_CROPS = 3, GRID = 3;

function crop(x, h, w, y0, x0, size) {   // float64 (h,w,3) -> (size,size,3)
  const out = new Float64Array(size * size * 3);
  for (let y = 0; y < size; y++) { const a = ((y0 + y) * w + x0) * 3, b = y * size * 3; for (let j = 0; j < size * 3; j++) out[b + j] = x[a + j]; }
  return out;
}
export function normalize(v) {            // (224,224,3) float64 0..255 -> Float32Array (3,224,224)
  const n = VIEW * VIEW, out = new Float32Array(3 * n);
  // float32 at every step, like numpy: fround(v) / 255 -> - mean -> / std (each op correctly rounded to float32)
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[c * n + i] = Math.fround(Math.fround(Math.fround(Math.fround(v[i * 3 + c]) / 255) - MEAN[c]) / STD[c]);
  return out;
}
export function globalView(base) {
  const [th, tw] = fitMinSide(base.h, base.w, VIEW);
  const x = (th === base.h && tw === base.w) ? { data: Float64Array.from(base.data), h: th, w: tw } : resizeArea(base, th, tw);
  return crop(x.data, th, tw, Math.floor((th - VIEW) / 2), Math.floor((tw - VIEW) / 2), VIEW);
}
function laplacianVar(gray, size) {       // gray: Float64Array (size*size), variance of the 4-neighbour Laplacian over the interior
  const n = (size - 2) * (size - 2); let sum = 0; const lap = new Float64Array(n); let i = 0;
  for (let y = 1; y < size - 1; y++) for (let x = 1; x < size - 1; x++) {
    const c = y * size + x;
    const v = 4.0 * gray[c] - gray[c - size] - gray[c + size] - gray[c - 1] - gray[c + 1]; lap[i++] = v; sum += v;
  }
  const m = sum / n; let s = 0; for (let j = 0; j < n; j++) { const d = lap[j] - m; s += d * d; }
  return s / n;
}
export function gridPositions(n, size, grid = GRID) {
  if (n <= size) return [0];
  const pos = [];
  for (let i = 0; i < grid; i++) { const p = Math.floor((n - size) * i / (grid - 1) + 0.5); if (!pos.includes(p)) pos.push(p); }
  return pos;
}
// img {data uint8 | float64, h, w}: the base, or (v6) the image's own untouched pixels. Only the candidate windows are
// converted to float64, so a 24-megapixel image costs no more than a small one (same as basma/views.py).
export function textureCrops(img, k = K_CROPS) {
  let x, h, w;
  if (Math.min(img.h, img.w) < VIEW) { const [th, tw] = fitMinSide(img.h, img.w, VIEW); const r = resizeArea(img, th, tw); x = r.data; h = th; w = tw; }
  else { x = img.data; h = img.h; w = img.w; }
  const window = (y0, x0) => crop(x, h, w, y0, x0, VIEW);      // Float64Array (224,224,3); uint8 values are exact in float64
  const cands = [];
  for (const y of gridPositions(h, VIEW)) for (const xx of gridPositions(w, VIEW)) {
    const v = window(y, xx), gray = new Float64Array(VIEW * VIEW);
    for (let i = 0; i < VIEW * VIEW; i++) gray[i] = 0.299 * v[i * 3] + 0.587 * v[i * 3 + 1] + 0.114 * v[i * 3 + 2];
    cands.push({ s: laplacianVar(gray, VIEW), y, x: xx, i: cands.length });
  }
  cands.sort((a, b) => (b.s - a.s) || (a.i - b.i));   // stable: ties keep grid order
  return cands.slice(0, k).map(c => window(c.y, c.x));
}
// base {data uint8, h, w} (+ native: the full-resolution image when it is larger than the base) -> {views: Float32Array (V*3*224*224), n: V, info}
// View 0 is the global view of the base; the texture crops come from the native pixels when given (v6), else from the base.
export function inferenceViews(base, native = null) {
  const g = globalView(base), crops = textureCrops(native || base);
  const all = [g, ...crops], n = all.length, per = 3 * VIEW * VIEW, out = new Float32Array(n * per);
  all.forEach((v, i) => out.set(normalize(v), i * per));
  return { views: out, n, info: { n_crops: crops.length, h: base.h, w: base.w } };
}
