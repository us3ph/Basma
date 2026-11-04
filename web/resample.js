// Deterministic resampling, operation for operation the same as basma/resample.py (float64, same order of
// operations), so the browser computes the same views as the Python scorer.
export const MAX_SIDE = 512, VIEW = 224, REDUCE_ABOVE = 1024;

export function toU8(x) { const v = Math.floor(x + 0.5); return v < 0 ? 0 : v > 255 ? 255 : v; }

// img: {data: Uint8Array (H*W*3), h, w}
export function blockReduce(img, k) {
  if (k <= 1) return img;
  const hh = Math.floor(img.h / k), ww = Math.floor(img.w / k), out = new Uint8Array(hh * ww * 3), kk = k * k, half = Math.floor(kk / 2);
  for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) {
    let s0 = 0, s1 = 0, s2 = 0;
    for (let dy = 0; dy < k; dy++) { let p = ((y * k + dy) * img.w + x * k) * 3; for (let dx = 0; dx < k; dx++) { s0 += img.data[p]; s1 += img.data[p + 1]; s2 += img.data[p + 2]; p += 3; } }
    const o = (y * ww + x) * 3; out[o] = Math.floor((s0 + half) / kk); out[o + 1] = Math.floor((s1 + half) / kk); out[o + 2] = Math.floor((s2 + half) / kk);
  }
  return { data: out, h: hh, w: ww };
}

// area-average along the first axis. x: Float64Array laid out (n_in, m) row-major -> (n_out, m)
function areaAxis0(x, nIn, m, nOut) {
  if (nOut === nIn) return x;
  const s = nIn / nOut;
  const cs = new Float64Array((nIn + 1) * m);                 // cumulative sums, row 0 = 0
  for (let i = 0; i < nIn; i++) { const a = i * m, b = (i + 1) * m; for (let j = 0; j < m; j++) cs[b + j] = cs[a + j] + x[a + j]; }
  const F = new Float64Array((nOut + 1) * m);
  for (let e = 0; e <= nOut; e++) {
    const idx = e * s, fl = Math.floor(idx), fr = idx - fl;
    const fo = e * m;
    if (fl >= nIn) { const a = nIn * m; for (let j = 0; j < m; j++) F[fo + j] = cs[a + j]; }
    else { const a = fl * m; for (let j = 0; j < m; j++) F[fo + j] = cs[a + j] + fr * x[a + j]; }
  }
  const out = new Float64Array(nOut * m);
  for (let i = 0; i < nOut; i++) { const a = i * m, b = (i + 1) * m; for (let j = 0; j < m; j++) out[a + j] = (F[b + j] - F[a + j]) / s; }
  return out;
}

function transpose(x, h, w, c) {   // (h, w, c) -> (w, h, c)
  const out = new Float64Array(x.length);
  for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) { const a = (y * w + xx) * c, b = (xx * h + y) * c; for (let k = 0; k < c; k++) out[b + k] = x[a + k]; }
  return out;
}

// img {data (uint8 or float64), h, w} -> {data: Float64Array, h: oh, w: ow}
export function resizeArea(img, oh, ow) {
  let x = img.data instanceof Float64Array ? img.data : Float64Array.from(img.data);
  x = areaAxis0(x, img.h, img.w * 3, oh);                     // rows
  x = transpose(x, oh, img.w, 3);
  x = areaAxis0(x, img.w, oh * 3, ow);                        // columns
  x = transpose(x, ow, oh, 3);
  return { data: x, h: oh, w: ow };
}

export function fitMaxSide(h, w, maxSide) {
  const m = Math.max(h, w); if (m <= maxSide) return [h, w];
  const sc = maxSide / m; return [Math.max(1, Math.floor(h * sc + 0.5)), Math.max(1, Math.floor(w * sc + 0.5))];
}
export function fitMinSide(h, w, minSide) {
  const m = Math.min(h, w), sc = minSide / m;
  return [Math.max(minSide, Math.floor(h * sc + 0.5)), Math.max(minSide, Math.floor(w * sc + 0.5))];
}

export function toBase(img) {
  let { h, w } = img;
  const k = Math.max(1, Math.floor(Math.max(h, w) / REDUCE_ABOVE));
  if (k > 1) { img = blockReduce(img, k); h = img.h; w = img.w; }
  const [th, tw] = fitMaxSide(h, w, MAX_SIDE);
  if (th === h && tw === w) return img;
  const r = resizeArea(img, th, tw);
  const out = new Uint8Array(r.data.length); for (let i = 0; i < out.length; i++) out[i] = toU8(r.data[i]);
  return { data: out, h: th, w: tw };
}
