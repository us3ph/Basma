// Port of basma/fusion.py Fusion: logistic regression over the view logits -> isotonic table -> verdict.
export function features(r) {
  const lc = r.lc.length ? r.lc : [r.lg];
  const mean = lc.reduce((a, b) => a + b, 0) / lc.length;
  return [r.lg, mean, Math.max(...lc), Math.min(...lc), r.lc.length / 3.0, Math.log(Math.max(1, r.h * r.w)) / 12.0];
}
function interp(x, xs, ys) {       // numpy.interp with clamping at the ends
  if (x <= xs[0]) return ys[0]; if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]); return ys[lo] + t * (ys[hi] - ys[lo]);
}
export function createFusion(f) {
  const raw = (r) => { const x = features(r); let s = f.intercept; for (let i = 0; i < x.length; i++) s += f.coef[i] * x[i]; return s; };
  const prob = (r) => interp(raw(r), f.iso_x, f.iso_y);
  const verdict = (p) => p >= f.thr_ai ? "AI" : (p <= f.thr_real ? "Real" : "Uncertain");
  return { raw, prob, verdict, thresholds: { ai: f.thr_ai, real: f.thr_real } };
}
