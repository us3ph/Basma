// node test/parity_fusion.mjs : fusion.js must reproduce the Python probabilities/verdicts of assets/onnx/test_vectors.json
import { readFileSync } from "node:fs";
import { createFusion } from "../fusion.js";
const f = createFusion(JSON.parse(readFileSync(new URL("../assets/fusion.json", import.meta.url))));
const vecs = JSON.parse(readFileSync(new URL("../assets/onnx/test_vectors.json", import.meta.url)));
let ok = true;
for (const v of vecs) {
  const lg = v.views_logit_pt; const p = f.prob({ lg: lg[0], lc: lg.slice(1), h: v.h ?? 512, w: v.w ?? 512 });
  const same = Math.abs(p - v.p_ai) < 1e-6 && f.verdict(p) === v.verdict; ok &&= same;
  console.log(`${v.id} label=${v.label} js p=${p.toFixed(6)} py p=${v.p_ai.toFixed(6)} verdict ${f.verdict(p)} / ${v.verdict} ${same ? "OK" : "DIFF"}`);
}
console.log(ok ? "FUSION PARITY OK" : "FUSION PARITY FAILED"); process.exit(ok ? 0 : 1);
