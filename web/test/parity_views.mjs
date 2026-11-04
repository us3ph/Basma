// node test/parity_views.mjs : JS resample/views must reproduce Python bit for bit (sha1 of base and of the views).
import { readFileSync } from "node:fs"; import { createHash } from "node:crypto";
import { toBase } from "../resample.js"; import { inferenceViews } from "../views.js";
const vecs = JSON.parse(readFileSync(new URL("./vectors_views.json", import.meta.url)));
let ok = true;
for (const v of vecs) {
  const data = new Uint8Array(Buffer.from(v.pixels_b64, "base64"));
  const base = toBase({ data, h: v.h, w: v.w });
  const bsha = createHash("sha1").update(Buffer.from(base.data.buffer, base.data.byteOffset, base.data.length)).digest("hex");
  const { views, n } = inferenceViews(base);
  const vsha = createHash("sha1").update(Buffer.from(views.buffer)).digest("hex");
  let sum = 0; for (let i = 0; i < views.length; i++) sum += views[i];
  const same = bsha === v.base_sha && vsha === v.views_sha && n === v.n_views;
  ok &&= same;
  console.log(`${v.name.padEnd(22)} base ${base.h}x${base.w} ${bsha === v.base_sha ? "OK" : "DIFF"} | views ${n} ${vsha === v.views_sha ? "OK" : "DIFF"} | sum ${sum.toFixed(3)} vs ${v.views_sum.toFixed(3)} | first ${Array.from(views.slice(0, 3)).map(x => x.toFixed(5))} vs ${v.views_first.slice(0, 3).map(x => x.toFixed(5))}`);
}
console.log(ok ? "VIEWS PARITY OK" : "VIEWS PARITY FAILED"); process.exit(ok ? 0 : 1);
