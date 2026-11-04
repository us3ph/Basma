"""Export the trained detector to ONNX for the browser and quantize it (weight-only 8-bit MatMulNBits), checking
that decisions survive on validation images.

  python scripts/export_onnx.py --model runs/x/best --fusion runs/x/report/fusion.json --data-dirs data/... --img-root data/img --out web/assets/onnx

Input  pixel_values (V, 3, 224, 224) float32 (ImageNet-normalised views). Outputs logit (V,), logit_g, logit_p, patch (V, 256).
The DINOv2 position embeddings are pre-interpolated to 224 and baked in (no Resize op in the graph)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basma.train import load_model  # noqa: E402
from basma.data import EvalDataset, load_manifests, extract_all  # noqa: E402
from basma.score import select_rows  # noqa: E402
from basma.predict import load_fusion  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, pixel_values):
        o = self.m(pixel_values)
        return o["logit"], o["logit_g"], o["logit_p"], o["patch"]


def bake_pos_embed(model, size=224):
    emb = model.backbone.embeddings
    with torch.no_grad():
        dummy = torch.zeros(1, 3, size, size)
        pe = emb.patch_embeddings(dummy)
        pe = torch.cat([emb.cls_token.expand(1, -1, -1), pe], dim=1)
        pos = emb.interpolate_pos_encoding(pe, size, size).detach().clone()
    emb.interpolate_pos_encoding = lambda embeddings, height, width: pos.to(embeddings.dtype)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--fusion", required=True)
    ap.add_argument("--data-dirs", nargs="+", required=True); ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", default="web/assets/onnx"); ap.add_argument("--n", type=int, default=150); ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = load_model(a.model, "cpu")
    bake_pos_embed(model)
    w = Wrapper(model).eval()
    x = torch.randn(2, 3, 224, 224)
    fp32 = out / "model.onnx"
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(w, (x,), str(fp32), input_names=["pixel_values"], output_names=["logit", "logit_g", "logit_p", "patch"],
                          opset_version=a.opset, dynamo=False, do_constant_folding=True,
                          dynamic_axes={"pixel_values": {0: "views"}, "logit": {0: "views"}, "logit_g": {0: "views"}, "logit_p": {0: "views"}, "patch": {0: "views"}})
    print(f"exported fp32 in {time.time() - t0:.0f}s: {fp32.stat().st_size / 1e6:.0f} MB", flush=True)
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    m = onnx.load(str(fp32))
    heads = [n.name for n in m.graph.node if n.op_type == "MatMul" and ("head_g" in n.name or "head_p" in n.name)]
    q = MatMulNBitsQuantizer(m, algo_config=DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, bits=8), nodes_to_exclude=heads)
    q.process(); w8 = out / "model_w8.onnx"; q.model.save_model_to_file(str(w8), use_external_data_format=False)
    print(f"quantized w8 (bs128): {w8.stat().st_size / 1e6:.0f} MB, heads kept fp32: {len(heads)}", flush=True)

    # parity on validation images through the exact inference pipeline
    rows = select_rows(load_manifests(a.data_dirs), ["val"], a.n, seed=3); extract_all(a.data_dirs, a.img_root)
    fusion = load_fusion(a.fusion)
    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]); s8 = ort.InferenceSession(str(w8), providers=["CPUExecutionProvider"])
    ds = EvalDataset(rows, a.img_root, "social")
    rep = {"n": len(rows), "max_abs_logit_diff_fp32": 0.0, "max_abs_logit_diff_w8": 0.0, "verdict_agree_fp32": 0, "verdict_agree_w8": 0, "verdicts": {}}
    vecs = []
    for i in range(len(rows)):
        xv, nv, _, h, wd = ds[i]
        v = xv[:nv].numpy().astype(np.float32)
        with torch.no_grad():
            lt = model(torch.from_numpy(v))["logit"].numpy()
        l32 = s32.run(["logit"], {"pixel_values": v})[0]; l8 = s8.run(["logit"], {"pixel_values": v})[0]
        def verdict(l):
            r = {"lg": float(l[0]), "lc": [float(t) for t in l[1:]], "h": int(h), "w": int(wd)}
            p = fusion.prob(r); return fusion.verdict(p), p
        vt, pt = verdict(lt); v32, _ = verdict(l32); v8, p8 = verdict(l8)
        rep["max_abs_logit_diff_fp32"] = max(rep["max_abs_logit_diff_fp32"], float(np.abs(lt - l32).max()))
        rep["max_abs_logit_diff_w8"] = max(rep["max_abs_logit_diff_w8"], float(np.abs(lt - l8).max()))
        rep["verdict_agree_fp32"] += vt == v32; rep["verdict_agree_w8"] += vt == v8
        rep["verdicts"][vt] = rep["verdicts"].get(vt, 0) + 1
        if len(vecs) < 6:
            vecs.append({"id": rows[i]["id"], "label": rows[i]["label"], "cond": "social", "views_logit_pt": [float(t) for t in lt], "p_ai": pt, "verdict": vt})
    rep["verdict_agree_fp32"] /= len(rows); rep["verdict_agree_w8"] /= len(rows)
    json.dump(rep, open(out / "parity.json", "w"), indent=1); json.dump(vecs, open(out / "test_vectors.json", "w"), indent=1)
    import shutil; shutil.copy(a.fusion, out.parent / "fusion.json")   # web/assets/fusion.json for the page
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
