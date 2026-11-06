"""NTIRE 2026 Robust AI-Generated Image Detection in the Wild (CVPR 2026 workshop), public test set:
2,500 images with labels (deepfakesMSU/NTIRE-RobustAIGenDetection-test-public; label 1 = AI, is_distorted 0/1).

Every image goes through the shipped detector's exact path (base -> 1 global + up to 3 texture views -> model ->
fusion.json), and, for reference, through the Thermostatic baseline. Reported: ROC AUC on the clean images
(is_distorted = 0), on the distorted ones ("robust", the challenge's ranking metric) and on all; AUC per distortion
type; and the shipped AI / Real / Uncertain verdicts. Basma never trained on NTIRE data, so this is a cross-dataset
test, while the challenge teams trained on the NTIRE training set.

  python scripts/bench_ntire.py --model runs/kaggle_v2/best --fusion web/assets/fusion.json \
      --images /tmp/ntire/img --labels /tmp/ntire/test_labels.csv --out /kaggle/working/ntire
"""
import argparse
import ast
import csv
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from basma.data import MAX_VIEWS, collate_eval  # noqa: E402
from basma.predict import load_fusion  # noqa: E402
from basma.resample import to_base  # noqa: E402
from basma.train import load_model  # noqa: E402
from basma.views import inference_views  # noqa: E402

LEADERS = [("MICV (1st)", 0.9978, 0.9738), ("Ant International (2nd)", 0.9973, 0.9731), ("TeleAI-TeleGuard (3rd)", 0.9762, 0.9215),
           ("INTSIG (4th)", 0.9810, 0.9090), ("vincentlc (5th)", 0.9497, 0.8633)]   # open test, Table 3 of arXiv 2604.11487


def open_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"unreadable {path}: {e}", flush=True); return Image.new("RGB", (256, 256), (128, 128, 128))


class BasmaFiles(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        views, info = inference_views(to_base(np.asarray(open_rgb(self.paths[i]))))
        x = torch.zeros(MAX_VIEWS, 3, views.shape[2], views.shape[3]); x[:views.shape[0]] = torch.from_numpy(views)
        return x, views.shape[0], i, info["h"], info["w"]


class PrepFiles(Dataset):
    def __init__(self, paths, prep):
        self.paths, self.prep = paths, prep

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.prep(open_rgb(self.paths[i])), i


def auc(y, s):
    y = np.asarray(y); return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--fusion", required=True)
    ap.add_argument("--images", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="thermostatic", help="thermostatic or none")
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    found = {p.name: p for p in pathlib.Path(a.images).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")}
    rows = list(csv.DictReader(open(a.labels)))
    rows = [r for r in rows if r["image_name"] in found] if a.limit else rows
    if a.limit: rows = rows[:a.limit]
    missing = [r["image_name"] for r in rows if r["image_name"] not in found]
    assert not missing, f"{len(missing)} labelled images not found under {a.images}, e.g. {missing[:3]}"
    for r in rows:
        r["label"] = int(r["label"]); r["is_distorted"] = int(r["is_distorted"])
        r["distortions"] = ast.literal_eval(r["distortions"]) if r["distortions"].strip() else []
    paths = [str(found[r["image_name"]]) for r in rows]
    print(f"{len(rows)} images on {dev}: {sum(r['label'] for r in rows)} AI / {sum(1 - r['label'] for r in rows)} real, "
          f"{sum(r['is_distorted'] for r in rows)} distorted", flush=True)

    t0 = time.time()
    model = load_model(a.model, dev); fusion = load_fusion(a.fusion)
    for x, nv, idx, h, w in DataLoader(BasmaFiles(paths), batch_size=a.batch_size, num_workers=a.workers, collate_fn=collate_eval):
        B, V = x.shape[:2]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = model(x.view(B * V, *x.shape[2:]).to(dev))["logit"].float().view(B, V).cpu().numpy()
        for b in range(B):
            r = rows[int(idx[b])]; k = int(nv[b])
            f = {"lg": float(lg[b, 0]), "lc": [float(v) for v in lg[b, 1:k]], "h": int(h[b]), "w": int(w[b])}
            p = fusion.prob(f)
            r.update(basma=fusion.raw(f), basma_p=p, verdict=fusion.verdict(p), lg=f["lg"], lc=f["lc"], h=f["h"], w=f["w"])
    print(f"basma scored in {(time.time() - t0) / 60:.1f} min", flush=True)

    systems = ["basma", "basma_p"]
    if a.baseline == "thermostatic":
        from basma.baselines import Thermostatic
        t1 = time.time(); bl = Thermostatic(dev)
        for x, idx in DataLoader(PrepFiles(paths, bl.prep), batch_size=32, num_workers=a.workers):
            s = bl(x)
            for b in range(len(idx)): rows[int(idx[b])]["thermostatic"] = float(s[b])
        systems.append("thermostatic"); print(f"thermostatic scored in {(time.time() - t1) / 60:.1f} min", flush=True)

    with open(os.path.join(a.out, "scores.jsonl"), "w") as fh:
        for r in rows: fh.write(json.dumps({k: v for k, v in r.items() if k != ""}) + "\n")

    subsets = {"clean": [r for r in rows if not r["is_distorted"]], "robust": [r for r in rows if r["is_distorted"]], "all": rows}
    res = {"n": len(rows), "auc": {s: {k: auc([r["label"] for r in rs], [r[s] for r in rs]) for k, rs in subsets.items()} for s in systems}}
    kinds = sorted({d for r in subsets["robust"] for d in r["distortions"]})
    res["by_distortion"] = {}
    for d in kinds:
        rs = [r for r in subsets["robust"] if d in r["distortions"]]
        res["by_distortion"][d] = {"n": len(rs), "n_ai": sum(r["label"] for r in rs), **{s: auc([r["label"] for r in rs], [r[s] for r in rs]) for s in systems}}
    res["by_n_distortions"] = {}
    for n in sorted({len(r["distortions"]) for r in subsets["robust"]}):
        rs = [r for r in subsets["robust"] if len(r["distortions"]) == n]
        res["by_n_distortions"][n] = {"n": len(rs), **{s: auc([r["label"] for r in rs], [r[s] for r in rs]) for s in systems}}
    res["verdicts"] = {}
    for k, rs in subsets.items():
        ai = [r for r in rs if r["label"] == 1]; real = [r for r in rs if r["label"] == 0]
        frac = lambda xs, v: round(sum(r["verdict"] == v for r in xs) / max(1, len(xs)), 4)  # noqa: E731
        res["verdicts"][k] = {"ai_called_AI": frac(ai, "AI"), "ai_called_Real": frac(ai, "Real"), "ai_uncertain": frac(ai, "Uncertain"),
                              "real_called_Real": frac(real, "Real"), "real_called_AI": frac(real, "AI"), "real_uncertain": frac(real, "Uncertain")}
    json.dump(res, open(os.path.join(a.out, "ntire_results.json"), "w"), indent=1)

    f4 = lambda v: "–" if v is None else f"{v:.4f}"  # noqa: E731
    name = {"basma": "**Basma** (fused score)", "basma_p": "Basma (calibrated probability)", "thermostatic": "Thermostatic (Aug 2026)"}
    L = ["# NTIRE 2026 Robust AI-Generated Image Detection in the Wild: public test", "",
         f"{len(rows)} images ({sum(r['label'] for r in rows)} AI, {sum(1 - r['label'] for r in rows)} real; half distorted). ROC AUC, higher is better. "
         "Clean = images with is_distorted 0, Robust = is_distorted 1 (the challenge ranks by Robust), All = both. "
         "Basma never saw NTIRE data; the challenge teams trained on the NTIRE training set (same pipeline, 20 of its generators).", "",
         "| System | Clean | Robust | All |", "|---|---|---|---|"]
    for s in systems: L.append(f"| {name[s]} | {f4(res['auc'][s]['clean'])} | {f4(res['auc'][s]['robust'])} | {f4(res['auc'][s]['all'])} |")
    for n_, c, rb in LEADERS: L.append(f"| {n_}, challenge entry | {c:.4f} | {rb:.4f} | |")
    L += ["", "Challenge numbers: open test, Table 3 of arXiv 2604.11487 (its \"ROC AUC\" column is shown under Clean).", "",
          "## Robust AUC per distortion (an image can carry several)", "", "| Distortion | Images | " + " | ".join(name[s] for s in systems) + " |",
          "|---|---|" + "---|" * len(systems)]
    for d, v in sorted(res["by_distortion"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {d} | {v['n']} | " + " | ".join(f4(v[s]) for s in systems) + " |")
    L += ["", "## Robust AUC by number of distortions", "", "| Distortions | Images | " + " | ".join(name[s] for s in systems) + " |", "|---|---|" + "---|" * len(systems)]
    for n, v in res["by_n_distortions"].items():
        L.append(f"| {n} | {v['n']} | " + " | ".join(f4(v[s]) for s in systems) + " |")
    L += ["", "## Basma's shipped verdicts (thresholds from fusion.json, AI at 1 % FPR on its own calibration reals)", "",
          "| Subset | AI images called AI | AI called Real | AI Uncertain | Real images called Real | Real called AI | Real Uncertain |", "|---|---|---|---|---|---|---|"]
    pc = lambda v: f"{100 * v:.1f} %"  # noqa: E731
    for k, v in res["verdicts"].items():
        L.append(f"| {k} | {pc(v['ai_called_AI'])} | {pc(v['ai_called_Real'])} | {pc(v['ai_uncertain'])} | {pc(v['real_called_Real'])} | {pc(v['real_called_AI'])} | {pc(v['real_uncertain'])} |")
    open(os.path.join(a.out, "NTIRE.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
