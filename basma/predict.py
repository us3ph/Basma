"""Score one image file: python -m basma.predict --model runs/x/best --fusion runs/x/report/fusion.json image.jpg"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from .data import image_views
from .fusion import Fusion
from .train import load_model


def load_fusion(path):
    d = json.load(open(path))
    return Fusion(np.array(d["coef"]), d["intercept"], np.array(d["iso_x"]), np.array(d["iso_y"]), d["thr_ai"], d["thr_real"])


@torch.no_grad()
def predict(model, fusion, img, device="cpu"):
    views, info = image_views(img)
    lg = model(torch.from_numpy(views).to(device))["logit"].float().cpu().numpy()
    row = {"lg": float(lg[0]), "lc": [float(v) for v in lg[1:]], "h": info["h"], "w": info["w"]}
    p = fusion.prob(row); v = fusion.verdict(p)
    pct = round(100 * p) if v != "Real" else round(100 * (1 - p))
    return {"verdict": v, "percent": pct, "p_ai": round(p, 4), "views": [round(float(x), 3) for x in lg], "size": [info["w"], info["h"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--fusion", required=True); ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    model = load_model(a.model); fusion = load_fusion(a.fusion)
    for p in a.images:
        print(p, json.dumps(predict(model, fusion, Image.open(p))))


if __name__ == "__main__":
    main()
