"""Score images (every view) under evaluation conditions and write one JSONL row per image x condition.

  python -m basma.score --model runs/x/best --data-dirs data/a data/b --img-root data/img --splits val eval:* --out runs/x/scores
Row: id label source gen family vendor split cond lg lc[] n_crops h w"""
import argparse
import fnmatch
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .attacks import EVAL_CONDITIONS
from .data import EvalDataset, collate_eval, extract_all, load_manifests
from .train import load_model


def select_rows(rows, splits, per_split_limit=0, seed=0):
    out = []
    for sp in sorted({r["split"] for r in rows}):
        if not any(fnmatch.fnmatch(sp, pat) for pat in splits):
            continue
        rs = [r for r in rows if r["split"] == sp]
        if per_split_limit and len(rs) > per_split_limit:
            rng = np.random.RandomState(seed); rng.shuffle(rs); rs = rs[:per_split_limit]
        out += rs
    return out


@torch.no_grad()
def score_rows(model, rows, img_root, conds, out_path, device, batch_size=16, workers=4, fp16=True, log=print):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        for cond in conds:
            dl = DataLoader(EvalDataset(rows, img_root, cond), batch_size=batch_size, num_workers=workers, collate_fn=collate_eval)
            n = 0
            for x, nv, idx, h, w in dl:
                B, V = x.shape[:2]
                flat = x.view(B * V, *x.shape[2:]).to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                    o = model(flat)
                lg = o["logit"].float().view(B, V).cpu().numpy()
                for b in range(B):
                    r = rows[int(idx[b])]; k = int(nv[b])
                    fh.write(json.dumps({"id": r["id"], "label": r["label"], "source": r["source"], "gen": r["gen"], "family": r["family"], "vendor": r["vendor"],
                                         "split": r["split"], "cond": cond, "lg": float(lg[b, 0]), "lc": [float(v) for v in lg[b, 1:k]], "n_crops": k - 1,
                                         "h": int(h[b]), "w": int(w[b])}) + "\n")
                n += B
            log(f"  scored {cond}: {n} images ({(time.time() - t0) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "calib", "eval:*"])
    ap.add_argument("--conds", nargs="+", default=list(EVAL_CONDITIONS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max images per split")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifests(a.data_dirs); extract_all(a.data_dirs, a.img_root)
    rows = select_rows(rows, a.splits, a.limit)
    print(f"scoring {len(rows)} images x {len(a.conds)} conditions on {device}", flush=True)
    model = load_model(a.model, device)
    score_rows(model, rows, a.img_root, a.conds, os.path.join(a.out, "scores.jsonl"), device, a.batch_size, a.workers)


if __name__ == "__main__":
    main()
