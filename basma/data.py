"""Manifests, tar extraction, the training dataset (attack + views) and the deterministic evaluation dataset."""
import io
import glob
import json
import os
import tarfile
import zlib

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .attacks import TRAIN_BUCKETS, attack_eval, attack_train
from .resample import to_base
from .views import inference_views, pil_to_tensor, train_views

ImageFile.LOAD_TRUNCATED_IMAGES = True
BUCKET_IDS = {b: i for i, b in enumerate(TRAIN_BUCKETS)}
MAX_VIEWS = 4   # global + up to 3 texture crops


def load_exclude():
    """Row ids the leak guard found to be near-copies of test images (BASMA_EXCLUDE: file(s), glob allowed)."""
    ids, pattern = set(), os.environ.get("BASMA_EXCLUDE", "")
    for f in (glob.glob(pattern, recursive=True) if pattern else []):
        ids.update(line.strip() for line in open(f) if line.strip())
    return ids


def load_manifests(dirs, exclude=True):
    """All manifest_*.jsonl rows from the given directories (each row gets 'dir'). With exclude, training-side rows (never
    test rows) listed by the leak guard (BASMA_EXCLUDE) are dropped."""
    excl = load_exclude() if exclude else set()
    rows = []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith("manifest_") and f.endswith(".jsonl"):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            r = json.loads(line); r["dir"] = d; rows.append(r)
    # drop content duplicates across plans (same id = same raw bytes)
    seen, out, dropped = set(), [], 0
    for r in rows:
        if r["id"] in seen:
            continue
        if r["id"] in excl and not r["split"].startswith("eval:"):
            dropped += 1; continue
        seen.add(r["id"]); out.append(r)
    if excl:
        print(f"leak guard: {dropped} training-side rows excluded ({len(excl)} ids listed)", flush=True)
    return out


def extract_all(dirs, img_root, log=print):
    """Extract every tar shard into img_root/<id>.jpg (skips shards already extracted)."""
    os.makedirs(img_root, exist_ok=True)
    done_f = os.path.join(img_root, ".extracted")
    done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
    n = 0
    for d in dirs:
        cands = [os.path.join(sd, f) for sd in (os.path.join(d, "shards"), d) if os.path.isdir(sd) for f in sorted(os.listdir(sd)) if f.endswith(".tar")]
        for path in cands:                      # shards may sit in <dir>/shards/ or directly in <dir> (Kaggle kernel outputs)
            f = os.path.basename(path)
            if f in done:
                continue
            with tarfile.open(path) as t:
                t.extractall(img_root, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(img_root)
            done.add(f); n += 1
            with open(done_f, "w") as fh:
                fh.write("\n".join(sorted(done)))
    log(f"extracted {n} new shards into {img_root}")


def path_of(row, img_root):
    return os.path.join(img_root, row["member"] if "member" in row else row["id"] + ".jpg")


def group_key(row):
    return f"{row['label']}:{row['family']}"


def balanced_weights(rows, power=0.5):
    """Sampling weight per row: labels 50/50, families within a label weighted by count^power (flatter than natural)."""
    from collections import Counter
    cnt = Counter(group_key(r) for r in rows)
    lab_tot = {0: 0.0, 1: 0.0}
    gw = {}
    for g, n in cnt.items():
        lab = int(g.split(":")[0]); gw[g] = n ** power; lab_tot[lab] += gw[g]
    w = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        g = group_key(r); w[i] = (gw[g] / lab_tot[r["label"]]) / cnt[g] * 0.5
    return w


class TrainDataset(Dataset):
    def __init__(self, rows, img_root, n_crops=1, seed=0, buckets=TRAIN_BUCKETS):
        self.rows, self.img_root, self.n_crops, self.seed, self.buckets = rows, img_root, n_crops, seed, buckets
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 7919 + i) & 0xFFFFFFFF)
        try:
            img = Image.open(path_of(r, self.img_root)).convert("RGB")
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        img, bucket = attack_train(img, rng, self.buckets)
        views = train_views(img, rng, self.n_crops)
        x = torch.from_numpy(np.stack([pil_to_tensor(v) for v in views]))
        return x, torch.tensor(float(r["label"])), BUCKET_IDS[bucket], i


def collate_train(batch):
    xs, ys, bs, idx = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.tensor(bs), torch.tensor(idx)


def eval_seed(row_id, cond):
    return zlib.crc32(f"{row_id}:{cond}".encode()) & 0xFFFFFFFF


class EvalDataset(Dataset):
    """Deterministic: condition applied with a per-image seed, then the exact inference pipeline (base -> views)."""
    def __init__(self, rows, img_root, cond="clean"):
        self.rows, self.img_root, self.cond = rows, img_root, cond

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = Image.open(path_of(r, self.img_root)).convert("RGB")
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        img = attack_eval(img, self.cond, eval_seed(r["id"], self.cond))
        base = to_base(np.asarray(img))
        views, info = inference_views(base)
        x = torch.zeros(MAX_VIEWS, 3, views.shape[2], views.shape[3])
        x[:views.shape[0]] = torch.from_numpy(views)
        return x, views.shape[0], i, info["h"], info["w"]


def collate_eval(batch):
    xs, nv, idx, h, w = zip(*batch)
    return torch.stack(xs), torch.tensor(nv), torch.tensor(idx), torch.tensor(h), torch.tensor(w)


def image_views(img):
    """PIL image -> (views float32 (V,3,224,224), info); the single-image inference entry point."""
    base = to_base(np.asarray(img.convert("RGB")))
    return inference_views(base)
