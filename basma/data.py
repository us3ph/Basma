"""Manifests, tar extraction, the training dataset (attack + views) and the deterministic evaluation dataset."""
import io
import glob
import json
import os
import pickle
import shutil
import tarfile
import zlib

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .attacks import BUCKET_FN, TRAIN_BUCKETS, attack_eval, attack_train
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


def load_native_index(dirs):
    """v6: id -> row of native_*.jsonl (an untouched window of the image's original pixels, member <id>.n.jpg)."""
    idx = {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith("native_") and f.endswith(".jsonl"):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            r = json.loads(line); idx[r["id"]] = r
    return idx


def load_manifests(dirs, exclude=True):
    """All manifest_*.jsonl rows from the given directories (each row gets 'dir'). With exclude, training-side rows (never
    test rows) listed by the leak guard (BASMA_EXCLUDE) are dropped. Rows with a native-pixel window get 'native'."""
    excl = load_exclude() if exclude else set()
    native = load_native_index(dirs)
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
        if r["id"] in native:
            r["native"] = native[r["id"]]["member"]
    if excl:
        print(f"leak guard: {dropped} training-side rows excluded ({len(excl)} ids listed)", flush=True)
    if native:
        print(f"native windows: {sum(1 for r in out if 'native' in r)} of {len(out)} rows have original-pixel windows", flush=True)
    return out


_TAR_INDEX = {}      # img_root -> {member: (shard path, byte offset, size)} when shards are read in place


def _shards(dirs):
    """Tar shards of each data dir; they may sit in <dir>/shards/ or directly in <dir> (Kaggle kernel outputs)."""
    return [os.path.join(sd, f) for d in dirs for sd in (os.path.join(d, "shards"), d) if os.path.isdir(sd) for f in sorted(os.listdir(sd)) if f.endswith(".tar")]


def extract_all(dirs, img_root, log=print):
    """Make every image readable from img_root. When the disk has room, extract the shards into img_root/<id>.jpg (skips
    shards already extracted); otherwise index them and read each image straight from its shard (no extra disk).
    BASMA_IMG_MODE=extract|index forces one."""
    os.makedirs(img_root, exist_ok=True)
    shards = _shards(dirs)
    done_f, idx_f = os.path.join(img_root, ".extracted"), os.path.join(img_root, ".tarindex.pkl")
    done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
    todo = [p for p in shards if os.path.basename(p) not in done]
    mode = os.environ.get("BASMA_IMG_MODE", "auto")
    if mode == "auto":
        need, free = sum(os.path.getsize(p) for p in todo) * 1.05 + 8e9, shutil.disk_usage(img_root).free
        mode = "extract" if not todo or free > need else "index"
        log(f"images: {len(todo)} of {len(shards)} shards to unpack need {need / 1e9:.0f} GB, {free / 1e9:.0f} GB free -> {mode}")
    if mode == "index":
        idx = {}
        for path in shards:
            with tarfile.open(path) as t:
                for ti in t:
                    if ti.isfile():
                        idx[ti.name] = (path, ti.offset_data, ti.size)
        with open(idx_f, "wb") as fh:
            pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _TAR_INDEX[img_root] = idx
        log(f"indexed {len(idx)} images in {len(shards)} shards (read in place)")
        return
    if os.path.exists(idx_f):
        os.remove(idx_f)
    _TAR_INDEX[img_root] = {}
    n = 0
    for path in todo:
        with tarfile.open(path) as t:
            t.extractall(img_root, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(img_root)
        done.add(os.path.basename(path)); n += 1
        with open(done_f, "w") as fh:
            fh.write("\n".join(sorted(done)))
    log(f"extracted {n} new shards into {img_root}")


def open_image(row, img_root, member=None):
    """PIL image of a manifest row (or of the given member, e.g. its native window): the extracted file, or its bytes
    read from the tar shard when shards are indexed."""
    rel = member or (row["member"] if "member" in row else row["id"] + ".jpg")
    idx = _TAR_INDEX.get(img_root)
    if idx is None:                              # another process (score, baselines, DataLoader worker after spawn)
        idx_f = os.path.join(img_root, ".tarindex.pkl")
        idx = _TAR_INDEX[img_root] = pickle.load(open(idx_f, "rb")) if os.path.exists(idx_f) else {}
    hit = idx.get(rel)
    if hit is None:
        return Image.open(os.path.join(img_root, rel))
    path, off, size = hit
    with open(path, "rb") as fh:
        fh.seek(off)
        return Image.open(io.BytesIO(fh.read(size)))


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
            img = open_image(r, self.img_root).convert("RGB")
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        img, bucket = attack_train(img, rng, self.buckets)
        native = None
        if r.get("native"):                      # v6: texture crops come from the untouched original pixels, same edit bucket
            try:
                native = BUCKET_FN[bucket](open_image(r, self.img_root, r["native"]).convert("RGB"), rng)
            except Exception:  # noqa: BLE001
                native = None
        views = train_views(img, rng, self.n_crops, native=native)
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
            img = open_image(r, self.img_root).convert("RGB")
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        seed = eval_seed(r["id"], self.cond)
        img = attack_eval(img, self.cond, seed)
        base = to_base(np.asarray(img))
        native = None
        if r.get("native"):                      # v6: crops from the original-pixel window under the same condition
            try:
                native = np.asarray(attack_eval(open_image(r, self.img_root, r["native"]).convert("RGB"), self.cond, seed))
            except Exception:  # noqa: BLE001
                native = None
        views, info = inference_views(base, native)
        x = torch.zeros(MAX_VIEWS, 3, views.shape[2], views.shape[3])
        x[:views.shape[0]] = torch.from_numpy(views)
        return x, views.shape[0], i, info["h"], info["w"]


def collate_eval(batch):
    xs, nv, idx, h, w = zip(*batch)
    return torch.stack(xs), torch.tensor(nv), torch.tensor(idx), torch.tensor(h), torch.tensor(w)


def image_views(img):
    """PIL image -> (views float32 (V,3,224,224), info); the single-image inference entry point. The global view
    comes from the 512-px base, the texture crops from the image's own pixels (v6: never resized)."""
    arr = np.asarray(img.convert("RGB"))
    base = to_base(arr)
    return inference_views(base, arr if arr.shape != base.shape else None)
