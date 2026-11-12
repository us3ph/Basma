"""v6 re-fetch: read every image of every build again from its source and store an UNTOUCHED window of its original
pixels next to the 512-px overview (basma/refetch_parts.json says which source units each part handles).

  python -m basma.refetch --part a --manifests '/kaggle/input/**/manifest_*.jsonl' --out /kaggle/working

Output: shards/native_<part>_NNNN.tar with <id>.n.jpg members (JPEG q95 4:4:4, a side x side window of the original
pixels, position seeded by the id) and native_<part>.jsonl (id, original size, window). Images whose longest side is
<= 512 already have their original pixels in the overview, so they get no window. Ids are the sha1[:16] of the raw
source bytes, exactly as in the manifests, so an image is kept only when the source still serves the same bytes."""
import argparse
import collections
import glob
import hashlib
import io
import json
import os
import random
import tarfile
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFile, ImageOps

from . import build_data as B
from . import sources as S
from .resample import MAX_SIDE

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000
SIDE, JPEG_Q = 512, 95


def native_window(raw, rid, side=SIDE):
    """raw bytes -> (jpeg bytes, info) or (None, info). The window is cut from the decoded image after EXIF rotation
    (as the overview was), never resized; side shrinks to the shortest image side when that is smaller."""
    try:
        im = Image.open(io.BytesIO(raw)); fmt = im.format
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if max(w, h) <= MAX_SIDE:
            return None, {"w": w, "h": h, "fmt": fmt, "why": "small"}
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA"))
        im = im.convert("RGB")
        s = min(side, w, h)
        rng = np.random.default_rng(int(rid, 16))
        x, y = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
        win = im.crop((x, y, x + s, y + s))
        buf = io.BytesIO(); win.save(buf, "JPEG", quality=JPEG_Q, subsampling=0)
        return buf.getvalue(), {"w": w, "h": h, "fmt": fmt, "x": x, "y": y, "s": s}
    except Exception as e:  # noqa: BLE001
        return None, {"why": f"{type(e).__name__}: {str(e)[:80]}"}


def _work(item):
    rid, raw = item
    jpg, info = native_window(raw, rid)
    return rid, jpg, info


class NativeWriter:
    def __init__(self, out_dir, part, per_shard=2000):
        self.out_dir, self.part, self.per_shard = out_dir, part, per_shard
        os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
        self.n, self.tar, self.shard_idx, self.in_shard = 0, None, -1, 0
        self.index = open(os.path.join(out_dir, f"native_{part}.jsonl"), "a", encoding="utf-8")

    def add(self, rid, source, jpg, info):
        if self.tar is None or self.in_shard >= self.per_shard:
            if self.tar is not None:
                self.tar.close()
            self.shard_idx += 1; self.in_shard = 0
            self.tar = tarfile.open(os.path.join(self.out_dir, "shards", f"native_{self.part}_{self.shard_idx:04d}.tar"), "w")
        member = f"{rid}.n.jpg"
        ti = tarfile.TarInfo(member); ti.size = len(jpg); ti.mtime = 0
        self.tar.addfile(ti, io.BytesIO(jpg))
        row = {"id": rid, "source": source, "member": member, "shard": f"native_{self.part}_{self.shard_idx:04d}.tar", "bytes": len(jpg), **info}
        self.index.write(json.dumps(row) + "\n")
        self.n += 1; self.in_shard += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
        self.index.close()


def parse_unit(src, kind, unit):
    if kind == "parquet":
        f, rg = unit.rsplit("#", 1); return (f, int(rg))
    if kind == "arrow":
        return (unit.rsplit("/", 1)[0], unit)
    return unit                                   # tar shard path


def main():
    global SIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True); ap.add_argument("--parts", default=os.path.join(os.path.dirname(__file__), "refetch_parts.json"))
    ap.add_argument("--manifests", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--side", type=int, default=SIDE)
    ap.add_argument("--only-sources", default="", help="comma list, for tests"); ap.add_argument("--limit-units", type=int, default=0, help="per source, for tests")
    a = ap.parse_args()
    SIDE = a.side
    t0 = time.time(); log = lambda m: print(f"[{(time.time() - t0) / 60:6.1f} min] {m}", flush=True)
    wanted = {}                                   # id -> source
    for p in sorted(glob.glob(a.manifests, recursive=True)):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); wanted.setdefault(r["id"], r["source"])
    plan = json.load(open(a.parts))["parts"][a.part]
    by_src = collections.defaultdict(list)
    for name, kind, unit in plan["items"]:
        by_src[name].append((kind, unit))
    only = set(filter(None, a.only_sources.split(",")))
    reg = {}
    for pn in ["cf", "recent", "reals", "eval", "v3ai", "v3real", "v4a", "v4b", "v5a", "v5b", "v5c", "v5d", "v5e"]:
        for src, _ in B.PLANS[pn](1.0):
            reg.setdefault(src.name, src)
    log(f"part {a.part}: {len(by_src)} sources, {len(plan['items'])} items, {plan['bytes'] / 1e9:.0f} GB to read; {len(wanted)} wanted ids")
    writer = NativeWriter(a.out, a.part); done = set()
    stats = collections.defaultdict(collections.Counter); errors = []
    pool = Pool(a.workers)
    for name in sorted(by_src, key=lambda n: -sum(1 for _ in by_src[n])):
        if only and name not in only:
            continue
        src = reg[name]; items = by_src[name]
        if items[0][0] == "whole":
            units = src.units(random.Random(0), {})          # the manifests only name a folder: scan the whole source
        else:
            units = [parse_unit(src, kind, unit) for kind, unit in items]
        if a.limit_units:
            units = units[:a.limit_units]
        want_src = sum(1 for i, s in wanted.items() if s == name)
        log(f"{name}: {len(units)} units, {want_src} wanted images")
        for ui, u in enumerate(units):
            batch = []
            for attempt in range(3):
                try:
                    batch = []
                    for raw, meta in src.read(u):
                        rid = hashlib.sha1(raw).hexdigest()[:16]
                        if rid in wanted and rid not in done:
                            done.add(rid); batch.append((rid, raw))
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{name} {u}: {type(e).__name__}: {str(e)[:100]}"
                    if attempt == 2:
                        errors.append(err); log("   READ ERROR " + err)
                    time.sleep(5 * (attempt + 1))
            for rid, jpg, info in pool.imap_unordered(_work, batch, chunksize=4):
                if jpg is None:
                    stats[name][info.get("why", "error")[:20]] += 1
                else:
                    writer.add(rid, name, jpg, info); stats[name]["native"] += 1
            if (ui + 1) % 10 == 0 or ui + 1 == len(units):
                log(f"   {name} unit {ui + 1}/{len(units)} matched={sum(stats[name].values())}/{want_src} native={stats[name]['native']} written={writer.n}")
        log(f"   done {name}: {dict(stats[name])}")
    pool.close(); writer.close()
    summary = {"part": a.part, "written": writer.n, "minutes": (time.time() - t0) / 60, "stats": {k: dict(v) for k, v in stats.items()}, "errors": errors}
    json.dump(summary, open(os.path.join(a.out, f"summary_native_{a.part}.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in summary.items() if k != "stats"}))


if __name__ == "__main__":
    main()
