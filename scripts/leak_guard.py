"""Leak guard: no training image may be a near-copy of a test image.

Every image the model can learn from (train, val, calib rows of all data builds) is compared with every test image:
the Basma Bench held-out pools (eval:* rows) and the NTIRE 2026 public test. Exact copies are already removed by the
builds (same bytes = same id); this also catches resized / recompressed / re-encoded copies, using two 64-bit perceptual
hashes (pHash: DCT of a 32x32 grey image; dHash: gradient of a 9x8 grey image). A training image is excluded when
    pHash distance <= 10 and dHash distance <= 12   (near-identical: pHash <= 4 and dHash <= 6)
to any test image. Flat test images (almost one colour) have meaningless hashes and are left out of the comparison.
Test rows are never touched: the bench stays exactly as it was.

  python scripts/leak_guard.py --data-dirs <build dirs...> --ntire /tmp/ntire/img --out <dir> [--workers 4]
Outputs: exclude_ids.txt (one row id per line; training sets BASMA_EXCLUDE to it), leak_guard.json, LEAKGUARD.md."""
import argparse
import collections
import glob
import io
import json
import os
import sys
import tarfile
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFile
from scipy.fft import dct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from basma.data import load_manifests  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000
P_SIM, D_SIM, P_SAME, D_SAME = 10, 12, 4, 6


def hashes(fh):
    """(pHash, dHash) as uint64, or None if the image cannot be read."""
    try:
        im = Image.open(fh)
        im.draft("L", (128, 128))                       # JPEG: decode at reduced size (much faster, same hash)
        g = im.convert("L")
        a = np.asarray(g.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        low = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
        p = low > np.median(low)
        d8 = np.asarray(g.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        d = d8[:, 1:] > d8[:, :-1]
        pack = lambda bits: int(np.packbits(bits.flatten()).view(">u8")[0])   # noqa: E731
        return pack(p), pack(d)
    except Exception:  # noqa: BLE001
        return None


def hash_tar(path):
    out = []
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile():
                h = hashes(io.BytesIO(tf.extractfile(m).read()))
                if h:
                    out.append((os.path.splitext(os.path.basename(m.name))[0], *h))
    return out


def hash_file(path):
    with open(path, "rb") as fh:
        h = hashes(fh)
    return (os.path.basename(path), *h) if h else None


def popcount(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--ntire", default="", help="directory with the NTIRE public test images")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    rows = load_manifests(a.data_dirs, exclude=False)
    by_id = {r["id"]: r for r in rows}
    tars = sorted({os.path.join(r["dir"], "shards", r["shard"]) for r in rows})
    print(f"{len(rows)} rows in {len(tars)} shards; hashing with {a.workers} workers", flush=True)
    H = {}
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(hash_tar, tars, chunksize=1)):
            for rid, p, d in res:
                H[rid] = (p, d)
            if (i + 1) % 50 == 0 or i + 1 == len(tars):
                print(f"  {i + 1}/{len(tars)} shards, {len(H)} images hashed ({(time.time() - t0) / 60:.1f} min)", flush=True)
        nt = []
        if a.ntire:
            files = sorted(f for f in glob.glob(os.path.join(a.ntire, "**", "*"), recursive=True) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
            nt = [x for x in pool.imap(hash_file, files, chunksize=16) if x]
            print(f"  NTIRE: {len(nt)} of {len(files)} images hashed", flush=True)
    # test side: bench eval rows + NTIRE; flat images are skipped (their hashes match anything flat)
    ref = [(f"bench:{by_id[i]['split'][5:]}", i, p, d) for i, (p, d) in H.items() if by_id[i]["split"].startswith("eval:")]
    ref += [("ntire", n, p, d) for n, p, d in nt]
    flat = lambda p: bin(p).count("1") <= 4 or bin(p).count("1") >= 60   # noqa: E731
    n_flat = sum(flat(p) for _, _, p, _ in ref); ref = [x for x in ref if not flat(x[2])]
    rp = np.array([x[2] for x in ref], dtype=np.uint64); rd = np.array([x[3] for x in ref], dtype=np.uint64)
    cand = [(i, p, d) for i, (p, d) in H.items() if not by_id[i]["split"].startswith("eval:")]
    print(f"comparing {len(cand)} training-side images with {len(ref)} test images ({n_flat} flat test images left out)", flush=True)
    cp = np.array([c[1] for c in cand], dtype=np.uint64); cd = np.array([c[2] for c in cand], dtype=np.uint64)
    hits = []
    for s in range(0, len(cand), 1024):
        dp = popcount(cp[s:s + 1024, None] ^ rp[None, :]); dd = popcount(cd[s:s + 1024, None] ^ rd[None, :])
        for ci, ri in zip(*np.nonzero((dp <= P_SIM) & (dd <= D_SIM))):
            hits.append((s + ci, ri, int(dp[ci, ri]), int(dd[ci, ri])))
    excl, by_pair, same, examples = set(), collections.Counter(), collections.Counter(), []
    for ci, ri, p, d in hits:
        rid = cand[ci][0]; r = by_id[rid]; pool_name = ref[ri][0]
        key = f"{r['source']} ({r['split']}) ~ {pool_name}"
        if rid not in excl:
            by_pair[key] += 1
            if p <= P_SAME and d <= D_SAME:
                same[key] += 1
        excl.add(rid)
        if len(examples) < 40:
            examples.append({"train_id": rid, "train_source": r["source"], "test": pool_name, "test_id": ref[ri][1], "phash_dist": p, "dhash_dist": d})
    open(os.path.join(a.out, "exclude_ids.txt"), "w").write("".join(f"{i}\n" for i in sorted(excl)))
    summary = {"rows": len(rows), "hashed": len(H), "unreadable": len(rows) - len(H), "training_side": len(cand), "test_side": len(ref),
               "flat_test_images_skipped": n_flat, "ntire_images": len(nt), "excluded": len(excl), "near_identical": sum(same.values()),
               "by_source_and_test_pool": dict(by_pair.most_common()), "near_identical_by_pair": dict(same.most_common()),
               "thresholds": {"similar": [P_SIM, D_SIM], "near_identical": [P_SAME, D_SAME]}, "examples": examples, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(os.path.join(a.out, "leak_guard.json"), "w"), indent=1)
    md = [f"# Leak guard\n", f"{len(cand):,} training-side images (train, val, calib) compared with {len(ref):,} test images "
          f"(Basma Bench held-out pools + {len(nt):,} NTIRE public test images).\n",
          f"**Excluded from training: {len(excl):,}** ({sum(same.values()):,} near-identical, the rest very similar).\n",
          "| Training source (split) ~ test pool | Excluded | Near-identical |", "|---|---|---|"]
    md += [f"| {k} | {v} | {same.get(k, 0)} |" for k, v in by_pair.most_common()] or ["| none | 0 | 0 |"]
    open(os.path.join(a.out, "LEAKGUARD.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(json.dumps({k: v for k, v in summary.items() if k not in ("examples", "by_source_and_test_pool", "near_identical_by_pair")}))


if __name__ == "__main__":
    main()
