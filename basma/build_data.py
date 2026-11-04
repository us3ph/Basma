"""Build the normalised image corpus for one plan on a CPU machine (Kaggle CPU kernel): stream the sources, keep a
seeded sample, re-encode EVERY image the same way (longest side <= 512, JPEG q90 4:4:4) so neither file format nor
resolution can be a shortcut, pack into tar shards and write a manifest.

  python -m basma.build_data --plan mini --out data/mini --workers 4
  python -m basma.build_data --plan cf|recent|eval --out /kaggle/working/out

Manifest (JSONL): id label source gen family vendor split plan shard member w h orig_w orig_h orig_fmt orig_bytes bytes unit extra
Splits: train / val (8 % of train sources by hash) / eval:<pool>; AI images of held-out vendor families always go to
eval:unseen_vendors whatever the source."""
import argparse
import collections
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from PIL import Image, ImageFile, ImageOps

from .generators import canon, is_holdout
from .resample import MAX_SIDE, fit_max_side
from . import sources as S

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000
JPEG_Q = 90
MIN_SIDE = 96


def normalize(raw):
    """raw bytes -> (jpeg bytes, meta) or (None, {'error': ...})."""
    try:
        im = Image.open(io.BytesIO(raw))
        fmt, (w, h) = im.format, im.size
        if min(w, h) < MIN_SIDE:
            return None, {"error": "too small"}
        if fmt == "JPEG" and max(w, h) >= 4 * MAX_SIDE:
            im.draft("RGB", (2 * MAX_SIDE, 2 * MAX_SIDE))      # decode at reduced scale, much faster for big JPEGs
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
        im = im.convert("RGB")
        th, tw = fit_max_side(im.height, im.width, MAX_SIDE)
        if (th, tw) != (im.height, im.width):
            im = im.resize((tw, th), Image.LANCZOS, reducing_gap=3.0)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_Q, subsampling=0)
        return buf.getvalue(), {"orig_w": w, "orig_h": h, "orig_fmt": fmt or "", "w": im.width, "h": im.height, "orig_bytes": len(raw)}
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"{type(e).__name__}: {str(e)[:80]}"}


def _work(item):
    idx, raw = item
    jpg, meta = normalize(raw)
    return idx, jpg, meta


class ShardWriter:
    def __init__(self, out_dir, plan, per_shard=1500):
        self.out_dir, self.plan, self.per_shard = out_dir, plan, per_shard
        os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
        self.n, self.tar, self.shard_idx, self.in_shard = 0, None, -1, 0
        self.man = open(os.path.join(out_dir, f"manifest_{plan}.jsonl"), "a", encoding="utf-8")

    def _open(self):
        if self.tar is not None:
            self.tar.close()
        self.shard_idx += 1
        self.in_shard = 0
        self.tar = tarfile.open(os.path.join(self.out_dir, "shards", f"{self.plan}_{self.shard_idx:04d}.tar"), "w")

    def add(self, row, jpg):
        if self.tar is None or self.in_shard >= self.per_shard:
            self._open()
        member = f"{row['id']}.jpg"
        ti = tarfile.TarInfo(member); ti.size = len(jpg); ti.mtime = 0
        self.tar.addfile(ti, io.BytesIO(jpg))
        row["shard"], row["member"], row["bytes"] = f"{self.plan}_{self.shard_idx:04d}.tar", member, len(jpg)
        self.man.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n += 1; self.in_shard += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
        self.man.close()


def assign_split(src, label, family, sha):
    if src.split.startswith("eval:") or src.split == "calib":   # calib: real photos never trained on, used only to set thresholds
        return src.split
    if label == 1 and is_holdout(family):
        return "eval:unseen_vendors"
    return "val" if int(sha[:8], 16) % 100 < 8 else "train"


def _reader(src, units, q, chunk, stop):
    """Background thread: stream every unit of a source into chunks of (raw, meta) items (bounded queue)."""
    try:
        buf = []
        for u in units:
            if stop.is_set():
                break
            try:
                for item in src.read(u):
                    buf.append(item)
                    if len(buf) >= chunk:
                        q.put(buf); buf = []
                    if stop.is_set():
                        break
            except Exception as e:  # noqa: BLE001
                q.put(("error", f"{src.name} {u}: {type(e).__name__}: {str(e)[:120]}"))
            q.put(("unit_done", u))
        if buf:
            q.put(buf)
    finally:
        q.put(None)


def build(plan_name, plan, out_dir, workers=4, seed=0, log=print, chunk=64):
    import queue
    import threading
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    writer = ShardWriter(out_dir, plan_name)
    seen = set()
    stats = collections.Counter(); errors = collections.Counter()
    t0 = time.time()
    skip = set(filter(None, os.environ.get("BASMA_SKIP", "").split(",")))
    with Pool(workers) as pool:
        for src, budget in plan:
            if src.name in skip:
                log(f"skipping {src.name} (BASMA_SKIP)"); continue
            units = src.units(rng, budget)
            max_images = budget.get("max_images", 10 ** 9)
            cap_gen = budget.get("cap_per_gen", 10 ** 9)
            cap_label = budget.get("cap_per_label", {})
            n_src, n_gen, n_label, raw_bytes, n_units = 0, collections.Counter(), collections.Counter(), 0, 0
            log(f"[{(time.time() - t0) / 60:6.1f} min] {src.name}: {len(units)} units")
            q = queue.Queue(maxsize=6); stop = threading.Event()
            th = threading.Thread(target=_reader, args=(src, units, q, chunk, stop), daemon=True); th.start()
            while True:
                items = q.get()
                if items is None:
                    break
                if isinstance(items, tuple):
                    if items[0] == "error":
                        errors["read:" + items[1][:60]] += 1; log("   READ ERROR " + items[1])
                    else:
                        n_units += 1
                        if n_units % 5 == 0 or n_units == len(units):
                            log(f"   {src.name} unit {n_units}/{len(units)} kept={n_src} labels={dict(n_label)} raw={raw_bytes / 1e9:.1f} GB written={writer.n} ({(time.time() - t0) / 60:.0f} min)")
                    continue
                if stop.is_set():
                    continue
                todo = []
                for raw, meta in items:
                    raw_bytes += len(raw)
                    sha = hashlib.sha1(raw).hexdigest()
                    if sha in seen:
                        stats["dup"] += 1; continue
                    label = int(meta["label"])
                    if label == 1:
                        gen, family, vendor = canon(meta["gen"])
                        if src.family_override and not is_holdout(family):
                            family = src.family_override
                    else:
                        gen, family, vendor = str(meta["gen"] or f"real:{src.name}"), (src.family_override or f"real:{src.name}"), "real"
                    gkey = f"{label}:{gen}"                                   # caps are per (label, generator)
                    if n_gen[gkey] >= cap_gen or n_label[label] >= cap_label.get(label, 10 ** 9):
                        stats["capped"] += 1; continue
                    seen.add(sha); n_gen[gkey] += 1; n_label[label] += 1
                    todo.append((sha, raw, dict(meta, gen=gen, family=family, vendor=vendor, label=label)))
                    if n_src + len(todo) >= max_images:
                        stop.set(); break
                for i, jpg, m in pool.imap(_work, [(i, raw) for i, (_, raw, _) in enumerate(todo)], chunksize=4):
                    sha, _, meta = todo[i]
                    if jpg is None:
                        errors[m["error"][:40]] += 1; n_gen[f"{meta['label']}:{meta['gen']}"] -= 1; n_label[meta["label"]] -= 1; continue
                    row = {"id": sha[:16], "label": meta["label"], "source": src.name, "gen": meta["gen"], "family": meta["family"], "vendor": meta["vendor"],
                           "split": assign_split(src, meta["label"], meta["family"], sha), "plan": plan_name, "unit": meta.get("unit", ""), "extra": meta.get("extra", {})}
                    row.update(m)
                    writer.add(row, jpg)
                    n_src += 1; stats[f"{src.name}:{meta['label']}"] += 1
            th.join()
            log(f"   done {src.name}: {n_src} images, {len([g for g in n_gen if n_gen[g] > 0])} generators, {raw_bytes / 1e9:.1f} GB read")
    writer.close()
    summary = {"plan": plan_name, "images": writer.n, "minutes": (time.time() - t0) / 60, "stats": dict(stats), "errors": dict(errors.most_common(10))}
    json.dump(summary, open(os.path.join(out_dir, f"summary_{plan_name}.json"), "w"), indent=1)
    log(json.dumps(summary))
    return summary


# ---------------------------------------------------------------- plans
def rapidata(name, repo, split="train", **budget):
    return (S.ParquetSource(name, repo, [("image1", 1, None), ("image2", 1, None)], split=split, meta_cols=("model1", "model2"), row_meta=S.rapidata_row_meta), budget)


def plan_cf(scale=1.0):
    """Community Forensics small: thousands of generators (all AI shards, <= 20 images per model) + its real photos."""
    sh = json.load(open(os.path.join(os.path.dirname(__file__), "cf_small_shards.json")))
    k = lambda n: max(1, int(n * scale))
    ai = S.ParquetSource("cf_small_ai", "OwensLab/CommunityForensics-Small", [("image_data", None, None)], files=sh["ai_shards"],
                         meta_cols=("model_name", "real_source", "label", "architecture", "format"), row_meta=S.cf_row_meta, family_override="cf")
    real = S.ParquetSource("cf_small_real", "OwensLab/CommunityForensics-Small", [("image_data", None, None)], files=sh["real_shards"],
                           meta_cols=("model_name", "real_source", "label", "architecture", "format"), row_meta=S.cf_row_meta, family_override="cf")
    return [(ai, {"max_files": k(len(sh["ai_shards"])), "max_images": k(95_000), "cap_per_gen": 20}),
            (real, {"max_files": k(10), "max_images": k(30_000)})]


def plan_recent(scale=1.0):
    """Recent commercial generators (Rapidata pairs, GPT-image-2, VCT2) + diverse real photos."""
    k = lambda n: max(1, int(n * scale))
    coco_ai = S.ParquetSource("vct2", "NasrinImp/COCO_AI", [("coco_image", 0, "coco"), ("sd35_image", 1, "sd35"), ("sd3_image", 1, "sd3"), ("sd21_image", 1, "sd21"),
                                                            ("sdxl_image", 1, "sdxl"), ("dalle_image", 1, "dall-e-3"), ("midjourney_image", 1, "midjourney-6")])
    oip = S.ParquetSource("open_image_prefs", "data-is-better-together/open-image-preferences-v1",
                          [("image_quality_dev", 1, "flux-1-dev"), ("image_simplified_dev", 1, "flux-1-dev"), ("image_quality_sd", 1, "sd3.5-xl"), ("image_simplified_sd", 1, "sd3.5-xl")])
    plan = [
        rapidata("rd_flux2pro", "Rapidata/Flux-2-pro_t2i_human_preference", max_files=k(36), cap_per_gen=4000),
        rapidata("rd_4o", "Rapidata/OpenAI-4o_t2i_human_preference", max_files=k(22), cap_per_gen=4000),
        rapidata("rd_imagen4", "Rapidata/Imagen4_t2i_human_preference", max_files=k(24), cap_per_gen=4000),
        rapidata("rd_imagen4u", "Rapidata/Imagen-4-ultra-24-7-25_t2i_human_preference", max_files=k(30), cap_per_gen=4000),
        rapidata("rd_hidream", "Rapidata/Hidream_t2i_human_preference", max_files=k(23), cap_per_gen=4000),
        rapidata("rd_ideogram2", "Rapidata/Ideogram-V2_t2i_human_preference", max_files=k(30), cap_per_gen=4000),
        rapidata("rd_frames", "Rapidata/Runway_Frames_t2i_human_preferences", max_files=k(40), cap_per_gen=4000),
        rapidata("rd_lumina", "Rapidata/OpenGVLab_Lumina_t2i_human_preference", max_files=k(20), cap_per_gen=4000),
        rapidata("rd_banana", "Rapidata/bananamark-dataset", max_files=k(4), cap_per_gen=4000),
        rapidata("rd_faces", "Rapidata/Face_Generation_Benchmark", max_files=k(4), cap_per_gen=4000),
        rapidata("rd_fluxsd3mj", "Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3", max_files=k(10), cap_per_gen=4000),
        rapidata("rd_flux11", "Rapidata/117k_human_preferences_flux1.0_V_flux1.1Blueberry", max_files=k(4), cap_per_gen=2500),
        rapidata("rd_bench2026", "Rapidata/text-to-image-human-feedback-benchmark", max_files=k(140), cap_per_gen=3000),   # ~85 unique images per shard
        (S.HfFilesSource("gpt_image_2", "Goku-OpenLab/gpt-image-2-prompts-datasets", 1, "gpt-image-2", prefix="gpt-image-2/images/0/"), {"max_images": k(7000)}),
        (S.ZipSource("gpt_imgeval", 1, gen="gpt-4o-image", repo="Yejy53/GPT-ImgEval", zip_path="GenEval.zip"), {"max_images": k(1500)}),
        (coco_ai, {"max_files": k(13), "rgs_per_file": 8, "cap_per_gen": 6000, "max_images": k(40_000)}),
        (oip, {"max_files": k(4), "rgs_per_file": 4, "max_images": k(6000)}),
        # real photos
        (S.ParquetSource("coco", "detection-datasets/coco", [("image", 0, "coco")], file_filter=lambda f: "/train-" in f), {"max_files": k(4), "max_images": k(12_000)}),
        (S.ZipSource("flickr30k", 0, gen="flickr30k", repo="nlphuji/flickr30k", zip_path="flickr30k-images.zip"), {"max_images": k(12_000)}),
        (S.ParquetSource("pexels", "ujin-song/pexels-image-60k", [("image", 0, "pexels")]), {"max_files": k(14), "max_images": k(5000)}),
        (S.ParquetSource("wikiart", "huggan/wikiart", [("image", 0, "wikiart")]), {"max_files": k(7), "max_images": k(7000)}),
        (S.ParquetSource("fashion", "benitomartin/fashion-product-images-small-384x512", [("image", 0, "fashion-product")]), {"max_files": k(5), "rgs_per_file": 2, "max_images": k(8000)}),
    ]
    return plan


def plan_reals(scale=1.0):
    """Real photos the first model lacked, which is where its false positives came from (LAION-style web images 44 %,
    social-media photos 13 %, high-end photography 7 %): CC3M web images (collected 2018, before image generators),
    more Pexels stock photography, VISION phone/camera photos. Plus a calibration split of real photos from sources
    that are never trained on (Pascal VOC 2012, VizWiz phone photos): thresholds must hold on them too."""
    k = lambda n: max(1, int(n * scale))
    sh = json.load(open(os.path.join(os.path.dirname(__file__), "cf_small_shards.json")))
    cf_vision = S.ParquetSource("cf_vision", "OwensLab/CommunityForensics-Small", [("image_data", None, None)], files=sh["real_by_source"]["VISION"],
                                meta_cols=("model_name", "real_source", "label", "architecture", "format"), row_meta=S.cf_row_meta, family_override="real:cf_vision")
    return [
        (S.TarSource("cc3m", "pixparse/cc3m-wds", 0, "cc3m", prefix="cc3m-train-", per_shard=400), {"max_files": k(40), "max_images": k(15_000)}),
        (S.ParquetSource("pexels_more", "ujin-song/pexels-image-60k", [("image", 0, "pexels")], family_override="real:pexels"), {"max_files": k(24), "max_images": k(8000)}),
        (cf_vision, {"max_images": k(6000)}),
        (S.ParquetSource("voc2012", "nateraw/pascal-voc-2012", [("image", 0, "voc2012")], split="calib"), {"max_images": k(2900)}),
        (S.ParquetSource("vizwiz", "Multimodal-Fatima/VizWiz_train", [("image", 0, "vizwiz")], split="calib"), {"max_files": k(3), "max_images": k(2000)}),
    ]


def plan_eval(scale=1.0, tmp="/tmp/basma_dl"):
    """Held-out pools: CF-Eval (21 unseen generators + RAISE/COCO/FFHQ reals), GenImage test, WildRF, unseen-vendor
    Rapidata sets, unseen real sources (Flickr2K, Open Images)."""
    k = lambda n: max(1, int(n * scale))
    os.makedirs(tmp, exist_ok=True)
    cfe = S.ParquetSource("cf_eval", "OwensLab/CommunityForensics-Eval", [("image_data", None, None)], split="eval:cf_eval",
                          meta_cols=("model_name", "real_source", "label", "architecture", "format", "subset"), row_meta=S.cf_row_meta, family_override="cf_eval")
    gens = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong"]
    genimage = S.ArrowSource("genimage", "nebula/GenImage-arrow", [f"data/test/{g}" for g in gens[:max(1, int(len(gens) * min(1.0, scale * 8)))]], split="eval:genimage", files_per_dir=1, family_override="genimage")
    def wild_meta(n):
        parts = n.split("/")                       # WildRF/test/<platform>/<0_real|1_fake>/x.jpg or WildRF/<train|val>/<0_real|1_fake>/x.jpg
        fake = "/1_fake/" in n
        where = parts[2] if len(parts) > 4 and parts[1] == "test" else parts[1]
        return {"label": 1 if fake else 0, "gen": f"wildrf:{'fake' if fake else 'real'}:{where}"}
    wild = S.GDriveZipSource("wildrf", "1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn", os.path.join(tmp, "WildRF.zip"), label=None, split="eval:wildrf",
                             member_meta=wild_meta, family_override="wildrf")
    plan = [
        (cfe, {"max_files": k(413), "rgs_per_file": 1, "max_unit_mb": 200, "keep_big": k(14), "max_images": k(9000), "cap_per_gen": 450}),
        (genimage, {"max_images": k(16_000), "cap_per_gen": 1000, "cap_per_label": {0: 8000, 1: 8000}}),
        (wild, {"max_images": k(6000)}),
        rapidata("rd_seedream3", "Rapidata/Seedream-3_t2i_human_preference", max_files=k(14), cap_per_gen=1200),
        rapidata("rd_hunyuan21", "Rapidata/HunyuanImage-2.1_t2i_human_preference", max_files=k(12), cap_per_gen=1200),
        rapidata("rd_recraft3", "Rapidata/Recraft-v3-24-7-25_t2i_human_preference", max_files=k(12), cap_per_gen=1200),
        rapidata("rd_reve", "Rapidata/Reve-AI-Halfmoon_t2i_human_preference", max_files=k(15), cap_per_gen=1200),
        rapidata("rd_aurora", "Rapidata/xAI_Aurora_t2i_human_preferences", max_files=k(20), cap_per_gen=1200),
        (S.ZipSource("flickr2k", 0, gen="flickr2k", split="eval:real_flickr2k", repo="yangtao9009/Flickr2K", zip_path="Flickr2K.zip"), {"max_images": k(1000)}),
        (S.ParquetSource("openimages", "finedet/openimages", [("image", 0, "openimages")], split="eval:real_openimages"), {"max_files": k(3), "rgs_per_file": 6, "max_images": k(4000)}),
    ]
    return plan


def plan_mini():
    """Tiny end-to-end check of every reader type on a laptop (a few hundred images)."""
    sh = json.load(open(os.path.join(os.path.dirname(__file__), "cf_small_shards.json")))
    cf = S.ParquetSource("cf_small_ai", "OwensLab/CommunityForensics-Small", [("image_data", None, None)], files=sh["ai_shards"][:2],
                         meta_cols=("model_name", "real_source", "label", "architecture", "format"), row_meta=S.cf_row_meta, family_override="cf")
    coco_ai = S.ParquetSource("vct2", "NasrinImp/COCO_AI", [("coco_image", 0, "coco"), ("sd35_image", 1, "sd35"), ("dalle_image", 1, "dall-e-3"), ("midjourney_image", 1, "midjourney-6")])
    return [
        (cf, {"max_files": 1, "max_images": 60, "cap_per_gen": 10}),
        rapidata("rd_4o", "Rapidata/OpenAI-4o_t2i_human_preference", max_files=1, rgs_per_file=1, max_images=40, cap_per_gen=20),
        rapidata("rd_seedream3", "Rapidata/Seedream-3_t2i_human_preference", max_files=1, rgs_per_file=1, max_images=30, cap_per_gen=20),
        (coco_ai, {"max_files": 1, "rgs_per_file": 1, "max_images": 60, "cap_per_gen": 15}),
        (S.HfFilesSource("gpt_image_2", "Goku-OpenLab/gpt-image-2-prompts-datasets", 1, "gpt-image-2", prefix="gpt-image-2/images/4/"), {"max_images": 24}),
        (S.ZipSource("flickr30k", 0, gen="flickr30k", repo="nlphuji/flickr30k", zip_path="flickr30k-images.zip"), {"max_images": 40}),
        (S.ParquetSource("fashion", "benitomartin/fashion-product-images-small-384x512", [("image", 0, "fashion-product")]), {"max_files": 1, "rgs_per_file": 1, "max_images": 30}),
        (S.ZipSource("flickr2k", 0, gen="flickr2k", split="eval:real_flickr2k", repo="yangtao9009/Flickr2K", zip_path="Flickr2K.zip"), {"max_images": 12}),
    ]


PLANS = {"cf": plan_cf, "recent": plan_recent, "reals": plan_reals, "eval": plan_eval, "mini": lambda scale=1.0: plan_mini()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=list(PLANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink every budget (0.02 = quick smoke test of a real plan)")
    a = ap.parse_args()
    log_f = open(os.path.join(a.out, f"build_{a.plan}.log"), "a", encoding="utf-8") if os.path.isdir(a.out) else None
    def log(msg):
        print(msg, flush=True)
        if log_f: log_f.write(msg + "\n"); log_f.flush()
    os.makedirs(a.out, exist_ok=True)
    build(a.plan, PLANS[a.plan](a.scale), a.out, workers=a.workers, seed=a.seed, log=log)


if __name__ == "__main__":
    main()
