# Running Basma on Kaggle with little GPU quota

Data is built on **CPU kernels** (no GPU quota), training + evaluation on **one GPU kernel** capped by `max_hours`.

1. Push the code to GitHub (the notebooks clone `us3ph/Basma`), or upload `basma_code.zip` (`bash scripts/make_bundle.sh`)
   as a Kaggle dataset named `basma-code` and attach it.
2. Data kernels (run all three at once, Internet **On**, no accelerator):
   `kaggle kernels push -p kaggle/data_cf`, `-p kaggle/data_recent`, `-p kaggle/data_eval`.
   Each writes tar shards + `manifest_<plan>.jsonl` into its output (`/kaggle/working`). Expect 2-4 h each.
3. Train kernel (GPU T4 x2, Internet On, inputs = the three data kernels' outputs, already listed in
   `kaggle/train/kernel-metadata.json`): `kaggle kernels push -p kaggle/train`. It runs `scripts/kaggle_train.sh`:
   extract -> train (time-capped) -> score held-out pools under 10 conditions -> 3 baselines -> fusion ->
   `docs/RESULTS.md`, and packages `basma_model.tgz` (best checkpoint + fusion + report).
4. Download `basma_model.tgz`, then `python scripts/export_onnx.py ...` and `bash scripts/deploy_hf.sh` to publish
   the browser version.

Budgets (`configs/kaggle_2xT4_base.json`): DINOv2-base, batch 48 images x 2 views, 2 epochs over ~250K images,
capped at 3.3 h; scoring + baselines ~45 min.

## Data-kernel timings (half-size builds, 2026-11-04)

| kernel | images | wall time | note |
|---|---|---|---|
| basma-data-cf-half | 62,460 (47,500 AI from 2,382 generators + 14,960 real) | ~50 min | 68 GB of parquet |
| basma-data-eval-half | 23,019 | 42 min | CF-Eval, GenImage, WildRF, 5 unseen-vendor sets, Flickr2K, Open Images |
| basma-data-recent-half | 70,710 (46,201 AI + 24,509 real) | 137 min | 65 of those minutes were Flickr30k: 6,000 members read one at a time from a 4.4 GB zip. `ZipSource.read` now fetches members with 8 threads, so a rebuild takes ~15 min for that source. |

Rapidata preference sets repeat images across pairs (152,881 duplicates were dropped in the recent build), which is why some of them yield only a few hundred unique images per generator.

## Where attached notebook outputs are mounted

A kernel listed in `kernel_sources` is mounted at `/kaggle/input/notebooks/<owner>/<kernel-slug>/` (not `/kaggle/input/<kernel-slug>/`), with its output files at the top level (`manifest_cf.jsonl`, `cf_0000.tar`, ...). The training notebook does not rely on the path: it globs `/kaggle/input/**/manifest_*.jsonl` and writes the three directories into `configs/kaggle_run.json`.

## Version 2 (real-photo diversity)

The first model flagged too many real photos from sources it never trained on (LAION web images 44 %, WildRF social-media photos 13 %, Flickr2K 7 %, Open Images 6 %), because Community Forensics Small only contains four real sources (COCO, LandscapesHQ, FFHQ, VISION) and the training reals were narrow. Version 2 adds:

- `basma-data-reals` (CPU): CC3M web images (collected 2018, before image generators), more Pexels stock photography, VISION phone/camera photos, and a **calibration split** of real photos from sources that are never trained on (Pascal VOC 2012, VizWiz).
- Thresholds for every system: at most 1 % of validation reals **and** at most 1 % of calibration reals flagged (`fusion.thr_rule`).
- `basma-train-v2` (GPU) mounts the first run's output and reuses its baseline scores (`baselines.py --reuse`): the stored images and per-image attack seeds are identical, so only new rows are scored.
