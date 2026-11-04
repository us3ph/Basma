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
