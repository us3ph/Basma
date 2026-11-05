# Basma

AI-generated image detector that still works after the image was **posted, resized, colour-edited or screenshotted**.
Built like [Skry](https://github.com/us3ph/Skry): attack-style training, held-out evaluation at a fixed false-positive
rate, public baselines scored on the same images, and a browser-only deployment (nothing leaves your device).

* Design: [docs/DESIGN.md](docs/DESIGN.md) · Results: [docs/RESULTS.md](docs/RESULTS.md) · Kaggle: [docs/KAGGLE.md](docs/KAGGLE.md)
* Labels: `0` real, `1` AI. Verdicts: **AI** / **Real** / **Uncertain** with a calibrated percentage.

```bash
pip install -e .                                   # or: pip install -r requirements.txt
python -m basma.build_data --plan mini --out data/mini         # small corpus, every reader
python -m basma.train --config configs/cpu_mini.json           # dry run on CPU
python -m basma.predict --model runs/mini/best --fusion runs/mini/report/fusion.json photo.jpg
```

Pipeline: `build_data` (stream sources -> normalised JPEG shards) -> `train` (DINOv2 views + attacks) -> `score`
(held-out pools x 10 conditions) -> `baselines` -> `fusion` (calibration, thresholds, report) -> `scripts/export_onnx.py`
-> `web/` (ONNX in a worker, bit-exact preprocessing port).

## Results (v2, 2026-11-05)

Scored on 92,000 held-out rows (images x 10 attack conditions: JPEG, WebP, resize, screenshot, social-media pipeline,
colour edit, blur/noise, crop, combinations) from pools never trained on: Community Forensics Eval, GenImage, WildRF
(Reddit/Facebook/Twitter), unseen commercial generators (Seedream, Hunyuan, Recraft, Reve, Grok/Aurora, Qwen, MAI,
ERNIE, Muse) and unseen real photos (Flickr2K, Open Images). Baselines are scored on exactly the same rows.

**At an equal false-positive rate on unseen real photos** ([docs/EQUAL_FPR.md](docs/EQUAL_FPR.md)):

| AI images caught | FPR 1 % | FPR 2 % | FPR 5 % |
|---|---|---|---|
| Basma | 74.1 % | 80.3 % | 86.6 % |
| Community Forensics frontier detector (Aug 2026) | 67.7 % | 73.9 % | 82.7 % |
| OwensLab commfor-384 | 41.2 % | 47.4 % | 59.0 % |
| Ateeqq SigLIP detector | 19.9 % | 27.5 % | 39.7 % |

**At the shipped threshold** (set so at most 1 % of validation reals and 1 % of calibration reals from sources never
trained on are flagged; same rule for every system, [docs/RESULTS.md](docs/RESULTS.md)): Basma catches 81.8 % of
held-out AI images and flags 2.4 % of held-out real photos; the strongest baseline catches 69.3 % and flags 1.2 %.

Where Basma is strongest: recent commercial generators it never saw (Seedream 90 % vs 56 %, Reve 76 % vs 43 %,
Grok 96 % vs 78 % at 2 % FPR) and heavy re-processing (social-media pipeline 78 % vs 57 %, resize 82 % vs 65 %).
Where it is weaker: GenImage's older ImageNet-class generators (ADM, BigGAN, GLIDE, VQDM: 50 % vs 70 %), and
LAION-style web images, 8 % of which are still flagged as AI at the shipped threshold.

v1 (same model, narrower real photos) flagged 10.4 % of held-out real photos; v2 added web images (CC3M, 2018),
stock photography and phone photos, and a calibration set, which brought that to 2.4 % and raised ranking quality
at 1 % FPR from 55.9 % to 74.1 %.
