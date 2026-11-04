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
