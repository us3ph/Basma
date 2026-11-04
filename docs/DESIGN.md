# Basma design

Basma (Arabic for *fingerprint*) answers one question: **is this image AI-generated, even after it was posted,
resized, colour-edited or screenshotted?** It follows the method that made Skry work on humanized text:
train on what the attacker does, evaluate on what was never seen, report at a fixed false-positive rate, compare
against public detectors at the same threshold, and ship the whole thing to the browser.

## Why detectors break on screenshots, and what survives

Most published detectors read pixel-level generator fingerprints: upsampling patterns, frequency artifacts, noise
residue. A screenshot, a resize or a JPEG re-save wipes them. What survives is the image's *skeleton*: composition,
lighting logic, skin and texture statistics at a coarser scale, how text and hands are rendered, each generator's
house style. The ITW-SM study (arXiv 2507.10236) found DINOv2 features read that layer better than CLIP and that
compression/noise augmentation is what closes the gap in the wild. Basma is built on exactly those findings.

## Data

Every training and evaluation image is re-encoded the same way (longest side 512, JPEG q90 4:4:4) by
`basma/build_data.py`. Real photos are usually JPEG and AI images usually PNG; if the detector saw the originals it
would learn the file format instead of the content. Resolution is removed as a cue for the same reason.

Sources (all streamed from Hugging Face on a Kaggle **CPU** kernel, so no GPU quota is spent on data):

| plan | AI | real |
|---|---|---|
| `cf` | Community Forensics Small: every AI shard, at most 20 images per model (about 4,800 models: GANs, latent and pixel diffusion) | its real photos (FFHQ, VISION phone photos, COCO, Landscapes HQ) |
| `recent` | Rapidata pairs for Flux 2 pro, GPT-4o, Imagen 4 / 4 ultra, HiDream, Ideogram v2, Runway Frames, Lumina, nano-banana, DALL-E 3, Midjourney 5.2, SD3, the 2026 benchmark set (FLUX.2, GPT Image 2.5, nano-banana-2, Imagen 4, Ideogram 4, Qwen, ...), GPT-image-2, GPT-ImgEval, VCT2 (SD 2.1 / XL / 3 / 3.5, DALL-E 3, Midjourney 6), open-image-preferences (Flux dev, SD3.5) | COCO, Flickr30k, Pexels, WikiArt paintings, fashion product photos |
| `eval` | CF-Eval (21 generators the CF training set never had), GenImage test (8 generators), WildRF, the unseen-vendor Rapidata sets | RAISE / LAION / ImageNet reals of CF-Eval, ImageNet reals of GenImage, WildRF reals, Flickr2K, Open Images |

**Held-out vendor families** (`basma/generators.py`): seedream, hunyuan, recraft, reve, grok/aurora, qwen, glm, ernie,
mai, muse. An AI image of any of these goes to the `eval:unseen_vendors` pool wherever it came from, so the model
never trains on them. 8 % of the train sources are held out by hash as `val` for calibration and thresholds.

## Attacks (`basma/attacks.py`)

Each training sample draws one bucket: clean, JPEG (q 30-95), WebP, resize (0.3-1.6x, five filters), screenshot
(display scaling, UI bars, gamma, JPEG), social (platform-like downscale + JPEG, sometimes twice), colour edits
(brightness / contrast / saturation / gamma / hue, rare grayscale), blur / noise / sharpen, crop, and combos of two
or three. Evaluation uses ten fixed conditions with a per-image seed so every system sees the identical pixels.

## Model (`basma/model.py`, `basma/views.py`)

* Backbone: `facebook/dinov2-with-registers-base`, first 4 blocks frozen, layer-wise learning-rate decay.
* Views: a global view (short side to 224, centre crop) and the 3 most texture-rich 224 crops of the 512 base image
  (variance of the Laplacian on a 3 x 3 grid). Training uses a random-resized global crop and a texture-biased crop.
* Heads: an MLP on CLS + mean patch token, and a per-patch logit pooled with a top-quantile mean (rho 0.25).
  The per-patch map says *where* the signal is; its auxiliary loss keeps it informative on its own.
* Sampling: labels 50/50, families within a label weighted by count^0.5 so 4,800 small CF models do not drown
  the commercial generators, and vice versa.

## Fusion and verdict (`basma/fusion.py`)

A logistic regression over [global logit, crop mean / max / min, number of crops, log area] fitted on `val`,
isotonic calibration, then two thresholds: **AI** above the probability where at most 1 % of validation reals
fall, **Real** below the one where at most 2 % of validation AI images fall, **Uncertain** in between.

## Evaluation protocol

Baselines (Thermostatic's Community Forensics frontier detector from August 2026, the official Community
Forensics CVPR 2025 model, Ateeqq's SigLIP detector) are scored on the same images and conditions. Every system's
threshold is set the same way on the same validation reals, and results are reported per held-out pool, per
condition, per generator family and per real source: TPR at that threshold, FPR, AUROC.

## Browser (`web/`)

`resample.js` and `views.js` reproduce `basma/resample.py` and `basma/views.py` bit for bit (checked by
`web/test/parity_views.mjs`): integer block reduce for huge images, exact area-average resize, the same crop
selection, float32 normalisation in the same order. The detector is the ONNX export with weight-only 8-bit
MatMulNBits weights (fp32 fallback), run in a Web Worker; `fusion.js` is the port of the fusion layer. No image
leaves the browser.
