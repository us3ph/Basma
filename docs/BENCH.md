# Basma Bench

Our own fixed test, scored for every version. No version ever trains on these images, picks its checkpoint with them or sets its thresholds with them, so the numbers show real progress rather than familiarity with a test. Each version answers with its own shipped settings (the AI / Real / Uncertain a user would see).

* **Held-out pools**: 9,200 images from generators and photo sources never trained on (Community Forensics Eval, GenImage, WildRF from Reddit/Facebook/Twitter, unseen commercial generators, Flickr2K and Open Images photos), each scored as-is and after 9 edits (JPEG, WebP, resize, screenshot, social-media pipeline, colour, blur/noise, crop, several).
* **NTIRE 2026 public test**: 2,500 images from another lab, half with stacked distortions.

**AI detected** = AI images answered *AI*. **Human detected** = real photos answered *Real*. The rest is either *Uncertain* or wrong (tables below). AUROC is the threshold-free ranking quality (1.0 = perfect, 0.5 = guessing).

## AI detected / Human detected

| Version | Date | Held-out, original images | Held-out, after edits / sharing | NTIRE, clean | NTIRE, distorted |
|---|---|---|---|---|---|
| **v1** | 2026-11-04 | 87.1 % / 88.5 % | 87.1 % / 87.4 % | – | – |
| **v2** | 2026-11-05 | 83.7 % / 95.7 % | 81.6 % / 95.2 % | 80.6 % / 67.5 % | 55.5 % / 65.3 % |
| **v3** | 2026-11-07 | 79.0 % / 97.5 % | 75.8 % / 97.0 % | 75.1 % / 79.8 % | 48.5 % / 72.3 % |
| **v4** | 2026-11-08 | 89.4 % / 95.9 % | 86.2 % / 94.1 % | 82.9 % / 75.7 % | 56.6 % / 62.3 % |

## Mistakes: AI called Real / Real called AI

| Version | Held-out, original images | Held-out, after edits / sharing | NTIRE, clean | NTIRE, distorted |
|---|---|---|---|---|
| v1 | 11.4 % / 9.7 % | 11.4 % / 10.4 % | – | – |
| v2 | 12.3 % / 2.0 % | 13.8 % / 2.4 % | 11.2 % / 20.7 % | 34.5 % / 27.2 % |
| v3 | 14.3 % / 0.7 % | 15.7 % / 0.8 % | 12.3 % / 8.0 % | 34.6 % / 16.7 % |
| v4 | 4.4 % / 0.9 % | 5.8 % / 1.4 % | 5.4 % / 8.7 % | 25.2 % / 22.7 % |

## AUROC

| Version | Held-out, original images | Held-out, after edits / sharing | NTIRE, clean | NTIRE, distorted |
|---|---|---|---|---|
| v1 | 0.943 | 0.942 | – | – |
| v2 | 0.964 | 0.960 | 0.895 | 0.705 |
| v3 | 0.960 | 0.959 | 0.926 | 0.753 |
| v4 | 0.990 | 0.985 | 0.953 | 0.749 |

## Held-out pools (original / after edits)

AI pools show *AI detected*, real-photo pools *Human detected*; mixed pools show both as AI / Human.

| Pool | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Unseen commercial generators (Seedream, Reve, Grok, Qwen, …) | 97.0 % → 96.6 % | 95.0 % → 93.4 % | 94.0 % → 91.7 % | 94.2 % → 91.8 % |
| Community Forensics Eval | 84.9 % / 74.8 % → 84.8 % / 74.8 % | 81.4 % / 92.3 % → 78.2 % / 92.5 % | 75.4 % / 94.4 % → 70.4 % / 93.6 % | 84.2 % / 91.7 % → 80.5 % / 89.0 % |
| GenImage (older generators, ImageNet photos) | 61.6 % / 94.7 % → 61.4 % / 92.4 % | 53.2 % / 97.6 % → 51.3 % / 96.6 % | 45.6 % / 98.5 % → 42.1 % / 98.0 % | 81.9 % / 97.5 % → 77.6 % / 95.2 % |
| WildRF (Reddit / Facebook / Twitter) | 89.8 % / 83.8 % → 91.2 % / 83.5 % | 87.8 % / 92.8 % → 86.0 % / 93.4 % | 79.3 % / 96.1 % → 77.5 % / 96.0 % | 93.2 % / 94.0 % → 89.9 % / 92.8 % |
| Flickr2K photos | 90.7 % → 89.0 % | 97.8 % → 95.9 % | 98.8 % → 98.3 % | 98.2 % → 96.3 % |
| Open Images photos | 93.2 % → 92.6 % | 97.5 % → 96.7 % | 98.8 % → 98.3 % | 97.0 % → 95.8 % |

## Held-out by edit (AI detected / Human detected)

| Edit | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| original | 87.1 % / 88.5 % | 83.7 % / 95.7 % | 79.0 % / 97.5 % | 89.4 % / 95.9 % |
| JPEG 60 | 87.0 % / 88.4 % | 81.9 % / 95.5 % | 77.0 % / 96.9 % | 85.3 % / 95.1 % |
| WebP 50 | 87.2 % / 84.8 % | 81.6 % / 92.9 % | 75.4 % / 95.2 % | 84.7 % / 92.1 % |
| resized to 50 % | 88.0 % / 87.6 % | 83.0 % / 95.3 % | 77.2 % / 97.5 % | 87.3 % / 93.5 % |
| screenshot | 87.4 % / 87.4 % | 80.5 % / 95.3 % | 74.8 % / 97.1 % | 85.5 % / 93.6 % |
| social-media pipeline | 87.7 % / 87.0 % | 80.5 % / 94.5 % | 74.5 % / 96.8 % | 85.0 % / 92.3 % |
| colour edit | 85.0 % / 90.9 % | 79.4 % / 97.1 % | 73.3 % / 98.5 % | 84.9 % / 96.3 % |
| blur / noise | 87.1 % / 85.6 % | 83.3 % / 94.9 % | 75.9 % / 97.6 % | 88.1 % / 95.3 % |
| crop 50 % | 86.9 % / 87.7 % | 82.7 % / 95.7 % | 77.0 % / 97.1 % | 88.2 % / 94.5 % |
| several edits | 87.9 % / 87.3 % | 81.4 % / 95.2 % | 77.0 % / 96.5 % | 86.3 % / 93.9 % |

## Versions

* **v1**: first run: real photos from only 4 sources (COCO, LandscapesHQ, FFHQ, VISION); never shipped
* **v2**: added 29k varied real photos (CC3M, Pexels, VISION) and a threshold that also holds on never-trained real photos (VOC, VizWiz); live on the website 2026-11-05 to 2026-11-07
* **v3**: added 66k real photos of new kinds (LSDIR, SUN397, food, fashion, cars, pets, more Pexels / CC3M / WikiArt) and 56k AI images (Community Forensics, 2026 generators, GPT-image-2, StyleGAN2 faces), twice the training steps. Real photos called AI fell to 0.7 % (NTIRE clean 20.7 % -> 8.0 %); at the same false-alarm rate it catches more AI than v2 (held-out, 1 % of reals flagged: 78.8 -> 81.0 % original, 73.5 -> 77.0 % edited), but its shipped threshold is stricter, so fewer AI images are answered AI. Live on the website since 2026-11-07
* **v4**: added about 80k AI images aimed at what v3 missed (Synthbuster+, So-Fake-Set, ADM samples, DALL-E 2 faces, Karlo, OpenFake Civitai and Reddit) plus Reddit real photos; NTIRE and its photo sources are refused by the data builder, and a leak guard removed 242 training images that looked like bench or NTIRE images. AI called Real fell from 14.3 % to 4.4 %; at the same false-alarm rate (1 % of reals flagged) it catches 90.0 % vs 81.0 % (original) and 83.9 % vs 77.0 % (edited), NTIRE clean 64.2 vs 52.6 %. Caveat: ADM, GLIDE, DALL-E 2 and Firefly are now trained generators (other images of them), so those bench rows no longer test unseen generators; generators still unseen improved too (Hourglass 2 -> 28 %, GALIP 49 -> 83 %, DF-GAN 68 -> 98 %). Not better on NTIRE distorted (AUROC 0.749) and more of its real photos are called AI (22.7 % vs 16.7 %). Not live yet
