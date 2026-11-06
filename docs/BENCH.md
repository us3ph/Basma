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

## Mistakes: AI called Real / Real called AI

| Version | Held-out, original images | Held-out, after edits / sharing | NTIRE, clean | NTIRE, distorted |
|---|---|---|---|---|
| v1 | 11.4 % / 9.7 % | 11.4 % / 10.4 % | – | – |
| v2 | 12.3 % / 2.0 % | 13.8 % / 2.4 % | 11.2 % / 20.7 % | 34.5 % / 27.2 % |

## AUROC

| Version | Held-out, original images | Held-out, after edits / sharing | NTIRE, clean | NTIRE, distorted |
|---|---|---|---|---|
| v1 | 0.943 | 0.942 | – | – |
| v2 | 0.964 | 0.960 | 0.895 | 0.705 |

## Held-out pools (original / after edits)

AI pools show *AI detected*, real-photo pools *Human detected*; mixed pools show both as AI / Human.

| Pool | v1 | v2 |
|---|---|---|
| Unseen commercial generators (Seedream, Reve, Grok, Qwen, …) | 97.0 % → 96.6 % | 95.0 % → 93.4 % |
| Community Forensics Eval | 84.9 % / 74.8 % → 84.8 % / 74.8 % | 81.4 % / 92.3 % → 78.2 % / 92.5 % |
| GenImage (older generators, ImageNet photos) | 61.6 % / 94.7 % → 61.4 % / 92.4 % | 53.2 % / 97.6 % → 51.3 % / 96.6 % |
| WildRF (Reddit / Facebook / Twitter) | 89.8 % / 83.8 % → 91.2 % / 83.5 % | 87.8 % / 92.8 % → 86.0 % / 93.4 % |
| Flickr2K photos | 90.7 % → 89.0 % | 97.8 % → 95.9 % |
| Open Images photos | 93.2 % → 92.6 % | 97.5 % → 96.7 % |

## Held-out by edit (AI detected / Human detected)

| Edit | v1 | v2 |
|---|---|---|
| original | 87.1 % / 88.5 % | 83.7 % / 95.7 % |
| JPEG 60 | 87.0 % / 88.4 % | 81.9 % / 95.5 % |
| WebP 50 | 87.2 % / 84.8 % | 81.6 % / 92.9 % |
| resized to 50 % | 88.0 % / 87.6 % | 83.0 % / 95.3 % |
| screenshot | 87.4 % / 87.4 % | 80.5 % / 95.3 % |
| social-media pipeline | 87.7 % / 87.0 % | 80.5 % / 94.5 % |
| colour edit | 85.0 % / 90.9 % | 79.4 % / 97.1 % |
| blur / noise | 87.1 % / 85.6 % | 83.3 % / 94.9 % |
| crop 50 % | 86.9 % / 87.7 % | 82.7 % / 95.7 % |
| several edits | 87.9 % / 87.3 % | 81.4 % / 95.2 % |

## Versions

* **v1**: first run: real photos from only 4 sources (COCO, LandscapesHQ, FFHQ, VISION); never shipped
* **v2**: added 29k varied real photos (CC3M, Pexels, VISION) and a threshold that also holds on never-trained real photos (VOC, VizWiz); live on the website
