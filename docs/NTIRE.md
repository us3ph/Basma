# NTIRE 2026 Robust AI-Generated Image Detection in the Wild: public test

2500 images (1300 AI, 1200 real; half distorted). ROC AUC, higher is better. Clean = images with is_distorted 0, Robust = is_distorted 1 (the challenge ranks by Robust), All = both. Basma never saw NTIRE data; the challenge teams trained on the NTIRE training set (same pipeline, 20 of its generators).

| System | Clean | Robust | All |
|---|---|---|---|
| **Basma v2** (fused score) | 0.8954 | 0.7045 | 0.7978 |
| Basma v2 (calibrated probability) | 0.8943 | 0.7037 | 0.7962 |
| Thermostatic (Aug 2026) | 0.9193 | 0.6914 | 0.7904 |
| MICV (1st), challenge entry | 0.9978 | 0.9738 | |
| Ant International (2nd), challenge entry | 0.9973 | 0.9731 | |
| TeleAI-TeleGuard (3rd), challenge entry | 0.9762 | 0.9215 | |
| INTSIG (4th), challenge entry | 0.9810 | 0.9090 | |
| vincentlc (5th), challenge entry | 0.9497 | 0.8633 | |

Challenge numbers: open test, Table 3 of arXiv 2604.11487 (its "ROC AUC" column is shown under Clean).

## Robust AUC per distortion (an image can carry several)

| Distortion | Images | **Basma v2** (fused score) | Basma v2 (calibrated probability) | Thermostatic (Aug 2026) |
|---|---|---|---|---|
| downscale | 926 | 0.6955 | 0.6953 | 0.6912 |
| watermark | 448 | 0.6743 | 0.6732 | 0.6783 |
| brighten | 286 | 0.7310 | 0.7297 | 0.7171 |
| lensblur | 266 | 0.7265 | 0.7264 | 0.6501 |
| jpeg_ai | 154 | 0.5846 | 0.5857 | 0.5703 |
| cheng2020 | 152 | 0.6677 | 0.6652 | 0.6426 |
| jpeg | 150 | 0.6825 | 0.6825 | 0.6838 |
| jpeg2000 | 142 | 0.7529 | 0.7513 | 0.7002 |
| jpeg_recompression_2 | 112 | 0.7149 | 0.7163 | 0.7284 |
| randomaspectcrop | 105 | 0.6853 | 0.6832 | 0.7080 |
| randomcrop | 104 | 0.7040 | 0.6998 | 0.6452 |
| jpeg_recompression_1 | 96 | 0.6656 | 0.6637 | 0.6569 |
| wmforger | 91 | 0.6261 | 0.6204 | 0.5313 |
| adv_embed_resnet | 90 | 0.6232 | 0.6244 | 0.6212 |
| adv_embed_clip | 89 | 0.6096 | 0.6101 | 0.6066 |
| shotnoise | 87 | 0.6926 | 0.6918 | 0.6513 |
| jpeg_recompression_comb_jpegai | 86 | 0.6375 | 0.6323 | 0.7190 |
| isonoise | 85 | 0.6932 | 0.6948 | 0.8024 |
| perspective | 82 | 0.6595 | 0.6559 | 0.7327 |
| impulsenoise | 80 | 0.7117 | 0.7098 | 0.8243 |
| randomtonecurve | 72 | 0.7071 | 0.7055 | 0.7467 |
| clahe | 68 | 0.7361 | 0.7331 | 0.7031 |
| colorsat | 65 | 0.6717 | 0.6723 | 0.6555 |
| rgbshift | 58 | 0.8065 | 0.8095 | 0.8080 |

## Robust AUC by number of distortions

| Distortions | Images | **Basma v2** (fused score) | Basma v2 (calibrated probability) | Thermostatic (Aug 2026) |
|---|---|---|---|---|
| 1 | 61 | 0.8167 | 0.8178 | 0.8222 |
| 2 | 324 | 0.7719 | 0.7691 | 0.7428 |
| 3 | 397 | 0.7117 | 0.7116 | 0.7104 |
| 4 | 346 | 0.6532 | 0.6541 | 0.6354 |
| 5 | 122 | 0.5946 | 0.5965 | 0.6596 |

## Basma's shipped verdicts (thresholds from fusion.json, AI at 1 % FPR on its own calibration reals)

| Subset | AI images called AI | AI called Real | AI Uncertain | Real images called Real | Real called AI | Real Uncertain |
|---|---|---|---|---|---|---|
| clean | 80.6 % | 11.2 % | 8.2 % | 67.5 % | 20.7 % | 11.8 % |
| robust | 55.5 % | 34.5 % | 10.0 % | 65.3 % | 27.2 % | 7.5 % |
| all | 68.1 % | 22.9 % | 9.1 % | 66.4 % | 23.9 % | 9.7 % |
