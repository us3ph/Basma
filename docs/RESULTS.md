# Basma results

Thresholds for every system are set the same way: the lowest score at which at most 1 % of the validation real images (train sources) and at most 1 % of the calibration real images (sources never trained on: Pascal VOC 2012, VizWiz), all conditions pooled, are flagged. Basma's threshold is on its calibrated probability (0.866); baselines use their own raw scores. TPR = share of AI images above the threshold, FPR = share of real images above it. Validation: 20000 rows, evaluation: 92000 rows (image x condition).

## Held-out pools (never trained on)

| pool | system | AI rows | real rows | TPR % | FPR % | AUROC % |
|---|---|---|---|---|---|---|
| eval:cf_eval | Basma | 14960 | 5040 | 81.3 | 3.3 | 96.1 |
|  | thermostatic | 14960 | 5040 | 69.3 | 1.8 | 93.2 |
|  | ateeqq | 14960 | 5040 | 20.5 | 4.2 | 66.1 |
|  | commfor | 14960 | 5040 | 43.0 | 0.6 | 91.5 |
| eval:genimage | Basma | 7560 | 12440 | 72.0 | 1.0 | 96.1 |
|  | thermostatic | 7560 | 12440 | 67.0 | 0.5 | 93.9 |
|  | ateeqq | 7560 | 12440 | 1.1 | 1.1 | 60.1 |
|  | commfor | 7560 | 12440 | 57.1 | 0.5 | 94.6 |
| eval:real_flickr2k | Basma | 0 | 6000 | – | 0.7 | – |
|  | thermostatic | 0 | 6000 | – | 0.8 | – |
|  | ateeqq | 0 | 6000 | – | 0.3 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:real_openimages | Basma | 0 | 6000 | – | 1.2 | – |
|  | thermostatic | 0 | 6000 | – | 0.7 | – |
|  | ateeqq | 0 | 6000 | – | 1.1 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:unseen_vendors | Basma | 20000 | 0 | 93.3 | – | – |
|  | thermostatic | 20000 | 0 | 68.1 | – | – |
|  | ateeqq | 20000 | 0 | 22.2 | – | – |
|  | commfor | 20000 | 0 | 19.7 | – | – |
| eval:wildrf | Basma | 9910 | 10090 | 89.8 | 1.9 | 98.6 |
|  | thermostatic | 9910 | 10090 | 73.4 | 2.5 | 96.0 |
|  | ateeqq | 9910 | 10090 | 36.6 | 0.3 | 89.0 |
|  | commfor | 9910 | 10090 | 46.5 | 0.9 | 92.2 |

## By condition (held-out pools, all images)

| condition | system | TPR % | FPR % | AUROC % |
|---|---|---|---|---|
| blur1 | Basma | 88.8 | 1.2 | 98.4 |
|  | thermostatic | 81.5 | 2.3 | 97.3 |
|  | ateeqq | 25.5 | 2.5 | 76.1 |
|  | commfor | 54.4 | 0.9 | 93.0 |
| clean | Basma | 89.6 | 1.2 | 98.7 |
|  | thermostatic | 80.2 | 1.2 | 96.9 |
|  | ateeqq | 23.7 | 1.5 | 80.0 |
|  | commfor | 45.5 | 0.1 | 95.1 |
| color | Basma | 85.4 | 0.8 | 98.5 |
|  | thermostatic | 70.3 | 1.0 | 95.8 |
|  | ateeqq | 20.5 | 0.9 | 80.9 |
|  | commfor | 39.6 | 0.4 | 91.9 |
| combo | Basma | 85.1 | 1.8 | 97.9 |
|  | thermostatic | 65.3 | 1.0 | 95.2 |
|  | ateeqq | 23.0 | 1.0 | 79.9 |
|  | commfor | 29.9 | 0.4 | 86.1 |
| crop50 | Basma | 88.2 | 1.2 | 98.6 |
|  | thermostatic | 67.9 | 0.4 | 96.7 |
|  | ateeqq | 12.4 | 0.1 | 79.0 |
|  | commfor | 42.3 | 0.3 | 93.0 |
| jpeg60 | Basma | 84.5 | 1.1 | 97.9 |
|  | thermostatic | 76.1 | 1.6 | 95.8 |
|  | ateeqq | 21.8 | 0.8 | 79.9 |
|  | commfor | 36.9 | 0.2 | 91.5 |
| resize50 | Basma | 86.9 | 2.1 | 97.9 |
|  | thermostatic | 60.7 | 0.3 | 95.9 |
|  | ateeqq | 25.9 | 2.1 | 78.9 |
|  | commfor | 37.4 | 1.1 | 87.3 |
| screenshot | Basma | 85.3 | 1.9 | 97.6 |
|  | thermostatic | 65.3 | 0.8 | 96.1 |
|  | ateeqq | 25.4 | 1.1 | 80.1 |
|  | commfor | 32.8 | 0.8 | 86.0 |
| social | Basma | 84.1 | 2.3 | 97.1 |
|  | thermostatic | 51.1 | 1.5 | 91.8 |
|  | ateeqq | 11.8 | 0.5 | 75.0 |
|  | commfor | 16.0 | 0.2 | 76.7 |
| webp50 | Basma | 83.2 | 1.6 | 97.2 |
|  | thermostatic | 74.2 | 2.1 | 95.2 |
|  | ateeqq | 24.0 | 1.1 | 79.8 |
|  | commfor | 33.1 | 1.4 | 84.2 |

## AI generator families (held-out pools), TPR % per system

| family | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| cf_eval | 14960 | 81.3 | 69.3 | 20.5 | 43.0 |
| wildrf | 9910 | 89.8 | 73.4 | 36.6 | 46.5 |
| genimage | 7560 | 72.0 | 67.0 | 1.1 | 57.1 |
| grok | 6670 | 96.6 | 73.0 | 18.3 | 20.6 |
| recraft | 4150 | 94.6 | 78.3 | 20.5 | 21.5 |
| seedream | 2810 | 91.9 | 49.5 | 19.8 | 7.0 |
| reve | 2280 | 75.4 | 35.8 | 18.4 | 7.2 |
| hunyuan | 1700 | 100.0 | 98.2 | 44.9 | 60.9 |
| qwen | 1440 | 95.4 | 67.6 | 29.0 | 8.6 |
| muse | 430 | 93.0 | 49.1 | 14.0 | 19.1 |
| mai | 280 | 100.0 | 87.5 | 26.4 | 16.1 |
| ernie | 240 | 96.2 | 77.5 | 33.8 | 7.1 |

## Real sources (held-out pools), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| genimage | 12440 | 1.0 | 0.5 | 1.1 | 0.5 |
| wildrf | 10090 | 1.9 | 2.5 | 0.3 | 0.9 |
| flickr2k | 6000 | 0.7 | 0.8 | 0.3 | 0.4 |
| openimages | 6000 | 1.2 | 0.7 | 1.1 | 0.4 |
| cf_eval | 5040 | 3.3 | 1.8 | 4.2 | 0.6 |

## Pool x condition (Basma TPR % / FPR %)

| pool | blur1 | clean | color | combo | crop50 | jpeg60 | resize50 | screenshot | social | webp50 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval:cf_eval | 85.8 / 1.8 | 85.2 / 2.6 | 80.0 / 2.0 | 78.3 / 4.4 | 83.6 / 3.0 | 79.5 / 2.2 | 83.0 / 4.4 | 80.9 / 3.4 | 77.9 / 5.0 | 78.5 / 4.6 |
| eval:genimage | 73.9 / 0.6 | 78.2 / 0.6 | 77.9 / 0.6 | 71.6 / 1.3 | 80.4 / 0.6 | 66.7 / 1.0 | 69.7 / 1.5 | 65.5 / 1.4 | 70.6 / 1.8 | 65.2 / 1.0 |
| eval:real_flickr2k | – / 0.5 | – / 0.2 | – / 0.3 | – / 0.5 | – / 0.2 | – / 0.7 | – / 1.0 | – / 1.2 | – / 1.2 | – / 1.0 |
| eval:real_openimages | – / 1.0 | – / 1.0 | – / 1.0 | – / 1.3 | – / 1.0 | – / 1.0 | – / 1.8 | – / 1.2 | – / 2.0 | – / 1.2 |
| eval:unseen_vendors | 95.3 / – | 95.5 / – | 90.2 / – | 93.4 / – | 93.5 / – | 92.4 / – | 95.2 / – | 94.2 / – | 91.8 / – | 91.2 / – |
| eval:wildrf | 91.4 / 2.0 | 93.1 / 2.1 | 89.6 / 0.8 | 88.8 / 2.1 | 90.5 / 1.9 | 89.8 / 1.2 | 89.1 / 2.6 | 89.4 / 2.6 | 88.0 / 2.4 | 88.0 / 1.6 |

## Calibration reals (never trained on; used to set every threshold), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| vizwiz | 8150 | 0.3 | 2.1 | 0.2 | 1.9 |
| voc2012 | 11850 | 0.5 | 0.3 | 0.2 | 0.4 |

## Validation (train sources, 8 % held out by hash), Basma

| condition | TPR % | FPR % | AUROC % |
|---|---|---|---|
| blur1 | 90.6 | 0.7 | 99.1 |
| clean | 91.3 | 0.5 | 99.3 |
| color | 87.2 | 0.3 | 99.1 |
| combo | 83.2 | 0.5 | 98.2 |
| crop50 | 90.0 | 0.4 | 99.1 |
| jpeg60 | 83.8 | 0.5 | 98.6 |
| resize50 | 87.3 | 1.0 | 98.8 |
| screenshot | 82.9 | 0.7 | 98.3 |
| social | 79.0 | 1.5 | 97.1 |
| webp50 | 81.6 | 1.1 | 98.2 |

## Verdicts on the held-out pools

| label:verdict | rows |
|---|---|
| ai:AI | 45149 |
| ai:Real | 1548 |
| ai:Uncertain | 5733 |
| real:AI | 602 |
| real:Real | 31967 |
| real:Uncertain | 7001 |

