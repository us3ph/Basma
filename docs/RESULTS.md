# Basma results

Thresholds for every system are set the same way: the lowest score at which at most 1 % of the validation real images (train sources) and at most 1 % of the calibration real images (sources never trained on: Pascal VOC 2012, VizWiz), all conditions pooled, are flagged. Basma's threshold is on its calibrated probability (0.864); baselines use their own raw scores. TPR = share of AI images above the threshold, FPR = share of real images above it. Validation: 20000 rows, evaluation: 92000 rows (image x condition).

## Held-out pools (never trained on)

| pool | system | AI rows | real rows | TPR % | FPR % | AUROC % |
|---|---|---|---|---|---|---|
| eval:cf_eval | Basma | 14960 | 5040 | 70.9 | 3.0 | 92.8 |
|  | thermostatic | 14960 | 5040 | 64.8 | 0.8 | 93.2 |
|  | ateeqq | 14960 | 5040 | 16.1 | 2.3 | 66.1 |
|  | commfor | 14960 | 5040 | 43.0 | 0.6 | 91.5 |
| eval:genimage | Basma | 7560 | 12440 | 42.5 | 0.4 | 86.4 |
|  | thermostatic | 7560 | 12440 | 63.1 | 0.3 | 93.9 |
|  | ateeqq | 7560 | 12440 | 0.5 | 0.6 | 60.1 |
|  | commfor | 7560 | 12440 | 57.1 | 0.5 | 94.6 |
| eval:real_flickr2k | Basma | 0 | 6000 | – | 0.5 | – |
|  | thermostatic | 0 | 6000 | – | 0.5 | – |
|  | ateeqq | 0 | 6000 | – | 0.1 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:real_openimages | Basma | 0 | 6000 | – | 0.3 | – |
|  | thermostatic | 0 | 6000 | – | 0.5 | – |
|  | ateeqq | 0 | 6000 | – | 0.7 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:unseen_vendors | Basma | 20000 | 0 | 91.9 | – | – |
|  | thermostatic | 20000 | 0 | 62.5 | – | – |
|  | ateeqq | 20000 | 0 | 15.5 | – | – |
|  | commfor | 20000 | 0 | 19.7 | – | – |
| eval:wildrf | Basma | 9910 | 10090 | 77.6 | 0.8 | 97.5 |
|  | thermostatic | 9910 | 10090 | 68.3 | 1.4 | 96.0 |
|  | ateeqq | 9910 | 10090 | 28.4 | 0.1 | 89.0 |
|  | commfor | 9910 | 10090 | 46.5 | 0.9 | 92.2 |

## By condition (held-out pools, all images)

| condition | system | TPR % | FPR % | AUROC % |
|---|---|---|---|---|
| blur1 | Basma | 75.9 | 0.8 | 95.3 |
|  | thermostatic | 77.2 | 1.3 | 97.3 |
|  | ateeqq | 18.9 | 1.3 | 76.1 |
|  | commfor | 54.4 | 0.9 | 93.0 |
| clean | Basma | 79.0 | 0.7 | 95.8 |
|  | thermostatic | 76.6 | 0.7 | 96.9 |
|  | ateeqq | 18.6 | 0.9 | 80.0 |
|  | commfor | 45.5 | 0.1 | 95.1 |
| color | Basma | 73.3 | 0.5 | 95.6 |
|  | thermostatic | 64.7 | 0.6 | 95.8 |
|  | ateeqq | 15.2 | 0.5 | 80.9 |
|  | commfor | 39.6 | 0.4 | 91.9 |
| combo | Basma | 77.0 | 0.9 | 96.1 |
|  | thermostatic | 59.8 | 0.4 | 95.2 |
|  | ateeqq | 16.9 | 0.4 | 79.9 |
|  | commfor | 29.9 | 0.4 | 86.1 |
| crop50 | Basma | 77.0 | 0.7 | 96.0 |
|  | thermostatic | 63.0 | 0.3 | 96.7 |
|  | ateeqq | 9.1 | 0.1 | 79.0 |
|  | commfor | 42.3 | 0.3 | 93.0 |
| jpeg60 | Basma | 77.0 | 0.9 | 95.2 |
|  | thermostatic | 71.7 | 0.9 | 95.8 |
|  | ateeqq | 16.1 | 0.5 | 79.9 |
|  | commfor | 36.9 | 0.2 | 91.5 |
| resize50 | Basma | 77.2 | 0.8 | 95.8 |
|  | thermostatic | 56.6 | 0.1 | 95.9 |
|  | ateeqq | 19.8 | 1.2 | 78.9 |
|  | commfor | 37.4 | 1.1 | 87.3 |
| screenshot | Basma | 74.8 | 0.8 | 95.6 |
|  | thermostatic | 59.7 | 0.6 | 96.1 |
|  | ateeqq | 19.1 | 0.4 | 80.1 |
|  | commfor | 32.8 | 0.8 | 86.0 |
| social | Basma | 74.5 | 0.9 | 95.7 |
|  | thermostatic | 45.2 | 0.8 | 91.8 |
|  | ateeqq | 7.9 | 0.3 | 75.0 |
|  | commfor | 16.0 | 0.2 | 76.7 |
| webp50 | Basma | 75.4 | 1.4 | 94.9 |
|  | thermostatic | 69.0 | 1.1 | 95.2 |
|  | ateeqq | 17.8 | 0.7 | 79.8 |
|  | commfor | 33.1 | 1.4 | 84.2 |

## AI generator families (held-out pools), TPR % per system

| family | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| cf_eval | 14960 | 70.9 | 64.8 | 16.1 | 43.0 |
| wildrf | 9910 | 77.6 | 68.3 | 28.4 | 46.5 |
| genimage | 7560 | 42.5 | 63.1 | 0.5 | 57.1 |
| grok | 6670 | 95.8 | 67.7 | 11.4 | 20.6 |
| recraft | 4150 | 94.1 | 71.5 | 15.7 | 21.5 |
| seedream | 2810 | 91.7 | 43.7 | 13.4 | 7.0 |
| reve | 2280 | 68.3 | 30.4 | 14.2 | 7.2 |
| hunyuan | 1700 | 100.0 | 97.2 | 34.7 | 60.9 |
| qwen | 1440 | 95.1 | 59.4 | 19.2 | 8.6 |
| muse | 430 | 87.7 | 43.3 | 7.0 | 19.1 |
| mai | 280 | 100.0 | 82.5 | 13.2 | 16.1 |
| ernie | 240 | 96.7 | 71.2 | 23.3 | 7.1 |

## Real sources (held-out pools), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| genimage | 12440 | 0.4 | 0.3 | 0.6 | 0.5 |
| wildrf | 10090 | 0.8 | 1.4 | 0.1 | 0.9 |
| flickr2k | 6000 | 0.5 | 0.5 | 0.1 | 0.4 |
| openimages | 6000 | 0.3 | 0.5 | 0.7 | 0.4 |
| cf_eval | 5040 | 3.0 | 0.8 | 2.3 | 0.6 |

## Pool x condition (Basma TPR % / FPR %)

| pool | blur1 | clean | color | combo | crop50 | jpeg60 | resize50 | screenshot | social | webp50 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval:cf_eval | 71.3 / 2.6 | 75.4 / 3.0 | 67.1 / 2.4 | 71.4 / 3.2 | 72.8 / 2.4 | 72.5 / 3.0 | 71.9 / 3.0 | 68.7 / 2.2 | 68.1 / 2.6 | 69.9 / 5.4 |
| eval:genimage | 41.4 / 0.3 | 45.6 / 0.4 | 42.1 / 0.2 | 44.0 / 0.4 | 46.0 / 0.5 | 42.9 / 0.4 | 42.2 / 0.3 | 40.0 / 0.5 | 40.1 / 0.6 | 40.7 / 0.8 |
| eval:real_flickr2k | – / 0.2 | – / 0.2 | – / 0.2 | – / 0.7 | – / 0.3 | – / 0.7 | – / 0.2 | – / 0.7 | – / 0.8 | – / 0.8 |
| eval:real_openimages | – / 0.3 | – / 0.5 | – / 0.2 | – / 0.3 | – / 0.2 | – / 0.7 | – / 0.2 | – / 0.2 | – / 0.2 | – / 0.7 |
| eval:unseen_vendors | 92.6 / – | 94.0 / – | 89.4 / – | 92.6 / – | 92.2 / – | 92.6 / – | 93.4 / – | 90.8 / – | 89.9 / – | 91.7 / – |
| eval:wildrf | 75.5 / 1.0 | 79.3 / 0.5 | 74.2 / 0.3 | 78.9 / 0.8 | 76.5 / 0.7 | 78.4 / 0.6 | 79.3 / 0.9 | 78.0 / 0.9 | 79.2 / 1.1 | 77.1 / 0.8 |

## Calibration reals (never trained on; used to set every threshold), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| vizwiz | 8150 | 0.3 | 1.2 | 0.0 | 1.9 |
| voc2012 | 11850 | 0.3 | 0.1 | 0.1 | 0.4 |

## Validation (train sources, 8 % held out by hash), Basma

| condition | TPR % | FPR % | AUROC % |
|---|---|---|---|
| blur1 | 95.9 | 0.7 | 99.9 |
| clean | 97.1 | 0.8 | 99.9 |
| color | 94.2 | 0.6 | 99.9 |
| combo | 94.5 | 0.8 | 99.7 |
| crop50 | 96.0 | 0.9 | 99.9 |
| jpeg60 | 96.0 | 0.7 | 99.8 |
| resize50 | 95.6 | 0.7 | 99.8 |
| screenshot | 93.2 | 1.0 | 99.7 |
| social | 90.3 | 1.0 | 99.5 |
| webp50 | 94.5 | 1.3 | 99.6 |

## Verdicts on the held-out pools

| label:verdict | rows |
|---|---|
| ai:AI | 39901 |
| ai:Real | 8151 |
| ai:Uncertain | 4378 |
| real:AI | 328 |
| real:Real | 38408 |
| real:Uncertain | 834 |

