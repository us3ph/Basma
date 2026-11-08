# Basma results

Thresholds for every system are set the same way: the lowest score at which at most 1 % of the validation real images (train sources) and at most 1 % of the calibration real images (sources never trained on: Pascal VOC 2012, VizWiz), all conditions pooled, are flagged. Basma's threshold is on its calibrated probability (0.882); baselines use their own raw scores. TPR = share of AI images above the threshold, FPR = share of real images above it. Validation: 20000 rows, evaluation: 92000 rows (image x condition).

## Held-out pools (never trained on)

| pool | system | AI rows | real rows | TPR % | FPR % | AUROC % |
|---|---|---|---|---|---|---|
| eval:cf_eval | Basma | 14960 | 5040 | 80.8 | 3.0 | 96.1 |
|  | thermostatic | 14960 | 5040 | 65.2 | 0.9 | 93.2 |
|  | ateeqq | 14960 | 5040 | 17.2 | 2.8 | 66.1 |
|  | commfor | 14960 | 5040 | 43.0 | 0.6 | 91.5 |
| eval:genimage | Basma | 7560 | 12440 | 78.1 | 1.0 | 98.1 |
|  | thermostatic | 7560 | 12440 | 63.6 | 0.3 | 93.9 |
|  | ateeqq | 7560 | 12440 | 0.7 | 0.7 | 60.1 |
|  | commfor | 7560 | 12440 | 57.1 | 0.5 | 94.6 |
| eval:real_flickr2k | Basma | 0 | 6000 | – | 0.5 | – |
|  | thermostatic | 0 | 6000 | – | 0.5 | – |
|  | ateeqq | 0 | 6000 | – | 0.1 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:real_openimages | Basma | 0 | 6000 | – | 0.9 | – |
|  | thermostatic | 0 | 6000 | – | 0.5 | – |
|  | ateeqq | 0 | 6000 | – | 0.8 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:unseen_vendors | Basma | 20000 | 0 | 92.0 | – | – |
|  | thermostatic | 20000 | 0 | 63.0 | – | – |
|  | ateeqq | 20000 | 0 | 17.4 | – | – |
|  | commfor | 20000 | 0 | 19.7 | – | – |
| eval:wildrf | Basma | 9910 | 10090 | 90.2 | 1.7 | 98.8 |
|  | thermostatic | 9910 | 10090 | 68.8 | 1.5 | 96.0 |
|  | ateeqq | 9910 | 10090 | 30.7 | 0.1 | 89.0 |
|  | commfor | 9910 | 10090 | 46.5 | 0.9 | 92.2 |

## By condition (held-out pools, all images)

| condition | system | TPR % | FPR % | AUROC % |
|---|---|---|---|---|
| blur1 | Basma | 88.1 | 1.1 | 98.7 |
|  | thermostatic | 77.6 | 1.4 | 97.3 |
|  | ateeqq | 20.9 | 1.6 | 76.1 |
|  | commfor | 54.4 | 0.9 | 93.0 |
| clean | Basma | 89.4 | 0.9 | 99.0 |
|  | thermostatic | 77.0 | 0.7 | 96.9 |
|  | ateeqq | 20.1 | 1.0 | 80.0 |
|  | commfor | 45.5 | 0.1 | 95.1 |
| color | Basma | 84.9 | 0.7 | 98.7 |
|  | thermostatic | 65.2 | 0.7 | 95.8 |
|  | ateeqq | 16.7 | 0.6 | 80.9 |
|  | commfor | 39.6 | 0.4 | 91.9 |
| combo | Basma | 86.3 | 1.4 | 98.5 |
|  | thermostatic | 60.3 | 0.5 | 95.2 |
|  | ateeqq | 18.6 | 0.4 | 79.9 |
|  | commfor | 29.9 | 0.4 | 86.1 |
| crop50 | Basma | 88.2 | 1.0 | 98.6 |
|  | thermostatic | 63.5 | 0.3 | 96.7 |
|  | ateeqq | 9.8 | 0.1 | 79.0 |
|  | commfor | 42.3 | 0.3 | 93.0 |
| jpeg60 | Basma | 85.4 | 0.9 | 98.4 |
|  | thermostatic | 72.0 | 1.0 | 95.8 |
|  | ateeqq | 17.6 | 0.6 | 79.9 |
|  | commfor | 36.9 | 0.2 | 91.5 |
| resize50 | Basma | 87.3 | 2.0 | 98.3 |
|  | thermostatic | 56.8 | 0.1 | 95.9 |
|  | ateeqq | 21.6 | 1.5 | 78.9 |
|  | commfor | 37.4 | 1.1 | 87.3 |
| screenshot | Basma | 85.5 | 1.6 | 98.2 |
|  | thermostatic | 60.4 | 0.6 | 96.1 |
|  | ateeqq | 21.0 | 0.6 | 80.1 |
|  | commfor | 32.8 | 0.8 | 86.0 |
| social | Basma | 85.0 | 1.8 | 98.0 |
|  | thermostatic | 45.8 | 0.8 | 91.8 |
|  | ateeqq | 8.8 | 0.3 | 75.0 |
|  | commfor | 16.0 | 0.2 | 76.7 |
| webp50 | Basma | 84.7 | 1.8 | 98.1 |
|  | thermostatic | 69.6 | 1.2 | 95.2 |
|  | ateeqq | 19.4 | 0.8 | 79.8 |
|  | commfor | 33.1 | 1.4 | 84.2 |

## AI generator families (held-out pools), TPR % per system

| family | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| cf_eval | 14960 | 80.8 | 65.2 | 17.2 | 43.0 |
| wildrf | 9910 | 90.2 | 68.8 | 30.7 | 46.5 |
| genimage | 7560 | 78.1 | 63.6 | 0.7 | 57.1 |
| grok | 6670 | 95.9 | 68.1 | 13.2 | 20.6 |
| recraft | 4150 | 93.8 | 72.2 | 17.2 | 21.5 |
| seedream | 2810 | 89.4 | 44.3 | 15.1 | 7.0 |
| reve | 2280 | 71.9 | 30.9 | 15.5 | 7.2 |
| hunyuan | 1700 | 99.9 | 97.3 | 37.8 | 60.9 |
| qwen | 1440 | 94.7 | 60.3 | 22.4 | 8.6 |
| muse | 430 | 90.0 | 44.2 | 8.4 | 19.1 |
| mai | 280 | 100.0 | 82.5 | 17.5 | 16.1 |
| ernie | 240 | 97.1 | 71.2 | 25.8 | 7.1 |

## Real sources (held-out pools), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| genimage | 12440 | 1.0 | 0.3 | 0.7 | 0.5 |
| wildrf | 10090 | 1.7 | 1.5 | 0.1 | 0.9 |
| flickr2k | 6000 | 0.5 | 0.5 | 0.1 | 0.4 |
| openimages | 6000 | 0.9 | 0.5 | 0.8 | 0.4 |
| cf_eval | 5040 | 3.0 | 0.9 | 2.8 | 0.6 |

## Pool x condition (Basma TPR % / FPR %)

| pool | blur1 | clean | color | combo | crop50 | jpeg60 | resize50 | screenshot | social | webp50 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval:cf_eval | 83.6 / 2.0 | 84.2 / 2.4 | 77.7 / 1.2 | 79.6 / 3.0 | 82.8 / 1.8 | 80.0 / 2.2 | 82.1 / 4.2 | 80.3 / 2.8 | 78.3 / 4.0 | 79.7 / 6.2 |
| eval:genimage | 79.2 / 0.7 | 81.9 / 0.6 | 81.8 / 0.6 | 81.5 / 1.0 | 82.8 / 0.6 | 71.4 / 0.7 | 76.5 / 1.3 | 72.1 / 1.3 | 81.6 / 1.5 | 71.8 / 1.4 |
| eval:real_flickr2k | – / 0.2 | – / 0.2 | – / 0.3 | – / 0.7 | – / 0.3 | – / 0.3 | – / 1.0 | – / 0.5 | – / 1.0 | – / 0.7 |
| eval:real_openimages | – / 0.7 | – / 0.3 | – / 0.7 | – / 1.7 | – / 0.5 | – / 0.3 | – / 1.3 | – / 0.8 | – / 1.2 | – / 1.0 |
| eval:unseen_vendors | 93.2 / – | 94.2 / – | 88.6 / – | 92.2 / – | 92.5 / – | 92.2 / – | 93.8 / – | 92.4 / – | 89.9 / – | 91.3 / – |
| eval:wildrf | 91.5 / 1.8 | 93.2 / 1.4 | 90.8 / 1.0 | 88.3 / 1.4 | 92.1 / 1.8 | 90.4 / 1.0 | 90.2 / 2.7 | 89.6 / 2.4 | 87.9 / 2.0 | 88.4 / 1.5 |

## Calibration reals (never trained on; used to set every threshold), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| vizwiz | 8150 | 0.7 | 1.2 | 0.1 | 1.9 |
| voc2012 | 11850 | 0.4 | 0.1 | 0.1 | 0.4 |

## Validation (train sources, 8 % held out by hash), Basma

| condition | TPR % | FPR % | AUROC % |
|---|---|---|---|
| blur1 | 95.3 | 0.6 | 99.8 |
| clean | 96.3 | 0.5 | 99.9 |
| color | 93.5 | 0.4 | 99.8 |
| combo | 93.6 | 0.8 | 99.7 |
| crop50 | 96.2 | 0.6 | 99.8 |
| jpeg60 | 93.4 | 0.6 | 99.8 |
| resize50 | 94.7 | 0.8 | 99.7 |
| screenshot | 93.6 | 0.9 | 99.6 |
| social | 90.6 | 0.9 | 99.4 |
| webp50 | 93.4 | 0.9 | 99.7 |

## Verdicts on the held-out pools

| label:verdict | rows |
|---|---|
| ai:AI | 45343 |
| ai:Real | 2965 |
| ai:Uncertain | 4122 |
| real:AI | 521 |
| real:Real | 37299 |
| real:Uncertain | 1750 |

