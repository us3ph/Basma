# Basma results

Thresholds for every system are set the same way: the lowest score at which at most 1 % of the validation real images (train sources) and at most 1 % of the calibration real images (sources never trained on: Pascal VOC 2012, VizWiz), all conditions pooled, are flagged. Basma's threshold is on its calibrated probability (0.774); baselines use their own raw scores. TPR = share of AI images above the threshold, FPR = share of real images above it. Validation: 20000 rows, evaluation: 92000 rows (image x condition).

## Held-out pools (never trained on)

| pool | system | AI rows | real rows | TPR % | FPR % | AUROC % |
|---|---|---|---|---|---|---|
| eval:cf_eval | Basma | 14960 | 5040 | 78.5 | 4.6 | 93.2 |
|  | thermostatic | 14960 | 5040 | 69.3 | 1.8 | 93.2 |
|  | ateeqq | 14960 | 5040 | 18.7 | 3.4 | 66.1 |
|  | commfor | 14960 | 5040 | 43.0 | 0.6 | 91.5 |
| eval:genimage | Basma | 7560 | 12440 | 51.5 | 1.6 | 90.1 |
|  | thermostatic | 7560 | 12440 | 67.0 | 0.5 | 93.9 |
|  | ateeqq | 7560 | 12440 | 0.8 | 0.9 | 60.1 |
|  | commfor | 7560 | 12440 | 57.1 | 0.5 | 94.6 |
| eval:real_flickr2k | Basma | 0 | 6000 | – | 1.8 | – |
|  | thermostatic | 0 | 6000 | – | 0.8 | – |
|  | ateeqq | 0 | 6000 | – | 0.2 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:real_openimages | Basma | 0 | 6000 | – | 1.7 | – |
|  | thermostatic | 0 | 6000 | – | 0.7 | – |
|  | ateeqq | 0 | 6000 | – | 0.9 | – |
|  | commfor | 0 | 6000 | – | 0.4 | – |
| eval:unseen_vendors | Basma | 20000 | 0 | 93.5 | – | – |
|  | thermostatic | 20000 | 0 | 68.1 | – | – |
|  | ateeqq | 20000 | 0 | 19.6 | – | – |
|  | commfor | 20000 | 0 | 19.7 | – | – |
| eval:wildrf | Basma | 9910 | 10090 | 86.2 | 3.0 | 97.7 |
|  | thermostatic | 9910 | 10090 | 73.4 | 2.5 | 96.0 |
|  | ateeqq | 9910 | 10090 | 33.4 | 0.2 | 89.0 |
|  | commfor | 9910 | 10090 | 46.5 | 0.9 | 92.2 |

## By condition (held-out pools, all images)

| condition | system | TPR % | FPR % | AUROC % |
|---|---|---|---|---|
| blur1 | Basma | 83.3 | 2.5 | 95.7 |
|  | thermostatic | 81.5 | 2.3 | 97.3 |
|  | ateeqq | 23.2 | 2.1 | 76.1 |
|  | commfor | 54.4 | 0.9 | 93.0 |
| clean | Basma | 83.7 | 2.0 | 96.4 |
|  | thermostatic | 80.2 | 1.2 | 96.9 |
|  | ateeqq | 21.8 | 1.2 | 80.0 |
|  | commfor | 45.5 | 0.1 | 95.1 |
| color | Basma | 79.4 | 1.4 | 96.1 |
|  | thermostatic | 70.3 | 1.0 | 95.8 |
|  | ateeqq | 18.3 | 0.8 | 80.9 |
|  | commfor | 39.6 | 0.4 | 91.9 |
| combo | Basma | 81.4 | 2.4 | 96.5 |
|  | thermostatic | 65.3 | 1.0 | 95.2 |
|  | ateeqq | 20.5 | 0.7 | 79.9 |
|  | commfor | 29.9 | 0.4 | 86.1 |
| crop50 | Basma | 82.7 | 2.4 | 96.4 |
|  | thermostatic | 67.9 | 0.4 | 96.7 |
|  | ateeqq | 11.0 | 0.1 | 79.0 |
|  | commfor | 42.3 | 0.3 | 93.0 |
| jpeg60 | Basma | 81.9 | 2.3 | 95.8 |
|  | thermostatic | 76.1 | 1.6 | 95.8 |
|  | ateeqq | 19.5 | 0.6 | 79.9 |
|  | commfor | 36.9 | 0.2 | 91.5 |
| resize50 | Basma | 83.0 | 2.4 | 96.2 |
|  | thermostatic | 60.7 | 0.3 | 95.9 |
|  | ateeqq | 23.6 | 1.6 | 78.9 |
|  | commfor | 37.4 | 1.1 | 87.3 |
| screenshot | Basma | 80.5 | 2.4 | 96.0 |
|  | thermostatic | 65.3 | 0.8 | 96.1 |
|  | ateeqq | 23.1 | 0.8 | 80.1 |
|  | commfor | 32.8 | 0.8 | 86.0 |
| social | Basma | 80.5 | 2.8 | 96.0 |
|  | thermostatic | 51.1 | 1.5 | 91.8 |
|  | ateeqq | 9.9 | 0.4 | 75.0 |
|  | commfor | 16.0 | 0.2 | 76.7 |
| webp50 | Basma | 81.6 | 3.4 | 95.1 |
|  | thermostatic | 74.2 | 2.1 | 95.2 |
|  | ateeqq | 21.5 | 1.0 | 79.8 |
|  | commfor | 33.1 | 1.4 | 84.2 |

## AI generator families (held-out pools), TPR % per system

| family | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| cf_eval | 14960 | 78.5 | 69.3 | 18.7 | 43.0 |
| wildrf | 9910 | 86.2 | 73.4 | 33.4 | 46.5 |
| genimage | 7560 | 51.5 | 67.0 | 0.8 | 57.1 |
| grok | 6670 | 96.9 | 73.0 | 15.6 | 20.6 |
| recraft | 4150 | 94.1 | 78.3 | 18.5 | 21.5 |
| seedream | 2810 | 91.8 | 49.5 | 17.3 | 7.0 |
| reve | 2280 | 77.8 | 35.8 | 17.1 | 7.2 |
| hunyuan | 1700 | 99.9 | 98.2 | 41.1 | 60.9 |
| qwen | 1440 | 95.1 | 67.6 | 25.1 | 8.6 |
| muse | 430 | 92.3 | 49.1 | 11.4 | 19.1 |
| mai | 280 | 100.0 | 87.5 | 21.4 | 16.1 |
| ernie | 240 | 100.0 | 77.5 | 28.7 | 7.1 |

## Real sources (held-out pools), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| genimage | 12440 | 1.6 | 0.5 | 0.9 | 0.5 |
| wildrf | 10090 | 3.0 | 2.5 | 0.2 | 0.9 |
| flickr2k | 6000 | 1.8 | 0.8 | 0.2 | 0.4 |
| openimages | 6000 | 1.7 | 0.7 | 0.9 | 0.4 |
| cf_eval | 5040 | 4.6 | 1.8 | 3.4 | 0.6 |

## Pool x condition (Basma TPR % / FPR %)

| pool | blur1 | clean | color | combo | crop50 | jpeg60 | resize50 | screenshot | social | webp50 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval:cf_eval | 81.5 / 4.8 | 81.4 / 4.4 | 75.5 / 3.2 | 77.6 / 4.6 | 80.3 / 4.6 | 78.6 / 3.4 | 79.5 / 5.0 | 76.9 / 4.0 | 74.9 / 4.4 | 78.7 / 7.7 |
| eval:genimage | 50.0 / 1.4 | 53.2 / 1.2 | 49.3 / 1.1 | 52.2 / 2.1 | 56.4 / 1.3 | 50.8 / 1.8 | 51.3 / 1.1 | 48.5 / 1.8 | 52.6 / 2.1 | 50.3 / 2.5 |
| eval:real_flickr2k | – / 1.3 | – / 1.0 | – / 0.7 | – / 1.5 | – / 1.5 | – / 1.5 | – / 2.2 | – / 2.3 | – / 3.2 | – / 2.7 |
| eval:real_openimages | – / 2.2 | – / 1.5 | – / 0.8 | – / 1.8 | – / 1.3 | – / 2.3 | – / 1.8 | – / 1.5 | – / 1.8 | – / 2.0 |
| eval:unseen_vendors | 95.2 / – | 95.0 / – | 91.6 / – | 93.0 / – | 93.8 / – | 93.8 / – | 95.0 / – | 92.2 / – | 92.2 / – | 93.3 / – |
| eval:wildrf | 87.2 / 3.6 | 87.8 / 2.8 | 83.5 / 1.8 | 85.7 / 2.4 | 83.7 / 3.7 | 86.7 / 2.9 | 88.2 / 3.0 | 86.6 / 2.9 | 86.3 / 3.2 | 86.2 / 3.6 |

## Calibration reals (never trained on; used to set every threshold), FPR % per system

| source | rows | Basma | thermostatic | ateeqq | commfor |
|---|---|---|---|---|---|
| vizwiz | 8150 | 0.9 | 2.1 | 0.1 | 1.9 |
| voc2012 | 11850 | 0.7 | 0.3 | 0.2 | 0.4 |

## Validation (train sources, 8 % held out by hash), Basma

| condition | TPR % | FPR % | AUROC % |
|---|---|---|---|
| blur1 | 98.5 | 0.7 | 99.9 |
| clean | 98.2 | 0.4 | 99.9 |
| color | 97.0 | 0.2 | 99.9 |
| combo | 96.1 | 1.0 | 99.7 |
| crop50 | 97.6 | 0.7 | 99.8 |
| jpeg60 | 97.3 | 0.8 | 99.8 |
| resize50 | 97.0 | 0.9 | 99.8 |
| screenshot | 95.9 | 1.4 | 99.7 |
| social | 93.6 | 1.1 | 99.5 |
| webp50 | 96.6 | 1.6 | 99.7 |

## Verdicts on the held-out pools

| label:verdict | rows |
|---|---|
| ai:AI | 42884 |
| ai:Real | 7153 |
| ai:Uncertain | 2393 |
| real:AI | 944 |
| real:Real | 37676 |
| real:Uncertain | 950 |

