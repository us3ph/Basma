# Equal false-positive comparison (held-out pools)

Every system's threshold is set on the 39570 held-out real rows themselves (image x condition), so all systems flag the same share of unseen real photos and only ranking differs. Diagnostic only: shipped thresholds come from validation and calibration reals. AI rows: 52430.

## AI images caught (TPR %) at each held-out false-positive rate

| system | FPR 1 % | FPR 2 % | FPR 5 % |
|---|---|---|---|
| Basma | 74.1 | 80.3 | 86.6 |
| thermostatic | 67.7 | 73.9 | 82.7 |
| commfor | 41.2 | 47.4 | 59.0 |
| ateeqq | 19.9 | 27.5 | 39.7 |

## Per AI pool at 2 % held-out FPR (TPR %)

| pool | rows | Basma | thermostatic | commfor | ateeqq |
|---|---|---|---|---|---|
| eval:cf_eval | 14960 | 76.7 | 73.4 | 54.2 | 25.4 |
| eval:genimage | 7560 | 49.7 | 70.4 | 66.3 | 2.2 |
| eval:wildrf | 9910 | 84.3 | 78.1 | 59.7 | 44.8 |
| unseen:ernie | 240 | 99.6 | 85.4 | 19.6 | 39.6 |
| unseen:grok | 6670 | 96.1 | 78.0 | 28.9 | 27.0 |
| unseen:hunyuan | 1700 | 99.9 | 98.8 | 73.5 | 55.1 |
| unseen:mai | 280 | 100.0 | 92.1 | 24.6 | 37.1 |
| unseen:muse | 430 | 90.7 | 54.2 | 22.6 | 19.8 |
| unseen:qwen | 1440 | 94.4 | 74.5 | 17.8 | 39.7 |
| unseen:recraft | 4150 | 93.4 | 84.2 | 36.4 | 26.3 |
| unseen:reve | 2280 | 75.7 | 42.5 | 12.5 | 24.6 |
| unseen:seedream | 2810 | 90.1 | 56.3 | 13.4 | 26.4 |

## Per condition at 2 % held-out FPR (TPR %)

| condition | Basma | thermostatic | commfor | ateeqq |
|---|---|---|---|---|
| blur1 | 82.2 | 85.1 | 65.7 | 31.8 |
| clean | 82.3 | 83.3 | 55.2 | 29.8 |
| color | 78.0 | 74.8 | 52.1 | 25.6 |
| combo | 79.9 | 71.1 | 39.3 | 29.7 |
| crop50 | 81.2 | 72.4 | 51.5 | 16.8 |
| jpeg60 | 80.2 | 80.4 | 48.4 | 27.9 |
| resize50 | 81.8 | 65.0 | 47.4 | 32.4 |
| screenshot | 79.1 | 70.6 | 42.6 | 32.7 |
| social | 78.3 | 57.4 | 24.5 | 17.4 |
| webp50 | 79.9 | 79.0 | 47.4 | 30.7 |

