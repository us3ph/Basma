#!/usr/bin/env bash
# Local dry run of every stage after training on the mini corpus (CPU). Waits for the mini training to finish.
cd "$(dirname "$0")/.."
while ! grep -q '"event": "done"' logs/train_mini.log 2>/dev/null; do sleep 10; done
D="data/mini"; I="data/img"; O="runs/mini"
python3 -m basma.score --model $O/best --data-dirs $D --img-root $I --out $O/scores --limit 40 --workers 2 --conds clean social screenshot 2>&1 | tail -4
python3 -m basma.baselines --name thermostatic --data-dirs $D --img-root $I --out $O/baselines --limit 40 --workers 2 --conds clean social screenshot 2>&1 | tail -3
python3 -m basma.baselines --name commfor --data-dirs $D --img-root $I --out $O/baselines --limit 40 --workers 2 --conds clean social screenshot 2>&1 | tail -3
python3 -m basma.fusion --scores $O/scores/scores.jsonl --baseline thermostatic=$O/baselines/thermostatic.jsonl commfor=$O/baselines/commfor.jsonl --out $O/report 2>&1 | tail -30
python3 -m basma.report_md $O/report/report.json > $O/report/RESULTS.md && head -30 $O/report/RESULTS.md
python3 scripts/export_onnx.py --model $O/best --fusion $O/report/fusion.json --data-dirs $D --img-root $I --out web/assets/onnx --n 20 2>&1 | tail -5
python3 -m basma.predict --model $O/best --fusion $O/report/fusion.json $(ls data/img/*.jpg | head -3)
echo CHAIN DONE
