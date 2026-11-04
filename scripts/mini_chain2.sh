#!/usr/bin/env bash
cd "$(dirname "$0")/.."
D="data/mini"; I="data/img"; O="runs/mini"
python3 -m basma.baselines --name commfor --data-dirs $D --img-root $I --out $O/baselines --limit 40 --workers 2 --conds clean social screenshot 2>&1 | tail -3
python3 -m basma.fusion --scores $O/scores/scores.jsonl --baseline thermostatic=$O/baselines/thermostatic.jsonl commfor=$O/baselines/commfor.jsonl --out $O/report 2>&1 | tail -30
python3 -m basma.report_md $O/report/report.json > $O/report/RESULTS.md && head -40 $O/report/RESULTS.md
python3 scripts/export_onnx.py --model $O/best --fusion $O/report/fusion.json --data-dirs $D --img-root $I --out web/assets/onnx --n 20 2>&1 | tail -6
python3 -m basma.predict --model $O/best --fusion $O/report/fusion.json $(ls data/img/*.jpg | head -3)
echo CHAIN2 DONE
