#!/usr/bin/env bash
# GPU kernel (T4 x2): train (time-capped) -> score held-out pools -> baselines on the same pools -> fusion + report
#   !bash scripts/kaggle_train.sh configs/kaggle_2xT4_base.json
set -uo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/kaggle_2xT4_base.json}"
OUT=$(python -c "import json; print(json.load(open('$CFG'))['out'])")
DATA=$(python -c "import json; print(' '.join(json.load(open('$CFG'))['data_dirs']))")
IMG=$(python -c "import json; print(json.load(open('$CFG'))['img_root'])")
mkdir -p logs
t0=$(date +%s); stamp(){ echo "[$(( ($(date +%s)-t0)/60 )) min] $*"; }
stamp "install";    pip install -q -r requirements.txt 2>&1 | tail -1
stamp "train";      python -m basma.train --config "$CFG"                  2>&1 | tee logs/train.log | grep -E '"event": "(data|plan|eval|saved|max-hours|done)"|Traceback|Error' || true
stamp "score";      python -m basma.score --model "$OUT/best" --data-dirs $DATA --img-root "$IMG" --out "$OUT/scores" --limit 2500 2>&1 | tee logs/score.log | grep -E "scored|scoring|Traceback" || true
for b in thermostatic ateeqq commfor; do
  stamp "baseline $b"; python -m basma.baselines --name $b --data-dirs $DATA --img-root "$IMG" --out "$OUT/baselines" --limit 2500 2>&1 | tee "logs/baseline_$b.log" | grep -E "^  |Traceback|Error" || true
done
BL=""; for b in thermostatic ateeqq commfor; do [ -s "$OUT/baselines/$b.jsonl" ] && BL="$BL $b=$OUT/baselines/$b.jsonl"; done
stamp "fusion";     python -m basma.fusion --scores "$OUT/scores/scores.jsonl" --baseline $BL --out "$OUT/report" 2>&1 | tee logs/fusion.log
python -m basma.report_md "$OUT/report/report.json" > docs/RESULTS.md && stamp "done -> $OUT/report/report.json, docs/RESULTS.md"
stamp "package";    tar czf /kaggle/working/basma_model.tgz "$OUT/best" "$OUT/report" docs/RESULTS.md && ls -lh /kaggle/working/basma_model.tgz
