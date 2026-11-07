#!/usr/bin/env bash
# GPU kernel (T4 x2): train (time-capped) -> score held-out pools -> baselines on the same pools -> fusion + report
#   !bash scripts/kaggle_train.sh configs/kaggle_run.json   (the notebook writes kaggle_run.json with the discovered data dirs)
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
if [ ! -d "$OUT/best" ]; then echo "TRAINING FAILED: no $OUT/best. Last lines of logs/train.log:"; tail -40 logs/train.log; exit 1; fi
stamp "score";      python -m basma.score --model "$OUT/best" --data-dirs $DATA --img-root "$IMG" --out "$OUT/scores" --limit 2000 2>&1 | tee logs/score.log | grep -E "scored|scoring|Traceback" || true
for b in thermostatic ateeqq commfor; do
  REUSE=""; [ -n "${BASELINE_CACHE:-}" ] && [ -s "$BASELINE_CACHE/$b.jsonl" ] && REUSE="--reuse $BASELINE_CACHE/$b.jsonl"
  stamp "baseline $b"; python -m basma.baselines --name $b --data-dirs $DATA --img-root "$IMG" --out "$OUT/baselines" --limit 2000 $REUSE 2>&1 | tee "logs/baseline_$b.log" | grep -E "^  |Traceback|Error" || true
done
BL=""; for b in thermostatic ateeqq commfor; do [ -s "$OUT/baselines/$b.jsonl" ] && BL="$BL $b=$OUT/baselines/$b.jsonl"; done
stamp "fusion";     python -m basma.fusion --scores "$OUT/scores/scores.jsonl" --baseline $BL --out "$OUT/report" 2>&1 | tee logs/fusion.log
python -m basma.report_md "$OUT/report/report.json" > docs/RESULTS.md && stamp "done -> $OUT/report/report.json, docs/RESULTS.md"
# Basma Bench, part 2: the NTIRE 2026 public test (2,500 images, never trained on) with this run's model and fusion;
# locally: python scripts/bench_track.py add --version vN ... --ntire $OUT/ntire/scores.jsonl (see docs/BENCH.md)
stamp "bench ntire"; python - <<'PYN' && python scripts/bench_ntire.py --model "$OUT/best" --fusion "$OUT/report/fusion.json" --images /tmp/ntire/img \
    --labels /tmp/ntire/test_labels.csv --out "$OUT/ntire" --baseline none 2>&1 | tee logs/ntire.log | grep -E "scored|Basma|Traceback" || echo "NTIRE bench skipped"
import subprocess
from huggingface_hub import hf_hub_download
R = "deepfakesMSU/NTIRE-RobustAIGenDetection-test-public"
z = hf_hub_download(R, "test_images.zip", repo_type="dataset", local_dir="/tmp/ntire"); hf_hub_download(R, "test_labels.csv", repo_type="dataset", local_dir="/tmp/ntire")
subprocess.run(["unzip", "-q", "-o", z, "-d", "/tmp/ntire/img"], check=True)
PYN
stamp "package";    tar czf /kaggle/working/basma_model.tgz "$OUT/best" "$OUT/report" $( [ -d "$OUT/ntire" ] && echo "$OUT/ntire" ) docs/RESULTS.md $( [ -f docs/LEAKGUARD.md ] && echo docs/LEAKGUARD.md ) && ls -lh /kaggle/working/basma_model.tgz
