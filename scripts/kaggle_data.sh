#!/usr/bin/env bash
# CPU kernel: build one data plan into /kaggle/working (becomes the kernel's output dataset).
#   !bash scripts/kaggle_data.sh cf|recent|eval
set -uo pipefail
cd "$(dirname "$0")/.."
PLAN="${1:?plan}"
pip install -q pyarrow fsspec huggingface-hub pillow 2>&1 | tail -1
python -m basma.build_data --plan "$PLAN" --out /kaggle/working --workers 4 2>&1 | tee "/kaggle/working/build_${PLAN}.console.log" | grep -E "^\[|done|images|Traceback|Error" || true
du -sh /kaggle/working/shards; ls /kaggle/working | head
