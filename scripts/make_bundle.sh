#!/usr/bin/env bash
# Zip the code (no data/runs) for upload as a Kaggle dataset "basma-code".
cd "$(dirname "$0")/.." && rm -f basma_code.zip && zip -qr basma_code.zip . -x "data/*" "runs/*" "logs/*" ".git/*" "*/__pycache__/*" "*.pyc" "basma_code.zip" "web/assets/onnx/*" && ls -lh basma_code.zip
