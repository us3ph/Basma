#!/usr/bin/env bash
# Publish the browser build: ONNX weights + fusion to a Hugging Face model repo, the page to a free Static Space.
#   bash scripts/deploy_hf.sh [hf-user]        (SKIP_FP32=1 skips the 346 MB fp32 fallback model;
#                                              PAGE_ONLY=1 updates only the page in the Space, no model upload)
set -euo pipefail
cd "$(dirname "$0")/.."
USER_="${1:-$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")}"
MODEL_REPO="$USER_/basma-base"; SPACE_REPO="$USER_/basma-detector"
export HF_HUB_DISABLE_XET=1
python3 - "$MODEL_REPO" "$SPACE_REPO" <<'PY'
import sys, os, time, shutil, pathlib
from huggingface_hub import HfApi
model, space = sys.argv[1], sys.argv[2]
api = HfApi()
if not api.repo_exists(model, repo_type="model"):
    api.create_repo(model, repo_type="model")
card = ("---\nlicense: mit\ntags: [ai-image-detection, dinov2, onnx]\n---\n# Basma (base)\nBrowser build of https://github.com/us3ph/Basma : "
        "ONNX detector (`onnx/model_w8.onnx` weight-only 8-bit, `onnx/model.onnx` fp32) + `fusion.json`. Labels 0 = real, 1 = AI. Results in RESULTS.md.\n")
open("/tmp/basma_card.md", "w").write(card)
files = [] if os.environ.get("PAGE_ONLY") == "1" else [("/tmp/basma_card.md", "README.md"), ("web/assets/fusion.json", "fusion.json"), ("docs/RESULTS.md", "RESULTS.md"),
         ("web/assets/onnx/model_w8.onnx", "onnx/model_w8.onnx")]
if os.environ.get("SKIP_FP32") != "1" and files:                     # fp32 is only a fallback for browsers that cannot run 8-bit MatMulNBits
    files.append(("web/assets/onnx/model.onnx", "onnx/model.onnx"))
for src, dst in files:
    if not os.path.exists(src):
        print("skip (missing)", src); continue
    for attempt in range(1, 6):
        try:
            api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=model, repo_type="model"); print("uploaded", dst); break
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}: {type(e).__name__}: {str(e)[:120]}"); time.sleep(20 * attempt)
if not api.repo_exists(space, repo_type="space"):
    api.create_repo(space, repo_type="space", space_sdk="static")
stage = pathlib.Path("/tmp/basma_space"); shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
for f in ["index.html", "app.js", "ort-worker.js", "resample.js", "views.js", "fusion.js", "frame-border.js", "ai-blob.js", "ghost-type.js"]:
    shutil.copy(f"web/{f}", stage / f)
(stage / "assets").mkdir(); shutil.copy("web/assets/fusion.json", stage / "assets" / "fusion.json")
(stage / "README.md").write_text("---\ntitle: Basma\nemoji: 🫆\ncolorFrom: blue\ncolorTo: indigo\nsdk: static\npinned: false\n---\nAI-image detector that runs in your browser. https://github.com/us3ph/Basma\n")
api.upload_folder(repo_id=space, folder_path=str(stage), repo_type="space")
print("space:", f"https://huggingface.co/spaces/{space}")
PY
