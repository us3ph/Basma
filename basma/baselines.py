"""Public detectors scored on exactly the same images and conditions as Basma (field 'score', higher = AI).

  python -m basma.baselines --name thermostatic --data-dirs ... --img-root ... --out runs/x/baselines
Baselines:
  thermostatic: Thermostatic/community-forensics-frontier-detector-2026-08 (Community Forensics ViT-S/16 384 fine-tuned
                Aug 2026, ONNX fp16; resize short side 440, centre crop 384, ImageNet norm; single raw logit)
  commfor:      OwensLab/commfor-model-384 (the official Community Forensics CVPR 2025 classifier, timm vit_small_patch16_384)
  ateeqq:       Ateeqq/ai-vs-human-image-detector (SigLIP2 fine-tune, labels 0 = ai, 1 = human)"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .attacks import EVAL_CONDITIONS
from .data import eval_seed, extract_all, load_manifests, path_of
from .attacks import attack_eval
from .score import select_rows

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32); IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def short_side_center(img, short, crop):
    w, h = img.size
    s = short / min(w, h)
    img = img.resize((max(crop, int(round(w * s))), max(crop, int(round(h * s)))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - crop) // 2, (h - crop) // 2
    return img.crop((l, t, l + crop, t + crop))


class CondDataset(Dataset):
    def __init__(self, rows, img_root, cond, prep):
        self.rows, self.img_root, self.cond, self.prep = rows, img_root, cond, prep

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = Image.open(path_of(r, self.img_root)).convert("RGB")
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        img = attack_eval(img, self.cond, eval_seed(r["id"], self.cond))
        return self.prep(img), i


class Thermostatic:
    name = "thermostatic"

    def __init__(self, device):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("Thermostatic/community-forensics-frontier-detector-2026-08", "community_forensics_frontier_fp16.onnx")
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(p, providers=prov)
        self.inp = self.sess.get_inputs()[0].name

    def prep(self, img):
        x = np.asarray(short_side_center(img, 440, 384), dtype=np.float32) / 255.0
        return torch.from_numpy(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).copy())

    def __call__(self, x):
        return self.sess.run(None, {self.inp: x.numpy().astype(np.float32)})[0].reshape(-1)


class CommFor:
    name = "commfor"

    def __init__(self, device):
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download("OwensLab/commfor-model-384", "model.safetensors"))
        self.model = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
        # the mixin checkpoint wraps the timm model; strip the prefix
        sd = {k.split(".", 1)[1] if k.startswith(("model.", "net.", "backbone.")) else k: v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print("commfor load: missing", missing[:5], "unexpected", unexpected[:5], flush=True)
        assert not [k for k in missing if "head" in k or "blocks" in k], "Community Forensics checkpoint did not map"
        self.model.to(device).eval(); self.device = device

    def prep(self, img):
        x = np.asarray(short_side_center(img, 440, 384), dtype=np.float32) / 255.0
        return torch.from_numpy(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).copy())

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x.to(self.device)).float().reshape(-1).cpu().numpy()


class Ateeqq:
    name = "ateeqq"

    def __init__(self, device):
        from transformers import AutoImageProcessor, SiglipForImageClassification
        self.proc = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
        self.model = SiglipForImageClassification.from_pretrained("Ateeqq/ai-vs-human-image-detector").to(device).eval()
        self.device = device
        self.ai_idx = int(self.model.config.label2id.get("ai", 0))

    def prep(self, img):
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def __call__(self, x):
        lg = self.model(pixel_values=x.to(self.device)).logits.float()
        return (lg[:, self.ai_idx] - lg[:, 1 - self.ai_idx]).cpu().numpy()


BASELINES = {"thermostatic": Thermostatic, "commfor": CommFor, "ateeqq": Ateeqq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(BASELINES))
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "eval:*"])
    ap.add_argument("--conds", nargs="+", default=list(EVAL_CONDITIONS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = select_rows(load_manifests(a.data_dirs), a.splits, a.limit); extract_all(a.data_dirs, a.img_root)
    bl = BASELINES[a.name](device)
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    with open(os.path.join(a.out, f"{a.name}.jsonl"), "w", encoding="utf-8") as fh:
        for cond in a.conds:
            dl = DataLoader(CondDataset(rows, a.img_root, cond, bl.prep), batch_size=a.batch_size, num_workers=a.workers)
            for x, idx in dl:
                s = bl(x)
                for b in range(len(idx)):
                    r = rows[int(idx[b])]
                    fh.write(json.dumps({"id": r["id"], "label": r["label"], "source": r["source"], "gen": r["gen"], "family": r["family"], "vendor": r["vendor"],
                                         "split": r["split"], "cond": cond, "score": float(s[b])}) + "\n")
            print(f"  {a.name} {cond}: {len(rows)} images ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
