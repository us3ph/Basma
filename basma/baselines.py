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


class ViTS16(torch.nn.Module):
    """Minimal ViT-S/16 (384 px) with timm parameter names, so the Community Forensics checkpoints load without timm."""
    def __init__(self, img=384, patch=16, dim=384, depth=12, heads=6, mlp=1536):
        super().__init__()
        nn = torch.nn
        self.patch_embed = nn.Module(); self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.pos_embed = nn.Parameter(torch.zeros(1, (img // patch) ** 2 + 1, dim))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            b = nn.Module(); b.norm1 = nn.LayerNorm(dim, eps=1e-6); b.attn = nn.Module(); b.attn.qkv = nn.Linear(dim, 3 * dim); b.attn.proj = nn.Linear(dim, dim)
            b.norm2 = nn.LayerNorm(dim, eps=1e-6); b.mlp = nn.Module(); b.mlp.fc1 = nn.Linear(dim, mlp); b.mlp.fc2 = nn.Linear(mlp, dim)
            self.blocks.append(b)
        self.norm = nn.LayerNorm(dim, eps=1e-6); self.head = nn.Linear(dim, 1); self.heads = heads

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1) + self.pos_embed
        for b in self.blocks:
            h = b.norm1(x); qkv = b.attn.qkv(h).reshape(B, h.shape[1], 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            a = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            x = x + b.attn.proj(a.transpose(1, 2).reshape(B, h.shape[1], -1))
            x = x + b.mlp.fc2(torch.nn.functional.gelu(b.mlp.fc1(b.norm2(x))))
        return self.head(self.norm(x)[:, 0])


def load_vit_s384(repo, device):
    """Community-Forensics-style checkpoint (timm vit_small_patch16_384 names, 1 logit) from a Hugging Face model repo."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    sd = {k.split(".", 1)[1] if k.startswith(("model.", "net.", "backbone.", "vit.", "module.")) else k: v for k, v in sd.items()}
    model = ViTS16()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"{repo}: checkpoint does not map onto ViT-S/16-384 (missing {missing[:4]}, unexpected {unexpected[:4]})")
    print(f"{repo}: loaded (unexpected keys ignored: {len(unexpected)} {unexpected[:3]})", flush=True)
    return model.to(device).eval()


class Thermostatic:
    """Prefers the safetensors weights in timm (GPU); falls back to the fp16 ONNX on CPU."""
    name = "thermostatic"
    REPO = "Thermostatic/community-forensics-frontier-detector-2026-08"

    def __init__(self, device):
        self.device, self.model, self.sess = device, None, None
        try:
            self.model = load_vit_s384(self.REPO, device)
        except Exception as e:  # noqa: BLE001
            print(f"thermostatic: timm path failed ({type(e).__name__}: {str(e)[:100]}), using ONNX", flush=True)
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(self.REPO, "community_forensics_frontier_fp16.onnx")
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(p, providers=prov); self.inp = self.sess.get_inputs()[0].name

    def prep(self, img):
        x = np.asarray(short_side_center(img, 440, 384), dtype=np.float32) / 255.0
        return torch.from_numpy(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1).copy())

    @torch.no_grad()
    def __call__(self, x):
        if self.model is not None:
            return self.model(x.to(self.device)).float().reshape(-1).cpu().numpy()
        return self.sess.run(None, {self.inp: x.numpy().astype(np.float32)})[0].reshape(-1)


class CommFor:
    name = "commfor"

    def __init__(self, device):
        self.model = load_vit_s384("OwensLab/commfor-model-384", device); self.device = device

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
    ap.add_argument("--splits", nargs="+", default=["val", "calib", "eval:*"])
    ap.add_argument("--conds", nargs="+", default=list(EVAL_CONDITIONS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse", default=None, help="earlier scores of this baseline (same shards): rows with the same id and condition are copied, "
                                                   "since the stored image and the per-image attack seed are identical")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = select_rows(load_manifests(a.data_dirs), a.splits, a.limit); extract_all(a.data_dirs, a.img_root)
    cache = {}
    if a.reuse and os.path.exists(a.reuse):
        with open(a.reuse, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    c = json.loads(line); cache[(c["id"], c["cond"])] = c["score"]
        print(f"  reusing {len(cache)} earlier {a.name} scores from {a.reuse}", flush=True)
    bl = None
    os.makedirs(a.out, exist_ok=True); t0 = time.time()

    def emit(fh, r, cond, score):
        fh.write(json.dumps({"id": r["id"], "label": r["label"], "source": r["source"], "gen": r["gen"], "family": r["family"], "vendor": r["vendor"],
                             "split": r["split"], "cond": cond, "score": float(score)}) + "\n")
    with open(os.path.join(a.out, f"{a.name}.jsonl"), "w", encoding="utf-8") as fh:
        for cond in a.conds:
            todo = [r for r in rows if (r["id"], cond) not in cache]
            for r in rows:
                if (r["id"], cond) in cache:
                    emit(fh, r, cond, cache[(r["id"], cond)])
            if todo:
                bl = bl or BASELINES[a.name](device)
                dl = DataLoader(CondDataset(todo, a.img_root, cond, bl.prep), batch_size=a.batch_size, num_workers=a.workers)
                for x, idx in dl:
                    s = bl(x)
                    for b in range(len(idx)):
                        emit(fh, todo[int(idx[b])], cond, s[b])
            print(f"  {a.name} {cond}: {len(rows)} images, {len(todo)} newly scored ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
