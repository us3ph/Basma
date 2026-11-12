"""Train the detector. Time-capped (max_hours), best checkpoint by validation TPR at 1 % FPR under attacks.

  python -m basma.train --config configs/kaggle_2xT4_base.json
Config keys: backbone, freeze_blocks, batch_size (images per step, each gives 1 + n_crops views), n_crops, epochs,
lr_backbone, lr_head, layer_decay, weight_decay, warmup_steps, max_hours, eval_every, fp16, workers, data_dirs,
img_root, out, limit, seed, val_limit, eval_conditions."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import EvalDataset, TrainDataset, balanced_weights, collate_eval, collate_train, extract_all, load_manifests
from .metrics import auroc, tpr_at_fpr
from .model import BasmaDetector, detector_loss

DEFAULTS = {"backbone": "facebook/dinov2-with-registers-base", "freeze_blocks": 4, "batch_size": 32, "n_crops": 1, "epochs": 2,
            "lr_backbone": 3e-5, "lr_head": 5e-4, "layer_decay": 0.85, "weight_decay": 0.05, "warmup_steps": 300, "max_hours": 3.5,
            "eval_every": 1000, "fp16": True, "workers": 4, "data_dirs": ["data/mini"], "img_root": "data/img", "out": "runs/mini",
            "limit": 0, "seed": 0, "val_limit": 1500, "eval_conditions": ["clean", "social", "screenshot"], "grad_clip": 1.0, "quantile": 0.25,
            "resume": ""}   # a run dir whose last/ holds model.pt + opt.pt: training continues from that step (chained Kaggle runs)


def log_json(path, obj):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")
    print(json.dumps(obj), flush=True)


@torch.no_grad()
def evaluate(model, rows, img_root, conds, device, batch_size=16, workers=4, fp16=True):
    """Mean-of-views logit per image under each condition -> AUROC and TPR at 1 % FPR per condition."""
    model.eval()
    res = {}
    for cond in conds:
        ds = EvalDataset(rows, img_root, cond)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, collate_fn=collate_eval)
        scores = np.zeros(len(rows)); labels = np.array([r["label"] for r in rows])
        for x, nv, idx, h, w in dl:
            B, V = x.shape[:2]
            flat = x.view(B * V, *x.shape[2:]).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                lg = model(flat)["logit"].float().view(B, V).cpu().numpy()
            for b in range(B):
                scores[idx[b]] = lg[b, :nv[b]].mean()
        res[cond] = {"auroc": auroc(labels, scores), "tpr@1%fpr": tpr_at_fpr(labels, scores, 0.01), "tpr@5%fpr": tpr_at_fpr(labels, scores, 0.05)}
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (JSON values)")
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(json.load(open(a.config)))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try: cfg[k] = json.loads(v)
        except json.JSONDecodeError: cfg[k] = v
    out = cfg["out"]; os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    t0 = time.time()

    rows = load_manifests(cfg["data_dirs"])
    extract_all(cfg["data_dirs"], cfg["img_root"])
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if cfg["limit"]:
        rs = np.random.RandomState(cfg["seed"]); rs.shuffle(train_rows); train_rows = train_rows[:cfg["limit"]]
    rs = np.random.RandomState(cfg["seed"] + 1); rs.shuffle(val_rows); val_rows = val_rows[:cfg["val_limit"]]
    from collections import Counter
    fam = Counter((r["label"], r["family"]) for r in train_rows)
    log_json(os.path.join(out, "train.log"), {"event": "data", "train": len(train_rows), "val": len(val_rows), "families": len(fam),
                                              "ai": sum(r["label"] for r in train_rows), "real": sum(1 - r["label"] for r in train_rows), "gpus": n_gpu})

    model = BasmaDetector(cfg["backbone"], quantile=cfg["quantile"])
    model.freeze_blocks(cfg["freeze_blocks"])
    model.to(device)
    groups = model.param_groups(cfg["lr_backbone"], cfg["lr_head"], cfg["layer_decay"], cfg["weight_decay"])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    net = torch.nn.DataParallel(model) if n_gpu > 1 else model
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"] and device.type == "cuda")

    ds = TrainDataset(train_rows, cfg["img_root"], n_crops=cfg["n_crops"], seed=cfg["seed"])
    weights = balanced_weights(train_rows)
    steps_per_epoch = max(1, len(train_rows) // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    log_json(os.path.join(out, "train.log"), {"event": "plan", "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                                              "views_per_step": cfg["batch_size"] * (1 + cfg["n_crops"])})

    def lr_at(step):
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        p = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    best, step, stop = -1.0, 0, False
    ckpt_dir = os.path.join(out, "best")
    if cfg["resume"]:
        best, step = resume(cfg["resume"], model, opt, scaler, ckpt_dir)
        log_json(os.path.join(out, "train.log"), {"event": "resumed", "from": cfg["resume"], "step": step, "best": best})
    for epoch in range(step // steps_per_epoch, cfg["epochs"]):
        ds.epoch = epoch
        n_samples = (steps_per_epoch - step % steps_per_epoch) * cfg["batch_size"]     # the rest of a resumed epoch
        sampler = WeightedRandomSampler(torch.as_tensor(weights), num_samples=n_samples, replacement=True,
                                        generator=torch.Generator().manual_seed(cfg["seed"] * 100 + epoch + step))
        dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["workers"], collate_fn=collate_train,
                        drop_last=True, persistent_workers=cfg["workers"] > 0, prefetch_factor=4 if cfg["workers"] > 0 else None)
        run_loss, run_main, n_run, t_ep = 0.0, 0.0, 0, time.time()
        for x, y, bucket, idx in dl:
            B, V = x.shape[:2]
            flat = x.view(B * V, *x.shape[2:]).to(device, non_blocking=True)
            yy = y.to(device).repeat_interleave(V)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * lr_at(step)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["fp16"] and device.type == "cuda"):
                o = net(flat)
            o = {k: v.float() for k, v in o.items()}
            loss, main_loss = detector_loss(o, yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg["grad_clip"])
            scaler.step(opt); scaler.update()
            run_loss += float(loss); run_main += float(main_loss); n_run += 1; step += 1
            if step % 50 == 0:
                el = time.time() - t0
                log_json(os.path.join(out, "train.log"), {"event": "step", "step": step, "epoch": epoch, "loss": run_loss / n_run, "main": run_main / n_run,
                                                          "lr": opt.param_groups[-1]["lr"], "img/s": n_run * B * V / (time.time() - t_ep), "min": el / 60})
                run_loss, run_main, n_run, t_ep = 0.0, 0.0, 0, time.time()
            hit_time = (time.time() - t0) / 3600 > cfg["max_hours"]
            if step % cfg["eval_every"] == 0 or hit_time or step == total_steps:
                res = evaluate(model, val_rows, cfg["img_root"], cfg["eval_conditions"], device, workers=cfg["workers"], fp16=cfg["fp16"])
                score = float(np.mean([r["tpr@1%fpr"] for r in res.values()]))
                log_json(os.path.join(out, "train.log"), {"event": "eval", "step": step, "val": res, "score": score, "min": (time.time() - t0) / 60})
                if score > best:
                    best = score
                    save(model, cfg, ckpt_dir, {"step": step, "score": score, "val": res})
                    log_json(os.path.join(out, "train.log"), {"event": "saved best", "step": step, "score": score})
            if hit_time:
                log_json(os.path.join(out, "train.log"), {"event": "max-hours reached", "step": step}); stop = True; break
        if stop:
            break
    save(model, cfg, os.path.join(out, "last"), {"step": step, "best": best, "finished": step >= total_steps})
    torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "best": best}, os.path.join(out, "last", "opt.pt"))
    log_json(os.path.join(out, "train.log"), {"event": "done", "best": best, "step": step, "total_steps": total_steps, "hours": (time.time() - t0) / 3600})


def save(model, cfg, d, info):
    os.makedirs(d, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "backbone": model.backbone_name, "quantile": model.quantile, "config": cfg, "info": info}, os.path.join(d, "model.pt"))
    json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=1)


def resume(run_dir, model, opt, scaler, best_dir):
    """Continue a time-capped run: weights + optimizer + scaler + step from <run_dir>/last; its best/ is carried over so
    a later evaluation only replaces it when it is better."""
    import shutil
    ck = torch.load(os.path.join(run_dir, "last", "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    st = torch.load(os.path.join(run_dir, "last", "opt.pt"), map_location="cpu", weights_only=False)
    opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
    if os.path.isdir(os.path.join(run_dir, "best")) and not os.path.isdir(best_dir):
        shutil.copytree(os.path.join(run_dir, "best"), best_dir)
    return float(st["best"]), int(st["step"])


def load_model(path, device="cpu"):
    ck = torch.load(os.path.join(path, "model.pt") if os.path.isdir(path) else path, map_location="cpu", weights_only=False)
    m = BasmaDetector(ck["backbone"], quantile=ck.get("quantile", 0.25), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


if __name__ == "__main__":
    main()
