"""Basma Bench: our own frozen test, scored for every new version so progress is measured on the same ground.

Public leaderboards mostly measure who trained on that benchmark's own data, so we keep a fixed set of images that
no version ever trains on, picks checkpoints with or sets thresholds with:
  * held-out pools: the eval:* rows of runs/<x>/scores/scores.jsonl (Community Forensics Eval, GenImage, WildRF,
    unseen commercial generators, Flickr2K + Open Images real photos), each image as-is and after 9 edits
    (JPEG, WebP, resize, screenshot, social-media pipeline, colour, blur/noise, crop, combination);
  * NTIRE 2026 public test: 2,500 images, half with stacked distortions (scripts/bench_ntire.py output).
Every version is judged with its own shipped fusion.json: the AI / Real / Uncertain answers a user would get.

  python scripts/bench_track.py add --version v2 --date 2026-11-05 --scores runs/kaggle_v2/scores/scores.jsonl \\
      --fusion runs/kaggle_v2/report/fusion.json [--ntire runs/kaggle_v2/ntire/scores.jsonl] [--note "..."]
  python scripts/bench_track.py report > docs/BENCH.md
History: bench/history.jsonl, one line per version (re-adding a version replaces its line)."""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from basma.predict import load_fusion  # noqa: E402

HIST = pathlib.Path(__file__).resolve().parents[1] / "bench" / "history.jsonl"
PARTS = [("held_clean", "Held-out, original images"), ("held_edited", "Held-out, after edits / sharing"),
         ("ntire_clean", "NTIRE, clean"), ("ntire_distorted", "NTIRE, distorted")]
POOL_NAMES = {"eval:unseen_vendors": "Unseen commercial generators (Seedream, Reve, Grok, Qwen, …)", "eval:cf_eval": "Community Forensics Eval",
              "eval:genimage": "GenImage (older generators, ImageNet photos)", "eval:wildrf": "WildRF (Reddit / Facebook / Twitter)",
              "eval:real_flickr2k": "Flickr2K photos", "eval:real_openimages": "Open Images photos"}
COND_NAMES = {"clean": "original", "jpeg60": "JPEG 60", "webp50": "WebP 50", "resize50": "resized to 50 %", "screenshot": "screenshot",
              "social": "social-media pipeline", "color": "colour edit", "blur1": "blur / noise", "crop50": "crop 50 %", "combo": "several edits"}


def summary(items):
    """items: (label, verdict, score) -> rates; AI 'detected' = called AI, human 'detected' = called Real."""
    ai = [v for y, v, _ in items if y == 1]; hu = [v for y, v, _ in items if y == 0]
    rate = lambda xs, v: round(100 * sum(x == v for x in xs) / len(xs), 2) if xs else None  # noqa: E731
    ys = [y for y, _, _ in items]
    au = round(float(roc_auc_score(ys, [s for _, _, s in items])), 4) if 0 < sum(ys) < len(ys) else None
    return {"n_ai": len(ai), "n_real": len(hu), "ai_detected": rate(ai, "AI"), "ai_missed": rate(ai, "Real"), "ai_uncertain": rate(ai, "Uncertain"),
            "human_detected": rate(hu, "Real"), "human_flagged": rate(hu, "AI"), "human_uncertain": rate(hu, "Uncertain"), "auroc": au}


def add(a):
    f = load_fusion(a.fusion)
    held = []
    for line in open(a.scores):
        r = json.loads(line)
        if not r["split"].startswith("eval:"):
            continue                                   # val / calib rows chose the checkpoint and thresholds: not bench
        p = f.prob(r); held.append((r, (r["label"], f.verdict(p), f.raw(r))))
    entry = {"version": a.version, "date": a.date, "note": a.note, "thr_ai": f.thr_ai, "thr_real": f.thr_real, "parts": {}, "pools": {}, "conds": {}}
    entry["parts"]["held_clean"] = summary([t for r, t in held if r["cond"] == "clean"])
    entry["parts"]["held_edited"] = summary([t for r, t in held if r["cond"] != "clean"])
    for pool in sorted({r["split"] for r, _ in held}):
        entry["pools"][pool] = {c: summary([t for r, t in held if r["split"] == pool and (r["cond"] == "clean") == (c == "clean")]) for c in ("clean", "edited")}
    for cond in COND_NAMES:
        its = [t for r, t in held if r["cond"] == cond]
        if its: entry["conds"][cond] = summary(its)
    if a.ntire:
        nt = [json.loads(line) for line in open(a.ntire)]   # verdicts were made by bench_ntire.py with the same fusion.json
        for key, d in (("ntire_clean", 0), ("ntire_distorted", 1)):
            entry["parts"][key] = summary([(r["label"], r["verdict"], r["basma"]) for r in nt if r["is_distorted"] == d])
    hist = [json.loads(line) for line in open(HIST)] if HIST.exists() else []
    hist = [h for h in hist if h["version"] != a.version] + [entry]
    HIST.parent.mkdir(exist_ok=True); HIST.write_text("".join(json.dumps(h) + "\n" for h in hist))
    print(json.dumps(entry["parts"], indent=1))


def report(_a):
    hist = [json.loads(line) for line in open(HIST)]
    pc = lambda v: "–" if v is None else f"{v:.1f} %"  # noqa: E731
    pair = lambda s: "–" if not s else f"{pc(s['ai_detected'])} / {pc(s['human_detected'])}"  # noqa: E731
    L = ["# Basma Bench", "",
         "Our own fixed test, scored for every version. No version ever trains on these images, picks its checkpoint with "
         "them or sets its thresholds with them, so the numbers show real progress rather than familiarity with a test. "
         "Each version answers with its own shipped settings (the AI / Real / Uncertain a user would see).", "",
         "* **Held-out pools**: 9,200 images from generators and photo sources never trained on (Community Forensics Eval, "
         "GenImage, WildRF from Reddit/Facebook/Twitter, unseen commercial generators, Flickr2K and Open Images photos), "
         "each scored as-is and after 9 edits (JPEG, WebP, resize, screenshot, social-media pipeline, colour, blur/noise, crop, several).",
         "* **NTIRE 2026 public test**: 2,500 images from another lab, half with stacked distortions.", "",
         "**AI detected** = AI images answered *AI*. **Human detected** = real photos answered *Real*. The rest is either "
         "*Uncertain* or wrong (tables below). AUROC is the threshold-free ranking quality (1.0 = perfect, 0.5 = guessing).", "",
         "## AI detected / Human detected", "",
         "| Version | Date | " + " | ".join(n for _, n in PARTS) + " |", "|---|---|" + "---|" * len(PARTS)]
    for h in hist:
        L.append(f"| **{h['version']}** | {h['date']} | " + " | ".join(pair(h["parts"].get(k)) for k, _ in PARTS) + " |")
    L += ["", "## Mistakes: AI called Real / Real called AI", "", "| Version | " + " | ".join(n for _, n in PARTS) + " |", "|---|" + "---|" * len(PARTS)]
    for h in hist:
        L.append(f"| {h['version']} | " + " | ".join("–" if not (s := h["parts"].get(k)) else f"{pc(s['ai_missed'])} / {pc(s['human_flagged'])}" for k, _ in PARTS) + " |")
    L += ["", "## AUROC", "", "| Version | " + " | ".join(n for _, n in PARTS) + " |", "|---|" + "---|" * len(PARTS)]
    for h in hist:
        L.append(f"| {h['version']} | " + " | ".join("–" if not (s := h["parts"].get(k)) or s["auroc"] is None else f"{s['auroc']:.3f}" for k, _ in PARTS) + " |")
    vs = [h["version"] for h in hist]
    L += ["", "## Held-out pools (original / after edits)", "",
          "AI pools show *AI detected*, real-photo pools *Human detected*; mixed pools show both as AI / Human.", "",
          "| Pool | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
    for pool in POOL_NAMES:
        cells = []
        for h in hist:
            p = h["pools"].get(pool)
            if not p: cells.append("–"); continue
            one = lambda s: (pc(s["ai_detected"]) if s["n_ai"] and not s["n_real"] else pc(s["human_detected"]) if s["n_real"] and not s["n_ai"] else pair(s))  # noqa: E731
            cells.append(f"{one(p['clean'])} → {one(p['edited'])}")
        L.append(f"| {POOL_NAMES[pool]} | " + " | ".join(cells) + " |")
    L += ["", "## Held-out by edit (AI detected / Human detected)", "", "| Edit | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
    for cond, name in COND_NAMES.items():
        L.append(f"| {name} | " + " | ".join(pair(h["conds"].get(cond)) for h in hist) + " |")
    notes = [f"* **{h['version']}**: {h['note']}" for h in hist if h.get("note")]
    if notes: L += ["", "## Versions", ""] + notes
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add"); s.add_argument("--version", required=True); s.add_argument("--date", required=True)
    s.add_argument("--scores", required=True); s.add_argument("--fusion", required=True); s.add_argument("--ntire", default="")
    s.add_argument("--note", default="")
    sub.add_parser("report")
    a = ap.parse_args()
    (add if a.cmd == "add" else report)(a)


if __name__ == "__main__":
    main()
