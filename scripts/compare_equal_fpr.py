"""Ranking quality at an equal false-positive rate on the held-out real photos (diagnostic).

Each system's threshold is set on the held-out reals themselves, so every system flags exactly the same share of
unseen real photos and the comparison is about ranking alone, not about where a threshold happened to land. This is an
oracle setting, never used for the shipped thresholds (those come from validation + calibration reals, see fusion.py).

  python scripts/compare_equal_fpr.py --scores runs/x/scores/scores.jsonl --fusion runs/x/report/fusion.json \
      --baseline thermostatic=runs/x/baselines/thermostatic.jsonl ... > docs/EQUAL_FPR.md"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basma.fusion import Fusion, load_rows  # noqa: E402

FPRS = (0.01, 0.02, 0.05)


def thr_at(neg, fpr):
    v = np.sort(np.asarray(neg)); return v[int(np.ceil((1 - fpr) * len(v))) - 1]


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--fusion", required=True)
    ap.add_argument("--baseline", nargs="*", default=[])
    a = ap.parse_args()
    fu = json.load(open(a.fusion)); F = Fusion(**{k: fu[k] for k in ("coef", "intercept", "iso_x", "iso_y", "thr_ai", "thr_real")})
    rows = load_rows(a.scores); key = lambda r: (r["id"], r["cond"])
    meta = {key(r): r for r in rows if r["split"].startswith("eval:")}
    systems = {"Basma": {k: F.raw(r) for k, r in meta.items()}}
    for spec in a.baseline:
        name, path = spec.split("=", 1)
        systems[name] = {key(r): r["score"] for r in load_rows(path) if key(r) in meta}
    keys = [k for k in meta if all(k in s for s in systems.values())]
    neg = [k for k in keys if meta[k]["label"] == 0]; pos = [k for k in keys if meta[k]["label"] == 1]
    md = ["# Equal false-positive comparison (held-out pools)", "",
          f"Every system's threshold is set on the {len(neg)} held-out real rows themselves (image x condition), so all systems flag the same "
          f"share of unseen real photos and only ranking differs. Diagnostic only: shipped thresholds come from validation and calibration reals. "
          f"AI rows: {len(pos)}.", "", "## AI images caught (TPR %) at each held-out false-positive rate", ""]
    thr = {n: {f: thr_at([s[k] for k in neg], f) for f in FPRS} for n, s in systems.items()}
    md += [table(["system", *[f"FPR {f * 100:g} %" for f in FPRS]], [[n] + [f"{np.mean([s[k] > thr[n][f] for k in pos]) * 100:.1f}" for f in FPRS] for n, s in systems.items()]), ""]
    pool = lambda r: r["split"] if r["split"] != "eval:unseen_vendors" else f"unseen:{r['family']}"
    md += ["## Per AI pool at 2 % held-out FPR (TPR %)", ""]
    rows_ = []
    for p in sorted({pool(meta[k]) for k in pos}):
        ks = [k for k in pos if pool(meta[k]) == p]
        rows_.append([p, len(ks)] + [f"{np.mean([s[k] > thr[n][0.02] for k in ks]) * 100:.1f}" for n, s in systems.items()])
    md += [table(["pool", "rows", *systems], rows_), "", "## Per condition at 2 % held-out FPR (TPR %)", ""]
    rows_ = []
    for c in sorted({meta[k]["cond"] for k in pos}):
        ks = [k for k in pos if meta[k]["cond"] == c]
        rows_.append([c] + [f"{np.mean([s[k] > thr[n][0.02] for k in ks]) * 100:.1f}" for n, s in systems.items()])
    md += [table(["condition", *systems], rows_), ""]
    print("\n".join(md))


if __name__ == "__main__":
    main()
