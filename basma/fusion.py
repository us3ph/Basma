"""Fuse the view logits into one calibrated probability, set thresholds at a fixed FPR on the validation reals,
and report every pool / generator family / condition at those thresholds (plus baselines on the same pools).

  python -m basma.fusion --scores runs/x/scores/scores.jsonl [--baseline name=path.jsonl ...] --out runs/x/report
Outputs: fusion.json (weights, isotonic table, thresholds; ported to JS), report.json, and the rows for RESULTS.md."""
import argparse
import collections
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .metrics import auroc, threshold_at_fpr

FPR_AI = 0.01       # "AI" verdict: at most 1 % of validation reals and 1 % of calibration reals (all conditions pooled) above the threshold
FNR_REAL = 0.02     # "Real" verdict: at most 2 % of validation AI images below the lower threshold


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def features(r):
    lc = r["lc"] if r["lc"] else [r["lg"]]
    return [r["lg"], float(np.mean(lc)), float(np.max(lc)), float(np.min(lc)), len(r["lc"]) / 3.0, np.log(max(1, r["h"] * r["w"])) / 12.0]


FEATURE_NAMES = ["global", "crop_mean", "crop_max", "crop_min", "n_crops/3", "log_area/12"]


class Fusion:
    def __init__(self, coef, intercept, iso_x, iso_y, thr_ai, thr_real):
        self.coef, self.intercept, self.iso_x, self.iso_y, self.thr_ai, self.thr_real = coef, intercept, iso_x, iso_y, thr_ai, thr_real

    def raw(self, r):
        return float(np.dot(self.coef, features(r)) + self.intercept)

    def prob(self, r):
        return float(np.interp(self.raw(r), self.iso_x, self.iso_y))

    def verdict(self, p):
        return "AI" if p >= self.thr_ai else ("Real" if p <= self.thr_real else "Uncertain")

    def to_json(self):
        return {"features": FEATURE_NAMES, "coef": [float(c) for c in self.coef], "intercept": float(self.intercept),
                "iso_x": [float(v) for v in self.iso_x], "iso_y": [float(v) for v in self.iso_y], "thr_ai": self.thr_ai, "thr_real": self.thr_real}


def fit(val_rows):
    X = np.array([features(r) for r in val_rows]); y = np.array([r["label"] for r in val_rows])
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)
    raw = lr.decision_function(X)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw, y)
    xs = np.linspace(raw.min() - 1, raw.max() + 1, 257)
    ys = iso.predict(xs)
    f = Fusion(lr.coef_[0], lr.intercept_[0], xs, ys, 1.0, 0.0)
    p = np.array([f.prob(r) for r in val_rows])
    f.thr_ai = threshold_at_fpr(p[y == 0], FPR_AI)
    ai_sorted = np.sort(p[y == 1]); k = int(np.floor(FNR_REAL * len(ai_sorted)))
    f.thr_real = float(ai_sorted[k] - 1e-9) if len(ai_sorted) else 0.0
    f.thr_real = min(f.thr_real, f.thr_ai)
    return f


def thr_rule(rows, score_of):
    """AI threshold, the same rule for every system: at most FPR_AI of the validation reals (train sources) AND at most
    FPR_AI of the calibration reals (sources never trained on) score at or above it."""
    ts = []
    for sp in ("val", "calib"):
        neg = [score_of(r) for r in rows if r["split"] == sp and r["label"] == 0]
        if neg:
            ts.append(threshold_at_fpr(neg, FPR_AI))
    return max(ts)


def summarize(rows, score_of, thr, group_keys):
    """rows -> {group: {n, auroc, tpr or fpr at thr}} for each grouping; AI rows report TPR, real rows FPR."""
    out = {}
    for gk in group_keys:
        g = collections.defaultdict(list)
        for r in rows:
            g[str(r[gk])].append(r)
        res = {}
        for name, rs in sorted(g.items()):
            s = np.array([score_of(r) for r in rs]); y = np.array([r["label"] for r in rs])
            d = {"n": len(rs), "n_ai": int(y.sum()), "n_real": int((1 - y).sum())}
            if y.sum() and (1 - y).sum():
                d["auroc"] = round(auroc(y, s), 4)
            if y.sum():
                d["tpr"] = round(float((s[y == 1] >= thr).mean()), 4)
            if (1 - y).sum():
                d["fpr"] = round(float((s[y == 0] >= thr).mean()), 4)
            res[name] = d
        out[gk] = res
    return out


def report_system(name, rows, score_of, thr_fn):
    """thr_fn(rows) -> AI threshold (thr_rule). Same protocol for Basma and each baseline."""
    val = [r for r in rows if r["split"] == "val"]
    cal = [r for r in rows if r["split"] == "calib"]
    thr = thr_fn(rows)
    ev = [r for r in rows if r["split"].startswith("eval:")]
    rep = {"system": name, "threshold": thr, "n_val": len(val), "n_calib": len(cal), "n_eval": len(ev)}
    rep["val_by_cond"] = summarize(val, score_of, thr, ["cond"])["cond"]
    if cal:
        rep["calib_by_source"] = summarize(cal, score_of, thr, ["source"])["source"]
    rep["eval_by_split"] = summarize(ev, score_of, thr, ["split"])["split"]
    rep["eval_by_cond"] = summarize(ev, score_of, thr, ["cond"])["cond"]
    rep["eval_by_family"] = summarize([r for r in ev if r["label"] == 1], score_of, thr, ["family"])["family"]
    rep["eval_real_by_source"] = summarize([r for r in ev if r["label"] == 0], score_of, thr, ["source"])["source"]
    rep["eval_split_x_cond"] = {}
    for sp in sorted({r["split"] for r in ev}):
        rep["eval_split_x_cond"][sp] = summarize([r for r in ev if r["split"] == sp], score_of, thr, ["cond"])["cond"]
    s_all = np.array([score_of(r) for r in ev]); y_all = np.array([r["label"] for r in ev])
    rep["eval_overall"] = {"auroc": round(auroc(y_all, s_all), 4), "tpr": round(float((s_all[y_all == 1] >= thr).mean()), 4) if y_all.sum() else None,
                           "fpr": round(float((s_all[y_all == 0] >= thr).mean()), 4) if (1 - y_all).sum() else None}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline", nargs="*", default=[], help="name=path to a baseline scores.jsonl (same ids/conds, field 'score')")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_rows(a.scores)
    val = [r for r in rows if r["split"] == "val"]
    f = fit(val)
    f.thr_ai = thr_rule(rows, f.prob)
    f.thr_real = min(f.thr_real, f.thr_ai)
    json.dump(f.to_json(), open(os.path.join(a.out, "fusion.json"), "w"), indent=1)
    report = {"basma": report_system("basma", rows, f.prob, lambda rs: f.thr_ai), "verdict_thresholds": {"thr_ai": f.thr_ai, "thr_real": f.thr_real},
              "fusion": f.to_json(), "baselines": {}}
    # verdict distribution on eval pools
    ev = [r for r in rows if r["split"].startswith("eval:")]
    vc = collections.Counter((r["label"], f.verdict(f.prob(r))) for r in ev)
    report["basma"]["verdicts"] = {f"{'ai' if l else 'real'}:{v}": n for (l, v), n in sorted(vc.items())}
    for spec in a.baseline:
        name, path = spec.split("=", 1)
        brows = load_rows(path)
        report["baselines"][name] = report_system(name, brows, lambda r: r["score"], lambda rs: thr_rule(rs, lambda r: r["score"]))
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps({"basma_eval": report["basma"]["eval_overall"], "by_split": {k: {kk: v[kk] for kk in ("tpr", "fpr", "auroc") if kk in v} for k, v in report["basma"]["eval_by_split"].items()},
                      "baselines": {n: b["eval_overall"] for n, b in report["baselines"].items()}}, indent=1))


if __name__ == "__main__":
    main()
