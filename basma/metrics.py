"""Threshold-free and fixed-FPR metrics (labels 1 = AI)."""
import numpy as np


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg]); _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks); ranks = sums[inv] / cnt[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_fpr(neg_scores, fpr):
    """Smallest threshold t such that mean(neg >= t) <= fpr (scores above t are called AI)."""
    neg = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if len(neg) == 0:
        return float("inf")
    k = int(np.floor(fpr * len(neg)))            # number of negatives allowed above the threshold
    return float(neg[len(neg) - k - 1] + 1e-9) if k < len(neg) else float(neg[0] - 1e-9)


def tpr_at_fpr(y, s, fpr):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    t = threshold_at_fpr(s[y == 0], fpr)
    return float((s[y == 1] > t).mean())
