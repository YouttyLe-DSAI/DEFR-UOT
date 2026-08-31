"""UAR / WAR / AUROC without sklearn, so the analysis runs anywhere."""

import numpy as np


def uar_war(y_true, y_pred, n_classes=None):
    """Unweighted (mean per-class recall) and weighted (overall) accuracy, in %."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0:
        return float('nan'), float('nan'), np.array([])

    classes = range(n_classes) if n_classes else sorted(set(y_true.tolist()))
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum():
            recalls.append(100.0 * (y_pred[mask] == c).mean())
    war = 100.0 * (y_pred == y_true).mean()
    return float(np.mean(recalls)), float(war), np.array(recalls)


def _average_ranks(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average tied ranks, so a constant score gives AUROC 0.5 rather than 1.0
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(scores, positive):
    """P(score of a positive > score of a negative), ties counted as half."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(positive).astype(bool).ravel()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fpr_at_tpr(scores, positive, tpr_target=0.95):
    """False positive rate at the threshold that first reaches tpr_target."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(positive).astype(bool).ravel()
    if positive.sum() == 0 or (~positive).sum() == 0:
        return float('nan')
    thr = np.quantile(scores[positive], 1.0 - tpr_target)
    return float((scores[~positive] >= thr).mean())
