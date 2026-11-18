"""Metric helpers: AUROC/AUPRC, calibration error, thresholds, patient-level bootstrap."""
import numpy as np

from .config import LABELS


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def per_class_auroc(scores, y, names=LABELS):
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, n in enumerate(names):
        if 0 < y[:, j].sum() < len(y):
            out[n] = float(roc_auc_score(y[:, j], scores[:, j]))
    return out


def mean_auroc(scores, y):
    a = per_class_auroc(scores, y)
    return float(np.mean(list(a.values()))) if a else float("nan"), a


def per_class_auprc(scores, y, names=LABELS):
    from sklearn.metrics import average_precision_score
    return {n: float(average_precision_score(y[:, j], scores[:, j]))
            for j, n in enumerate(names) if y[:, j].sum() > 0}


def ece(p, y, bins=15):
    """Expected calibration error (equal-width bins) on flattened probabilities."""
    p, y = np.asarray(p).ravel(), np.asarray(y).ravel()
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def classwise_ece(p, y, bins=15):
    return float(np.mean([ece(p[:, j], y[:, j], bins) for j in range(p.shape[1])]))


def reliability_curve(p, y, bins=10):
    p, y = np.asarray(p).ravel(), np.asarray(y).ravel()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() > 0:
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def f1_threshold(p, y):
    """Threshold maximising F1 for one class."""
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P = ys.sum()
    if P == 0:
        return 1.0
    f1 = 2 * tp / (tp + fp + P)
    k = int(np.argmax(f1))
    return float(ps[k])


def precision_threshold(p, y, target, min_support=20):
    """Lowest threshold t such that precision(p >= t) >= target with at least `min_support` predicted positives.
    Returns 1.01 (never assert) if the target is unattainable."""
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    n = np.arange(1, len(ys) + 1)
    prec = tp / n
    ok = np.where((prec >= target) & (n >= min_support))[0]
    if len(ok) == 0:
        return 1.01
    return float(ps[ok.max()])


def binary_stats(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, specificity=spec, f1=f1)


def patient_bootstrap(stat_fn, groups, n_boot=500, seed=0):
    """Cluster bootstrap over patients (images of one patient are resampled together).
    stat_fn(indices) -> float. Returns (2.5, 97.5) percentiles."""
    rng = np.random.RandomState(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [[] for _ in range(len(uniq))]
    for i, g in enumerate(inv):
        members[g].append(i)
    members = [np.array(m) for m in members]
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        try:
            v = stat_fn(idx)
            if np.isfinite(v):
                vals.append(v)
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
