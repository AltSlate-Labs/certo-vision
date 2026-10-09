import numpy as np


def accuracy(p, y):
    return float((p.argmax(1) == y).mean())


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def aurc(conf, correct):
    """Area under risk-coverage curve (lower = confidence ranks errors better)."""
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(risk.mean())


def score_mae_spearman(p, y):
    s = (p * np.arange(p.shape[1])).sum(1)
    mae = float(np.abs(s - y).mean())
    rs = np.argsort(np.argsort(s)); ry = np.argsort(np.argsort(y + 1e-9 * np.arange(len(y))))
    rho = float(np.corrcoef(rs, ry)[0, 1])
    return mae, rho


def report(p, y, conf=None, ordinal=False):
    r = {"acc": accuracy(p, y), "nll": nll(p, y), "brier": brier(p, y), "ece": ece(p, y)}
    if conf is not None:
        r["aurc"] = aurc(conf, p.argmax(1) == y)
    if ordinal:
        r["mae"], r["spearman"] = score_mae_spearman(p, y)
    return {k: round(v, 4) for k, v in r.items()}
