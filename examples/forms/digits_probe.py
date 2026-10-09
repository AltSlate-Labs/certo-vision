"""Heads on frozen SigLIP crop features: prototype (the decider), ridge probe, k-NN. Same form-level splits.

    python examples/forms/digits_probe.py --data examples/forms/data --cache results/cache/forms_siglip.npz
"""
import argparse
import json
import pathlib

import numpy as np

from certo_vision.eval import mean_std, split


def ridge_fit(X, y, lam, k=10):
    Y = np.eye(k)[y]
    mu = X.mean(0); Xc = X - mu
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (Y - Y.mean(0)))
    return mu, W, Y.mean(0)


def ridge_pred(X, m):
    mu, W, b = m
    return ((X - mu) @ W + b).argmax(1)


def knn_pred(Xtr, ytr, Xte, k):
    sims = Xte @ Xtr.T
    nn = np.argsort(-sims, axis=1)[:, :k]
    votes = np.zeros((len(Xte), 10))
    for j in range(k):
        votes[np.arange(len(Xte)), ytr[nn[:, j]]] += sims[np.arange(len(Xte)), nn[:, j]]
    return votes.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    labels = json.loads((a.data / "labels.json").read_text())
    C = np.load(a.cache)["C"]
    y = np.array([d for l in labels for d in l["digits"]])
    out = {}
    for name in ("ridge", "knn"):
        digit, whole, chosen = [], [], []
        for s in a.seeds:
            cal_f, test_f = split(len(labels), 0.3, s)
            cal = np.array([4 * f + j for f in cal_f for j in range(4)]); test = np.array([4 * f + j for f in test_f for j in range(4)])
            # pick the hyper-parameter on an inner split of the calibration forms
            inner_tr, inner_va = split(len(cal_f), 0.7, s + 100)
            itr = np.array([4 * f + j for f in cal_f[inner_tr] for j in range(4)]); iva = np.array([4 * f + j for f in cal_f[inner_va] for j in range(4)])
            grid = [0.01, 0.1, 1, 10, 100] if name == "ridge" else [1, 3, 5, 10, 20]
            def acc(h, tr, va):
                p = ridge_pred(C[va], ridge_fit(C[tr], y[tr], h)) if name == "ridge" else knn_pred(C[tr], y[tr], C[va], h)
                return (p == y[va]).mean()
            h = max(grid, key=lambda h: acc(h, itr, iva))
            p = ridge_pred(C[test], ridge_fit(C[cal], y[cal], h)) if name == "ridge" else knn_pred(C[cal], y[cal], C[test], h)
            digit.append(float((p == y[test]).mean())); whole.append(float((p.reshape(-1, 4) == y[test].reshape(-1, 4)).all(1).mean())); chosen.append(h)
        out[name] = {"digit_acc": mean_std(digit), "whole_number_acc": mean_std(whole), "hyper": chosen}
        print(f"{name:6s} digit {out[name]['digit_acc']['mean']:.3f}±{out[name]['digit_acc']['std']:.3f} whole {out[name]['whole_number_acc']['mean']:.3f} hyper {chosen}")
    rp = a.data.parent / "results.json"
    r = json.loads(rp.read_text()); r.setdefault("digits_head", {}).update(out); rp.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
