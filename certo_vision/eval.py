"""Shared evaluation plumbing: folder loading, embedding, calibration/test splits and the
zero-shot vs calibrated vs few-shot comparison used by the trial runner and the benchmarks."""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
from PIL import Image

from . import metrics
from .embedder import l2n

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_folder(root, per_class: int | None = None):
    """root/<label>/*.png -> [(path, label)], sorted, at most per_class per folder."""
    root = pathlib.Path(root).expanduser()
    items = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in cdir.iterdir() if f.suffix.lower() in IMG_EXT)
        items += [(f, cdir.name) for f in files[:per_class]]
    return items


def embed_all(emb, images, desc=""):
    vecs, times = [], []
    for i, im in enumerate(images):
        t = time.perf_counter()
        vecs.append(emb.embed_state(image=im))
        times.append((time.perf_counter() - t) * 1e3)
        if (i + 1) % 100 == 0:
            print(f"  {desc}: {i + 1}/{len(images)}", file=sys.stderr)
    return l2n(np.stack(vecs)), np.array(times)


def embed_files(emb, files, cache=None, tag="images"):
    """Embed image files, reusing a cache npz (keyed by tag) when it matches in length."""
    cache = pathlib.Path(cache) if cache else None
    z = dict(np.load(cache)) if cache and cache.exists() else {}
    if tag in z and len(z[tag]) == len(files):
        return z[tag]
    S, _ = embed_all(emb, [Image.open(f).convert("RGB") for f in files], tag)
    if cache:
        z[tag] = S
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **z)
    return S


def split(n, frac, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def eval_task(dec, qid, spec, S, y_keys, keys, ordinal=False, calib_frac=0.3, k_shot_alpha=0.3, seed=0):
    """One calibration/test split: zero-shot, calibrated, calibrated + few-shot, majority baseline."""
    y = np.array([keys.index(k) for k in y_keys])
    cal, test = split(len(y), calib_frac, seed)
    out = {}
    dec.questions.clear()
    out["zero_shot"] = metrics.report(dec.probs_batch(qid, spec, S[test]), y[test], ordinal=ordinal)
    dec.questions.clear()
    fit = dec.calibrate(qid, spec, S[cal], [keys[i] for i in y[cal]])
    p = dec.probs_batch(qid, spec, S[test])
    conf = dec.confidence_batch(qid, spec, S[test])
    out["calibrated"] = metrics.report(p, y[test], conf=conf, ordinal=ordinal) | {"fit": fit}
    dec.questions.clear()
    dec.calibrate(qid, spec, S[cal], [keys[i] for i in y[cal]], few_shot_alpha=k_shot_alpha)
    p = dec.probs_batch(qid, spec, S[test])
    out["calibrated_fewshot"] = metrics.report(p, y[test], ordinal=ordinal)
    out["n_calib"], out["n_test"] = int(len(cal)), int(len(test))
    out["majority_acc"] = round(float(np.bincount(y[test]).max() / len(test)), 4)
    return out


def mean_std(vals):
    v = np.array(vals, dtype=float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)) with tie-aware ranks; pos = in-scope, neg = out-of-scope."""
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
