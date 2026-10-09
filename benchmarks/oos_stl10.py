"""Out-of-scope test: does the confidence signal separate in-scope test images from images
the calibration set never saw, and does the model emit confident answers on them?

    python -m benchmarks.oos_stl10 --backbone gemma --model <dir> --images data/stl10 \
        --classes-desc data/stl10/classes.json --per-class 100 --cache results/emb_gemma.npz \
        --ood data/ood --out results/oos_gemma.json
Needs the --cache npz written by benchmarks.stl10 for the same image folder (clean embeddings + labels).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

from certo_vision.decider import Decider
from certo_vision.embedder import l2n
from certo_vision.eval import split, load_folder, embed_all


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)); pos = in-scope, neg = out-of-scope."""
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) + 1
    # tie-aware ranks
    order = np.argsort(x)
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


def signals(dec: Decider, q, S: np.ndarray) -> dict[str, np.ndarray]:
    agree, ood, maxp, ptrue = [], [], [], []
    for s in S:
        o = dec._confidence(q, s); a = 1.0  # MRL agreement removed from the decider
        p = dec._probs(q, s)
        agree.append(a); ood.append(o); maxp.append(float(p.max())); ptrue.append(float(p[0]))
    return {"agree": np.array(agree), "nn_ratio": np.array(ood),
            "conf": np.array(agree) * np.array(ood), "maxprob": np.array(maxp), "p_first": np.array(ptrue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["gemma", "siglip"], default="gemma")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", type=pathlib.Path, required=True)
    ap.add_argument("--classes-desc", type=pathlib.Path, required=True)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--target", default="cat")
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--ood", type=pathlib.Path, required=True)
    ap.add_argument("--ood-cache", type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mrl-dim", type=int, default=768)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    if a.backbone == "siglip":
        from certo_vision.embedder import SiglipEmbedder
        emb = SiglipEmbedder(model_id=a.model)
        if a.mrl_dim == 768:
            a.mrl_dim = emb.dim
    else:
        from certo_vision.embedder import GemmaEmbedder
        emb = GemmaEmbedder(model_id=a.model)
    dec = Decider(emb, mrl_dim=a.mrl_dim)

    z = np.load(a.cache)
    S, labels = z["S"], list(z["labels"])
    classes = json.loads(a.classes_desc.read_text())
    items = load_folder(a.images, a.per_class)
    assert [c for _, c in items] == labels, "cache labels do not match folder"
    cal, test = split(len(labels), 0.3, a.seed)

    choice = {"type": "choice", "instructions": "What is the main subject of this photo?", "criteria": classes}
    noul = {"type": "noul", "instructions": f"Does this photo show {classes[a.target]}?",
            "criteria": {"true": f"yes, it shows {classes[a.target]}", "false": f"no, it shows something other than {a.target}"}}
    yn = ["true" if c == a.target else "false" for c in labels]
    dec.questions.clear()
    dec.calibrate("subject", choice, S[cal], [labels[i] for i in cal], few_shot_alpha=0.3)
    dec.calibrate("is_target", noul, S[cal], [yn[i] for i in cal], few_shot_alpha=0.3)
    qs = {q.qid: q for q in dec.questions.values()}

    # OOD sets
    ood_sets = {}
    cache = {}
    if a.ood_cache and a.ood_cache.exists():
        cache = dict(np.load(a.ood_cache))
    for d in sorted(p for p in a.ood.iterdir() if p.is_dir()):
        if d.name in cache:
            ood_sets[d.name] = cache[d.name]
            continue
        files = sorted(f for f in d.iterdir() if f.suffix == ".png")
        if not files:
            print(f"skipping empty OOD set {d.name}")
            continue
        imgs = [Image.open(f).convert("RGB") for f in files]
        ood_sets[d.name], _ = embed_all(emb, imgs, d.name)
        cache[d.name] = ood_sets[d.name]
    if a.ood_cache:
        a.ood_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.ood_cache, **cache)

    report = {"backbone": a.backbone, "model": a.model, "seed": a.seed, "n_in_test": int(len(test)),
              "ood_sets": {k: int(len(v)) for k, v in ood_sets.items()}, "questions": {}}
    for qid, q in qs.items():
        sig_in = signals(dec, q, S[test])
        thr = {k: float(np.quantile(v, 0.05)) for k, v in sig_in.items()}  # keep 95% of in-scope
        r = {"in_scope": {k: {"mean": round(float(v.mean()), 4), "p05": round(thr[k], 4)} for k, v in sig_in.items()},
             "in_scope_maxprob_gt_0.9": round(float((sig_in["maxprob"] > 0.9).mean()), 4),
             "ood": {}}
        for name, So in ood_sets.items():
            sig = signals(dec, q, So)
            r["ood"][name] = {
                "mean": {k: round(float(v.mean()), 4) for k, v in sig.items()},
                "auroc_in_vs_ood": {k: round(auroc(sig_in[k], sig[k]), 4) for k in sig},
                "rejected_at_95pct_in_coverage": {k: round(float((sig[k] < thr[k]).mean()), 4) for k in ("agree", "nn_ratio", "conf", "maxprob")},
                "maxprob_gt_0.9": round(float((sig["maxprob"] > 0.9).mean()), 4),
            }
            if q.type == "noul":
                r["ood"][name]["frac_predicted_true"] = round(float((sig["p_first"] > 0.5).mean()), 4)
            else:
                pred = [q.keys[int(dec._probs(q, s).argmax())] for s in So]
                top = {k: pred.count(k) for k in set(pred)}
                r["ood"][name]["top_predictions"] = dict(sorted(top.items(), key=lambda kv: -kv[1])[:4])
        report["questions"][qid] = r
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
