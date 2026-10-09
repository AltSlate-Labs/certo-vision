"""Benchmark Jev-style image decisions on EmbeddingGemma 2 (or a SigLIP 2 baseline).

Real run (needs model weights + an image folder laid out as root/<class>/*.jpg):
    python -m benchmarks.stl10 --images data/stl10 --classes-desc data/stl10/classes.json --model <dir>
    python -m benchmarks.stl10 ... --backbone siglip --model <siglip dir>
    python -m benchmarks.stl10 ... --seeds 0 1 2 3 4 --cache results/emb_gemma.npz
Pipeline smoke test without weights:
    python -m benchmarks.stl10 --stub

Three tasks, all built from one image folder:
  choice : which class is this?                  (folder labels)
  noul   : is this a <target class>? (one-vs-rest)
  score  : how blurry is it? 0..4                (synthetic Gaussian blur => exact ordinal labels)
For each: zero-shot vs temperature-calibrated vs calibrated+few-shot, plus latency.
With several --seeds the calibration/test split is redrawn per seed and mean/std reported.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from PIL import Image, ImageFilter

from certo_vision.decider import Decider
from certo_vision.embedder import StubEmbedder, l2n
from certo_vision import metrics

BLUR_LEVELS = [
    "Level 0: perfectly sharp, crisp fine detail",
    "Level 1: very slightly soft",
    "Level 2: noticeably blurry, fine detail lost",
    "Level 3: heavily blurred, only shapes visible",
    "Level 4: extremely blurred, an unrecognisable smear of colour",
]
BLUR_RADII = [0, 1.5, 3.5, 7, 14]


def load_folder(root: pathlib.Path, per_class: int):
    items = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted([f for f in cdir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        items += [(f, cdir.name) for f in files[:per_class]]
    return items


def stub_dataset(classes: dict, per_class: int):
    items = []
    for c, desc in classes.items():
        for _ in range(per_class):
            im = Image.new("RGB", (64, 64))
            im.info["concept"] = f"{c.replace('_', ' ')}: {desc}"
            items.append((im, c))
    return items


def embed_all(emb, images, desc):
    vecs, times = [], []
    for i, im in enumerate(images):
        t = time.perf_counter()
        vecs.append(emb.embed_state(image=im))
        times.append((time.perf_counter() - t) * 1e3)
        if (i + 1) % 100 == 0:
            print(f"  {desc}: {i + 1}/{len(images)}")
    return l2n(np.stack(vecs)), np.array(times)


def split(n, frac, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def eval_task(dec, qid, spec, S, y_keys, keys, ordinal=False, calib_frac=0.3, k_shot_alpha=0.3, seed=0):
    y = np.array([keys.index(k) for k in y_keys])
    cal, test = split(len(y), calib_frac, seed)
    out = {}
    # zero-shot (fresh compile each variant)
    dec.questions.clear()
    out["zero_shot"] = metrics.report(dec.probs_batch(qid, spec, S[test]), y[test], ordinal=ordinal)
    dec.questions.clear()
    fit = dec.calibrate(qid, spec, S[cal], [keys[i] for i in y[cal]])
    p = dec.probs_batch(qid, spec, S[test])
    q = next(iter(dec.questions.values()))
    conf = np.array([dec._confidence(q, s, None) for s in S[test]])
    out["calibrated"] = metrics.report(p, y[test], conf=conf, ordinal=ordinal) | {"fit": fit}
    dec.questions.clear()
    dec.calibrate(qid, spec, S[cal], [keys[i] for i in y[cal]], few_shot_alpha=k_shot_alpha)
    p = dec.probs_batch(qid, spec, S[test])
    out["calibrated_fewshot"] = metrics.report(p, y[test], ordinal=ordinal)
    out["n_calib"], out["n_test"] = int(len(cal)), int(len(test))
    out["majority_acc"] = round(float(np.bincount(y[test]).max() / len(test)), 4)
    return out


def latency_vs_questions(dec, image, n_list=(1, 10, 50)):
    res = {}
    for n in n_list:
        qs = {f"q{i}": {"type": "noul", "instructions": f"Does the image show concept number {i}?"} for i in range(n)}
        dec.compile(qs)  # warm prototype cache: schemas are static in production
        ts = []
        for _ in range(5):
            r = dec.decide({"image": image}, qs)
            ts.append(r["timing_ms"]["embed"] + r["timing_ms"]["decide"])
        res[n] = round(float(np.median(ts)), 2)
    return res


def summarise(per_seed: dict) -> dict:
    """mean/std over seeds for every numeric metric of every task/variant."""
    out = {}
    seeds = list(per_seed)
    for task in ("choice", "noul", "score"):
        out[task] = {}
        for variant in ("zero_shot", "calibrated", "calibrated_fewshot"):
            keys = [k for k, v in per_seed[seeds[0]][task][variant].items() if isinstance(v, (int, float))]
            out[task][variant] = {}
            for k in keys:
                vals = np.array([per_seed[s][task][variant][k] for s in seeds], dtype=float)
                out[task][variant][k] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)}
        out[task]["majority_acc"] = round(float(np.mean([per_seed[s][task]["majority_acc"] for s in seeds])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=pathlib.Path)
    ap.add_argument("--classes-desc", type=pathlib.Path, help="json {class_folder: description}")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--target", help="class used for the noul task (default: first)")
    ap.add_argument("--mrl-dim", type=int, default=768)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--backbone", choices=["gemma", "siglip"], default="gemma")
    ap.add_argument("--model", default=None, help="HF id or local model dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cache", type=pathlib.Path, help="npz file to cache embeddings (clean + blur)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results.json"))
    a = ap.parse_args()

    if a.stub:
        classes = {"cat": "a small domestic cat", "dog": "a pet dog", "car": "an automobile on a road",
                   "ship": "a ship on water", "airplane": "an aircraft in the sky"}
        emb = StubEmbedder(noise=3.0)
        items = stub_dataset(classes, a.per_class)
        images = [im for im, _ in items]
    else:
        if a.backbone == "siglip":
            from certo_vision.embedder import SiglipEmbedder
            emb = SiglipEmbedder(model_id=a.model or "google/siglip2-so400m-patch14-384")
            if a.mrl_dim == 768:
                a.mrl_dim = emb.dim
        else:
            from certo_vision.embedder import GemmaEmbedder
            emb = GemmaEmbedder(model_id=a.model or "google/embeddinggemma-2")
        items = load_folder(a.images, a.per_class)
        classes = json.loads(a.classes_desc.read_text()) if a.classes_desc else {c: c for _, c in items}
        images = [Image.open(p).convert("RGB") for p, _ in items]
    labels = [c for _, c in items]
    dec = Decider(emb, mrl_dim=a.mrl_dim)
    results = {"n_images": len(images), "mrl_dim": a.mrl_dim, "backbone": a.backbone, "model": a.model, "seeds": a.seeds}

    # blur images (deterministic)
    if a.stub:
        blur_imgs, blur_y = [], []
        for i, lvl in enumerate(BLUR_LEVELS):
            for _ in range(a.per_class):
                im = Image.new("RGB", (64, 64)); im.info["concept"] = lvl; blur_imgs.append(im); blur_y.append(str(i))
    else:
        rng = np.random.default_rng(1)
        blur_imgs, blur_y = [], []
        for im in images:
            lvl = int(rng.integers(0, len(BLUR_LEVELS)))
            blur_imgs.append(im.filter(ImageFilter.GaussianBlur(BLUR_RADII[lvl])) if lvl else im)
            blur_y.append(str(lvl))

    if a.cache and a.cache.exists():
        z = np.load(a.cache)
        S, Sb, t_embed = z["S"], z["Sb"], z["t_embed"]
        assert len(S) == len(images), "cache does not match image set"
        print(f"loaded embeddings from {a.cache}")
    else:
        print("Embedding images (one pass per image)...")
        S, t_embed = embed_all(emb, images, "clean")
        Sb, _ = embed_all(emb, blur_imgs, "blur")
        if a.cache:
            a.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.cache, S=S, Sb=Sb, t_embed=t_embed, labels=np.array(labels), blur_y=np.array(blur_y))
    results["embed_ms_p50"], results["embed_ms_p95"] = float(np.median(t_embed)), float(np.percentile(t_embed, 95))

    choice = {"type": "choice", "instructions": "What is the main subject of this photo?", "criteria": classes}
    tgt = a.target or list(classes)[0]
    noul = {"type": "noul", "instructions": f"Does this photo show {classes[tgt]}?",
            "criteria": {"true": f"yes, it shows {classes[tgt]}", "false": f"no, it shows something other than {tgt}"}}
    yn = ["true" if c == tgt else "false" for c in labels]
    score = {"type": "score", "instructions": "How blurry is this image?", "criteria": BLUR_LEVELS}
    score_keys = [str(i) for i in range(len(BLUR_LEVELS))]

    per_seed = {}
    for seed in a.seeds:
        r = {}
        r["choice"] = eval_task(dec, "subject", choice, S, labels, list(classes), seed=seed)
        r["noul"] = eval_task(dec, "is_target", noul, S, yn, ["true", "false"], seed=seed)
        r["score"] = eval_task(dec, "blur", score, Sb, blur_y, score_keys, ordinal=True, seed=seed)
        per_seed[str(seed)] = r
        print(f"seed {seed}: choice {r['choice']['calibrated_fewshot']['acc']:.3f} noul {r['noul']['calibrated_fewshot']['acc']:.3f} score {r['score']['calibrated_fewshot']['acc']:.3f}")
    results["per_seed"] = per_seed
    results["summary"] = summarise(per_seed)
    # keep seed-0 at top level for backward compatibility
    results.update(per_seed[str(a.seeds[0])])

    # MRL ablation on the choice task (seed 0)
    results["mrl_ablation_choice"] = {}
    for d in (128, 256, 512, 768):
        if d > S.shape[1]:
            continue
        dd = Decider(emb, mrl_dim=d)
        results["mrl_ablation_choice"][d] = eval_task(dd, "subject", choice, S, labels, list(classes), seed=a.seeds[0])["calibrated"]

    results["latency_ms_vs_num_questions"] = latency_vs_questions(dec, images[0])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["summary"], indent=1))


if __name__ == "__main__":
    main()
