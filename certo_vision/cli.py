"""certo-vision command line: calibrate, decide, serve, trial."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

from . import metrics
from .decider import Decider
from .eval import embed_files, load_folder, split
from .recipe import labelled, load_recipe, make_embedder, option_keys, question_spec, specs

MIN_PER_OPTION = 10


def _backbone_args(ap):
    ap.add_argument("--backbone", choices=["siglip", "gemma", "stub"], default="siglip")
    ap.add_argument("--model", help="HF id or local model dir (default: the backbone's public checkpoint)")
    ap.add_argument("--device")


def cmd_calibrate(a):
    """Fit every question in the recipe on a labelled folder and save one calibration file."""
    recipe = load_recipe(a.recipe)
    emb = make_embedder(a.backbone, a.model, a.device)
    items = load_folder(a.images, a.per_class)
    if not items:
        sys.exit(f"no images under {a.images}/<label>/")
    S = embed_files(emb, [f for f, _ in items], a.cache)
    dec = Decider(emb)
    out = {"recipe": str(a.recipe), "images": str(a.images), "n_images": len(items), "questions": {}}
    for qid, qcfg in recipe["questions"].items():
        spec = question_spec(qcfg)
        keys = option_keys(spec)
        idx, y = labelled(items, qcfg, keys)
        counts = {k: y.count(k) for k in keys}
        r = {"type": spec["type"], "n": len(idx), "per_option": counts}
        thin = [k for k, c in counts.items() if c < MIN_PER_OPTION]
        if thin:
            r["warning"] = f"fewer than {MIN_PER_OPTION} images for {thin}; few-shot prototypes and abstain threshold will be noisy"
        Sq = S[np.array(idx)]
        if a.holdout > 0:  # held-out check with a throwaway decider, then refit on everything
            cal, test = split(len(y), 1 - a.holdout, a.seed)
            chk = Decider(emb)
            chk.calibrate(qid, spec, Sq[cal], [y[i] for i in cal], few_shot_alpha=a.alpha, coverage=a.coverage)
            yt = np.array([keys.index(y[i]) for i in test])
            p = chk.probs_batch(qid, spec, Sq[test])
            conf = chk.confidence_batch(qid, spec, Sq[test])
            thr = next(iter(chk.questions.values())).abstain_threshold
            r["holdout"] = metrics.report(p, yt, ordinal=spec["type"] == "score") | {
                "n_fit": int(len(cal)), "n_test": int(len(test)),
                "majority_acc": round(float(np.bincount(yt, minlength=len(keys)).max() / len(test)), 4),
                "abstain_rate": round(float((conf < thr).mean()), 4)}
        r["fit"] = dec.calibrate(qid, spec, Sq, y, few_shot_alpha=a.alpha, coverage=a.coverage)
        out["questions"][qid] = r
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dec.save(a.out)
    out["saved"] = str(a.out)
    print(json.dumps(out, indent=1))


def cmd_decide(a):
    """Answer the recipe's questions for one or more images, as JSON lines."""
    recipe = load_recipe(a.recipe)
    dec = Decider(make_embedder(a.backbone, a.model, a.device))
    if a.calib:
        dec.load(a.calib)
    qs = {q: s for q, s in specs(recipe).items() if not a.question or q in a.question}
    for path in a.images:
        r = dec.decide({"image": Image.open(path).convert("RGB")}, qs)
        print(json.dumps({"image": str(path)} | r))


def cmd_serve(a):
    import uvicorn
    from .server import create_app
    uvicorn.run(create_app(a.backbone, a.model, a.calib, a.recipe, a.device), host=a.host, port=a.port)


def cmd_trial(a):
    from .trial import run_trial, write_report
    recipe = load_recipe(a.recipe)
    dec = Decider(make_embedder(a.backbone, a.model, a.device))
    cache = a.cache or pathlib.Path("results/cache") / f"{recipe['name']}_{a.backbone}.npz"
    report = run_trial(dec, recipe, a.images, a.per_class, a.seeds, cache, a.alpha, a.coverage)
    report |= {"backbone": a.backbone, "model": a.model}
    write_report(report, a.out or f"results/trial_{recipe['name']}_{a.backbone}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="certo-vision")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calibrate", help="fit a recipe on a labelled image folder and save the calibration")
    _backbone_args(p)
    p.add_argument("--recipe", required=True)
    p.add_argument("--images", required=True, help="root/<label>/*.png")
    p.add_argument("--out", required=True, help="calibration .npz")
    p.add_argument("--per-class", type=int)
    p.add_argument("--alpha", type=float, default=0.3, help="few-shot prototype blend")
    p.add_argument("--coverage", type=float, default=0.95, help="in-scope coverage of the abstain gate")
    p.add_argument("--holdout", type=float, default=0.3, help="fraction held out for the sanity check (0 = skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", help="npz to cache image embeddings")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("decide", help="answer a recipe's questions for image files")
    _backbone_args(p)
    p.add_argument("--recipe", required=True)
    p.add_argument("--calib", help="calibration .npz (omit for zero-shot)")
    p.add_argument("--question", nargs="*", help="subset of question ids")
    p.add_argument("images", nargs="+")
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("serve", help="run the HTTP endpoint")
    _backbone_args(p)
    p.add_argument("--calib")
    p.add_argument("--recipe")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("trial", help="multi-seed evaluation of a recipe on a labelled folder")
    _backbone_args(p)
    p.add_argument("--recipe", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--per-class", type=int, default=150)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--cache")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_trial)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
