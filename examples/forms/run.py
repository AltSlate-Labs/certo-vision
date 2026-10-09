"""Forms example: can a pooled-embedding decider read a printed field name and four handwritten digits?

Two readings of each form are evaluated with the standard protocol (30/70 split, five seeds, majority baseline):
  whole image  -> field name (choice, 6), digit 1..4 (choice, 10 each), number >= 5000 (noul)
  per-box crop -> digit (choice, 10) on the cropped box, plus the whole-number accuracy when the four
                  crop decisions of a form are combined.

    python examples/forms/run.py --data examples/forms/data --model <siglip dir> --out examples/forms/results.json
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image

from certo_vision.decider import Decider
from certo_vision.eval import embed_all, eval_task, mean_std, split
from certo_vision.recipe import make_embedder

FIELDS = ["item", "quantity", "price", "weight", "count", "total"]
DIGITS = [str(i) for i in range(10)]
MARGIN = 6


def summarise(per_seed):
    out = {}
    for variant in ("zero_shot", "calibrated", "calibrated_fewshot"):
        out[variant] = {m: mean_std([per_seed[s][variant][m] for s in per_seed])
                        for m, v in per_seed[next(iter(per_seed))][variant].items() if isinstance(v, (int, float))}
    out["majority_acc"] = round(float(np.mean([per_seed[s]["majority_acc"] for s in per_seed])), 4)
    return out


def run(dec, qid, spec, S, y, keys, seeds):
    per_seed = {str(s): eval_task(dec, qid, spec, S, y, keys, seed=s) for s in seeds}
    r = {"n": int(len(y)), "summary": summarise(per_seed), "per_seed": per_seed}
    fs = r["summary"]["calibrated_fewshot"]
    print(f"[{qid} n={len(y)}] zero {r['summary']['zero_shot']['acc']['mean']:.3f} cal {r['summary']['calibrated']['acc']['mean']:.3f} "
          f"fewshot {fs['acc']['mean']:.3f}±{fs['acc']['std']:.3f} (majority {r['summary']['majority_acc']:.3f}) ece {fs['ece']['mean']:.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--backbone", default="siglip")
    ap.add_argument("--model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cache", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    labels = json.loads((a.data / "labels.json").read_text())
    emb = make_embedder(a.backbone, a.model)
    dec = Decider(emb)

    if a.cache and a.cache.exists():
        z = np.load(a.cache)
        S, C = z["S"], z["C"]
    else:
        forms = [Image.open(a.data / l["file"]).convert("RGB") for l in labels]
        S, _ = embed_all(emb, forms, "forms")
        crops = [im.crop((b[0] - MARGIN, b[1] - MARGIN, b[2] + MARGIN, b[3] + MARGIN)) for im, l in zip(forms, labels) for b in l["boxes"]]
        C, _ = embed_all(emb, crops, "crops")
        if a.cache:
            a.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.cache, S=S, C=C)

    field_spec = {"type": "choice", "instructions": "Which field name is printed on this form snippet?",
                  "criteria": {f: f"the word '{f}' printed before the boxes" for f in FIELDS}}
    digit_crit = {d: f"the handwritten digit {d}" for d in DIGITS}
    res = {"n_forms": len(labels), "backbone": a.backbone, "model": a.model, "seeds": a.seeds, "whole_image": {}, "crops": {}}

    res["whole_image"]["field"] = run(dec, "field", field_spec, S, [l["field"] for l in labels], FIELDS, a.seeds)
    for k in range(4):
        spec = {"type": "choice", "instructions": f"What is the handwritten digit in box {k + 1} of 4?", "criteria": digit_crit}
        res["whole_image"][f"digit_{k + 1}"] = run(dec, f"digit_{k + 1}", spec, S, [str(l["digits"][k]) for l in labels], DIGITS, a.seeds)
    spec = {"type": "noul", "instructions": "Is the four-digit handwritten number 5000 or more?",
            "criteria": {"true": "yes, the first digit is 5, 6, 7, 8 or 9", "false": "no, the first digit is 0, 1, 2, 3 or 4"}}
    res["whole_image"]["number_ge_5000"] = run(dec, "number_ge_5000", spec, S, ["true" if l["number"] >= 5000 else "false" for l in labels], ["true", "false"], a.seeds)

    # per-box crops: split by form so no form contributes to both calibration and test
    crop_spec = {"type": "choice", "instructions": "What is the handwritten digit in this box?", "criteria": digit_crit}
    y_crop = [str(d) for l in labels for d in l["digits"]]
    per_seed, whole = {}, []
    for s in a.seeds:
        cal_f, test_f = split(len(labels), 0.3, s)
        cal = np.array([4 * f + j for f in cal_f for j in range(4)]); test = np.array([4 * f + j for f in test_f for j in range(4)])
        dec.questions.clear()
        r = {}
        from certo_vision import metrics
        y = np.array([DIGITS.index(k) for k in y_crop])
        r["zero_shot"] = metrics.report(dec.probs_batch("digit", crop_spec, C[test]), y[test])
        dec.questions.clear(); dec.calibrate("digit", crop_spec, C[cal], [y_crop[i] for i in cal])
        r["calibrated"] = metrics.report(dec.probs_batch("digit", crop_spec, C[test]), y[test])
        dec.questions.clear(); dec.calibrate("digit", crop_spec, C[cal], [y_crop[i] for i in cal], few_shot_alpha=0.3)
        p = dec.probs_batch("digit", crop_spec, C[test])
        r["calibrated_fewshot"] = metrics.report(p, y[test])
        r["majority_acc"] = round(float(np.bincount(y[test]).max() / len(test)), 4)
        pred = p.argmax(1).reshape(-1, 4); truth = y[test].reshape(-1, 4)
        whole.append(float((pred == truth).all(1).mean()))
        per_seed[str(s)] = r
    res["crops"]["digit"] = {"n": int(len(y_crop)), "summary": summarise(per_seed), "per_seed": per_seed}
    res["crops"]["whole_number_fewshot"] = mean_std(whole)
    fs = res["crops"]["digit"]["summary"]["calibrated_fewshot"]
    print(f"[crops/digit n={len(y_crop)}] zero {res['crops']['digit']['summary']['zero_shot']['acc']['mean']:.3f} fewshot {fs['acc']['mean']:.3f}±{fs['acc']['std']:.3f} "
          f"(majority {res['crops']['digit']['summary']['majority_acc']:.3f}) | whole 4-digit number correct {res['crops']['whole_number_fewshot']['mean']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
