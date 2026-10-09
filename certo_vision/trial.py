"""Trial runner: a recipe over a labelled image folder, five-seed metrics with a majority
baseline, plus abstain / out-of-scope evaluation of the few-shot-calibrated questions.

    certo-vision trial --recipe recipes/banana.json --images ~/data/banana --seeds 0 1 2 3 4
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .decider import Decider
from .eval import IMG_EXT, auroc, embed_files, eval_task, load_folder, mean_std, split
from .recipe import labelled, option_keys, question_spec


def run_trial(dec: Decider, recipe: dict, images, per_class=150, seeds=(0, 1, 2, 3, 4), cache=None,
              alpha=0.3, coverage=0.95) -> dict:
    name = recipe["name"]
    items = load_folder(images, per_class)
    files, folders = [f for f, _ in items], [c for _, c in items]
    S = embed_files(dec.emb, files, cache, "images")
    folder_counts = {c: folders.count(c) for c in sorted(set(folders))}
    ood_S = {}
    for oname, odir in recipe.get("ood", {}).items():
        odir = pathlib.Path(odir).expanduser()
        ofiles = sorted(f for f in odir.iterdir() if f.suffix.lower() in IMG_EXT)[:per_class]
        if ofiles:
            ood_S[oname] = embed_files(dec.emb, ofiles, cache, f"ood_{oname}")

    report = {"name": name, "n_images": len(files), "folders": folder_counts, "seeds": list(seeds), "questions": {}}
    for qid, qcfg in recipe["questions"].items():
        spec = question_spec(qcfg)
        if spec["type"] == "choice" and qcfg.get("labels", "folder") == "folder":
            spec["criteria"] = {k: v for k, v in spec["criteria"].items() if k in folder_counts}
        keys = option_keys(spec)
        idx, y_keys = labelled(items, qcfg, keys)
        Sq = S[np.array(idx)]
        ordinal = spec["type"] == "score"
        per_seed = {}
        for seed in seeds:
            per_seed[str(seed)] = eval_task(dec, qid, spec, Sq, y_keys, keys, ordinal=ordinal, k_shot_alpha=alpha, seed=seed)
        summary = {}
        for variant in ("zero_shot", "calibrated", "calibrated_fewshot"):
            summary[variant] = {m: mean_std([per_seed[s][variant][m] for s in per_seed])
                                for m, v in per_seed[str(seeds[0])][variant].items() if isinstance(v, (int, float))}
        summary["majority_acc"] = round(float(np.mean([per_seed[s]["majority_acc"] for s in per_seed])), 4)
        # abstain / out-of-scope on the seed-0 few-shot-calibrated question
        dec.questions.clear()
        cal, test = split(len(y_keys), 0.3, seeds[0])
        dec.calibrate(qid, spec, Sq[cal], [y_keys[i] for i in cal], few_shot_alpha=alpha, coverage=coverage)
        thr = dec.compile({qid: spec})[qid].abstain_threshold  # unrounded, as the decider uses it
        conf_in = dec.confidence_batch(qid, spec, Sq[test])
        scope = {"abstain_threshold": round(thr, 4), "in_scope_abstain_rate": round(float((conf_in < thr).mean()), 4), "ood": {}}
        for oname, So in ood_S.items():
            conf_o = dec.confidence_batch(qid, spec, So)
            maxp = dec.probs_batch(qid, spec, So).max(1)
            scope["ood"][oname] = {"n": int(len(So)), "auroc": round(auroc(conf_in, conf_o), 4),
                                   "rejected": round(float((conf_o < thr).mean()), 4),
                                   "answered_confidently": round(float(((conf_o >= thr) & (maxp > 0.9)).mean()), 4)}
        report["questions"][qid] = {"type": spec["type"], "n": int(len(idx)), "n_calib": int(len(cal)), "n_test": int(len(test)),
                                    "class_counts": {k: y_keys.count(k) for k in keys}, "summary": summary,
                                    "scope": scope, "per_seed": per_seed}
        print(summary_line(name, qid, report["questions"][qid]))
    return report


def summary_line(name, qid, r) -> str:
    s, sc = r["summary"], r["scope"]
    fs, cal, zs = s["calibrated_fewshot"], s["calibrated"], s["zero_shot"]
    line = (f"[{name}/{qid} {r['type']} n={r['n']}] acc zero {zs['acc']['mean']:.3f} cal {cal['acc']['mean']:.3f} "
            f"fewshot {fs['acc']['mean']:.3f}±{fs['acc']['std']:.3f} (majority {s['majority_acc']:.3f}) ece {fs['ece']['mean']:.3f}")
    if "mae" in fs:
        line += f" mae {fs['mae']['mean']:.2f} rho {fs['spearman']['mean']:.2f}"
    line += f" | abstain in-scope {sc['in_scope_abstain_rate']:.2f}"
    if sc["ood"]:
        line += "; OOD rejected " + ", ".join(f"{k} {v['rejected']:.2f}" for k, v in sc["ood"].items())
    return line


def write_report(report: dict, out) -> None:
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
