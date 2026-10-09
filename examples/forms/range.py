"""Range questions on the forms: whole image vs derived from per-box crop decisions.

    python examples/forms/range.py --data examples/forms/data --cache results/cache/forms_siglip.npz --model <siglip dir>
Appends a "range" block to examples/forms/results.json.
"""
import argparse
import json
import pathlib

import numpy as np

from certo_vision import metrics
from certo_vision.decider import Decider
from certo_vision.eval import eval_task, mean_std, split
from certo_vision.recipe import make_embedder

BINS = {"0-2499": (0, 2499), "2500-4999": (2500, 4999), "5000-7499": (5000, 7499), "7500-9999": (7500, 9999)}
DIGITS = [str(i) for i in range(10)]


def bin_of(n):
    return next(k for k, (lo, hi) in BINS.items() if lo <= n <= hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    labels = json.loads((a.data / "labels.json").read_text())
    z = np.load(a.cache); S, C = z["S"], z["C"]
    dec = Decider(make_embedder("siglip", a.model))
    nums = np.array([l["number"] for l in labels])
    y_bin = [bin_of(n) for n in nums]
    keys = list(BINS)

    # whole image: a 4-way choice over ranges, and the same as a score (ordered bins)
    spec_c = {"type": "choice", "instructions": "In which range is the four-digit handwritten number?",
              "criteria": {k: f"the number is between {lo} and {hi}" for k, (lo, hi) in BINS.items()}}
    per = {str(s): eval_task(dec, "range", spec_c, S, y_bin, keys, seed=s) for s in a.seeds}
    whole = {v: mean_std([per[s][v]["acc"] for s in per]) for v in ("zero_shot", "calibrated", "calibrated_fewshot")}
    whole["majority_acc"] = round(float(np.mean([per[s]["majority_acc"] for s in per])), 4)

    # crops: calibrate one digit question on calibration-form crops, read all four digits of test forms, derive the range
    crop_spec = {"type": "choice", "instructions": "What is the handwritten digit in this box?", "criteria": {d: f"the handwritten digit {d}" for d in DIGITS}}
    y_crop = [str(d) for l in labels for d in l["digits"]]
    acc_bin, acc_ge, acc_first = [], [], []
    for s in a.seeds:
        cal_f, test_f = split(len(labels), 0.3, s)
        cal = np.array([4 * f + j for f in cal_f for j in range(4)]); test = np.array([4 * f + j for f in test_f for j in range(4)])
        dec.questions.clear(); dec.calibrate("digit", crop_spec, C[cal], [y_crop[i] for i in cal], few_shot_alpha=0.3)
        pred = dec.probs_batch("digit", crop_spec, C[test]).argmax(1).reshape(-1, 4)
        n_hat = (pred * np.array([1000, 100, 10, 1])).sum(1)
        truth = nums[test_f]
        acc_bin.append(float(np.mean([bin_of(p) == bin_of(t) for p, t in zip(n_hat, truth)])))
        acc_ge.append(float(np.mean((n_hat >= 5000) == (truth >= 5000))))
        acc_first.append(float(np.mean(pred[:, 0] == truth // 1000)))
    out = {"bins": BINS, "whole_image_choice": whole, "whole_image_per_seed": per,
           "from_crops": {"range_4bin": mean_std(acc_bin), "ge_5000": mean_std(acc_ge), "first_digit": mean_std(acc_first)}}
    print(json.dumps({k: v for k, v in out.items() if k != "whole_image_per_seed"}, indent=1))
    rp = a.data.parent / "results.json"
    r = json.loads(rp.read_text()); r["range"] = out; rp.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
