"""Figures and markdown tables for the report, from results/*.json.

    python report/make_figures.py        # writes report/figures/*.png and report/tables.md
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TRIALS = [("mvtec_bottle", "defective", "bottle: defective?"), ("mvtec_hazelnut", "defective", "hazelnut: defective?"),
          ("mvtec_carpet", "defective", "carpet: defective?"), ("mvtec_screw", "defective", "screw: defective?"),
          ("mvtec_hazelnut", "defect_type", "hazelnut: defect type (4)"), ("mvtec_screw", "defect_type", "screw: defect type (5)"),
          ("banana", "ripeness", "banana: ripeness (score, 4)"), ("banana", "overripe", "banana: overripe?"),
          ("banana", "stage", "banana: stage (4)")]


def trial_rows():
    rows = []
    for name, qid, label in TRIALS:
        r = json.loads((RES / f"trial_{name}_siglip.json").read_text())["questions"][qid]
        s, sc = r["summary"], r["scope"]
        rows.append({"label": label, "type": r["type"], "n": r["n"], "n_calib": r["n_calib"],
                     "min_per_option": min(r["class_counts"].values()),
                     "zero": s["zero_shot"]["acc"]["mean"], "cal": s["calibrated"]["acc"]["mean"],
                     "few": s["calibrated_fewshot"]["acc"]["mean"], "few_std": s["calibrated_fewshot"]["acc"]["std"],
                     "maj": s["majority_acc"], "ece_zero": s["zero_shot"]["ece"]["mean"], "ece_few": s["calibrated_fewshot"]["ece"]["mean"],
                     "nll_zero": s["zero_shot"]["nll"]["mean"], "nll_few": s["calibrated_fewshot"]["nll"]["mean"],
                     "mae": s["calibrated_fewshot"].get("mae", {}).get("mean"), "rho": s["calibrated_fewshot"].get("spearman", {}).get("mean"),
                     "abstain_in": sc["in_scope_abstain_rate"],
                     "ood_rej": min(v["rejected"] for v in sc["ood"].values()), "ood_n": sum(v["n"] for v in sc["ood"].values()),
                     "ood_auroc": min(v["auroc"] for v in sc["ood"].values())})
    return rows


def stl10_rows():
    rows = []
    for bb, lab in (("siglip", "SigLIP 2 so400m"), ("gemma", "EmbeddingGemma 2")):
        d = json.loads((RES / f"stl10_{bb}_5seed.json").read_text())
        for task, tl in (("choice", "subject (10-way)"), ("noul", "is it a cat?"), ("score", "blur level (5, synthetic)")):
            s = d["summary"][task]
            rows.append({"backbone": lab, "task": tl, "zero": s["zero_shot"]["acc"]["mean"], "cal": s["calibrated"]["acc"]["mean"],
                         "few": s["calibrated_fewshot"]["acc"]["mean"], "few_std": s["calibrated_fewshot"]["acc"]["std"], "maj": s["majority_acc"],
                         "ece_few": s["calibrated_fewshot"]["ece"]["mean"], "embed_ms": d["embed_ms_p50"]})
    return rows


def oos_rows():
    rows = []
    for bb, lab in (("siglip", "SigLIP 2"), ("gemma", "EmbeddingGemma 2")):
        d = json.loads((RES / f"oos_{bb}.json").read_text())
        for qid, ql in (("subject", "choice"), ("is_target", "noul")):
            for oset, ol in (("dtd", "textures (DTD)"), ("eurosat", "satellite (EuroSAT)"), ("pets", "Oxford pets")):
                o = d["questions"][qid]["ood"][oset]
                rows.append({"backbone": lab, "q": ql, "set": ol, **{f"auroc_{k}": o["auroc_in_vs_ood"][k] for k in ("nn_ratio", "agree", "maxprob")},
                             "rej_nn": o["rejected_at_95pct_in_coverage"]["nn_ratio"], "maxp": o["maxprob_gt_0.9"]})
    return rows


def fig_results(rows):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(rows))[::-1]
    h = 0.26
    ax.barh(y + h, [r["zero"] for r in rows], h, label="zero-shot (text prototypes only)", color="#9FB0C6")
    ax.barh(y, [r["few"] for r in rows], h, xerr=[r["few_std"] for r in rows], label="calibrated + few-shot (30% of images)", color="#2F6F8F", error_kw={"lw": 1})
    ax.barh(y - h, [r["maj"] for r in rows], h, label="majority class", color="#D9772B")
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("accuracy on the held-out 70% (mean over 5 seeds)")
    ax.axvline(0.9, color="#888", lw=0.8, ls="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "results.png", dpi=150)


def fig_calibration(rows):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(rows))[::-1]
    h = 0.38
    ax.barh(y + h / 2, [r["ece_zero"] for r in rows], h, label="zero-shot", color="#9FB0C6")
    ax.barh(y - h / 2, [r["ece_few"] for r in rows], h, label="calibrated + few-shot", color="#2F6F8F")
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlabel("expected calibration error (lower is better)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)


def fig_scope(rows):
    sig = [("auroc_nn_ratio", "nearest-neighbour ratio (the gate)"), ("auroc_maxprob", "max probability")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, q in zip(axes, ("choice", "noul")):
        sub = [r for r in rows if r["backbone"] == "SigLIP 2" and r["q"] == q]
        x = np.arange(len(sub))
        for i, (k, lab) in enumerate(sig):
            ax.bar(x + (i - 0.5) * 0.36, [r[k] for r in sub], 0.36, label=lab, color=["#2F6F8F", "#D9772B"][i])
        ax.set_xticks(x, [r["set"] for r in sub], fontsize=8)
        ax.set_ylim(0.3, 1.02)
        ax.axhline(0.5, color="#888", lw=0.8, ls="--")
        ax.set_title(f"{q} question", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC in-scope vs out-of-scope")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "scope.png", dpi=150)


def tables(tr, st, oo):
    L = ["## Trial table", "", "| Question | type | n | ≈ calib images per option (smallest) | zero-shot | calibrated | few-shot | majority | ECE few-shot | NLL zero, few-shot | MAE / ρ | in-scope abstain | OOD rejected (n) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tr:
        mae = f"{r['mae']:.2f} / {r['rho']:.2f}" if r["mae"] is not None else ""
        L.append(f"| {r['label']} | {r['type']} | {r['n']} | {max(1, round(r['min_per_option'] * r['n_calib'] / r['n']))} | {r['zero']:.3f} | {r['cal']:.3f} | **{r['few']:.3f}** ± {r['few_std']:.3f} | {r['maj']:.3f} | {r['ece_few']:.3f} | {r['nll_zero']:.2f}, {r['nll_few']:.2f} | {mae} | {r['abstain_in']:.2f} | {r['ood_rej']:.0%} ({r['ood_n']}) |")
    L += ["", "## STL-10 table", "", "| Backbone | task | zero-shot | calibrated | few-shot | majority | ECE few-shot | embed ms (GB10) |", "|---|---|---|---|---|---|---|---|"]
    for r in st:
        L.append(f"| {r['backbone']} | {r['task']} | {r['zero']:.3f} | {r['cal']:.3f} | **{r['few']:.3f}** ± {r['few_std']:.3f} | {r['maj']:.3f} | {r['ece_few']:.3f} | {r['embed_ms']:.0f} |")
    L += ["", "## Scope table", "", "| Backbone | question | out-of-scope set | AUROC NN ratio | AUROC max-prob | AUROC MRL agreement | rejected by NN gate | max-prob > 0.9 |", "|---|---|---|---|---|---|---|---|"]
    for r in oo:
        L.append(f"| {r['backbone']} | {r['q']} | {r['set']} | {r['auroc_nn_ratio']:.3f} | {r['auroc_maxprob']:.3f} | {r['auroc_agree']:.3f} | {r['rej_nn']:.0%} | {r['maxp']:.0%} |")
    (ROOT / "report" / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    tr, st, oo = trial_rows(), stl10_rows(), oos_rows()
    fig_results(tr); fig_calibration(tr); fig_scope(oo); tables(tr, st, oo)
    print(open(ROOT / "report" / "tables.md").read())
