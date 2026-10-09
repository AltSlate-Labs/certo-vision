"""README figures: the pipeline diagram and the demo strip (sample images with the model's decisions).

    python report/make_readme_figures.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
NAVY, BLUE, ORANGE, GREY, PAPER = "#14213D", "#2F6F8F", "#D9772B", "#9FB0C6", "#FBFBF8"


def box(ax, x, y, w, h, text, fc, tc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.4, shrinkA=2, shrinkB=2))


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
    # serving path (top)
    box(ax, 0.2, 2.6, 1.5, 1.0, "image", GREY, NAVY)
    box(ax, 2.2, 2.6, 2.3, 1.0, "SigLIP 2 encoder\nfrozen, one pass\n~72 ms on a GB10", NAVY)
    box(ax, 5.0, 2.6, 1.6, 1.0, "state vector\n1152-d", GREY, NAVY)
    box(ax, 7.1, 2.9, 1.9, 0.75, "dot products\n+ softmax", BLUE)
    box(ax, 7.1, 1.95, 1.9, 0.75, "nearest-neighbour\nscope gate", BLUE)
    box(ax, 9.5, 2.3, 1.4, 1.4, "answers\nnoul / choice / score\nprobabilities\nconfidence\nabstain", NAVY, fs=8.5)
    arrow(ax, 1.7, 3.1, 2.2, 3.1); arrow(ax, 4.5, 3.1, 5.0, 3.1)
    arrow(ax, 6.6, 3.1, 7.1, 3.27); arrow(ax, 6.6, 3.1, 7.1, 2.33)
    arrow(ax, 9.0, 3.27, 9.5, 3.2); arrow(ax, 9.0, 2.33, 9.5, 2.7)
    # calibration path (bottom)
    box(ax, 0.2, 0.3, 2.0, 1.1, "recipe\nquestions, options,\ncriteria text", GREY, NAVY)
    box(ax, 2.6, 0.3, 2.3, 1.1, "30 to 50 labelled images\nper option,\nfrom the deployment camera", GREY, NAVY, fs=8.5)
    box(ax, 5.3, 0.3, 3.3, 1.1, "calibrate once (seconds, numpy)\ntemperature + per-option bias (NLL)\ntext prototypes blended with image means\ngate threshold at 95% in-scope coverage", ORANGE, fs=8)
    box(ax, 9.1, 0.3, 1.8, 1.1, "calib.npz\nprototypes, fit,\ncalibration states", GREY, NAVY, fs=8.5)
    arrow(ax, 2.2, 0.85, 2.6, 0.85); arrow(ax, 4.9, 0.85, 5.3, 0.85); arrow(ax, 8.6, 0.85, 9.1, 0.85)
    arrow(ax, 8.05, 1.4, 8.05, 1.95)
    ax.text(0.2, 3.85, "serve: every question answered from the same embedding", color=NAVY, fontsize=9, weight="bold")
    ax.text(0.2, 1.6, "calibrate: per deployment, no training", color=NAVY, fontsize=9, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "pipeline.png", dpi=150, facecolor=PAPER)


def demo_strip():
    rows = [json.loads(l) for l in (ROOT / "demo/banana/decisions.jsonl").read_text().splitlines()]
    order = ["green", "semi_ripe", "ripe", "overripe", "not_a_banana"]
    rows.sort(key=lambda r: order.index(pathlib.Path(r["image"]).stem))
    fig, axes = plt.subplots(1, len(rows), figsize=(2.3 * len(rows), 3.6))
    fig.patch.set_facecolor(PAPER)
    for ax, r in zip(axes, rows):
        a = r["answers"]
        ax.imshow(Image.open(ROOT / r["image"]).convert("RGB")); ax.axis("off")
        st, sc, ov = a["stage"], a["ripeness"], a["overripe"]
        if st["abstain"]:
            txt = f"abstain\nconfidence {st['confidence']:.2f} < 0.97\n(would have said {st['choice']})"
            col = ORANGE
        else:
            txt = f"stage {st['choice']} p={st['probabilities'][st['choice']]:.2f}\nripeness {sc['score']:.2f} / 3\noverripe p={ov['noul']:.2f}\nconfidence {st['confidence']:.2f}"
            col = NAVY
        ax.set_title(pathlib.Path(r["image"]).stem.replace("_", " "), fontsize=9, color=NAVY)
        ax.text(0.5, -0.06, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=col, linespacing=1.5)
    fig.subplots_adjust(bottom=0.34, top=0.9, wspace=0.08)
    fig.savefig(FIG / "demo.png", dpi=150, facecolor=PAPER, bbox_inches="tight")


if __name__ == "__main__":
    pipeline(); demo_strip(); print("wrote pipeline.png demo.png")
