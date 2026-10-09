"""Figure for the forms example: six sample forms and the whole-image vs per-crop accuracy chart."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
labels = json.loads((HERE / "data/labels.json").read_text())
res = json.loads((HERE / "results.json").read_text())

fig, axes = plt.subplots(1, 6, figsize=(13, 1.6))
for ax, l in zip(axes, labels[:6]):
    ax.imshow(Image.open(HERE / "data" / l["file"]).crop((0, 80, 384, 260)), cmap="gray"); ax.axis("off")
    ax.set_title(f"{l['field']}  {l['number']:04d}", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "samples.png", dpi=130)

rows = [("field name (6-way), whole image", res["whole_image"]["field"]),
        *[(f"digit {k} (10-way), whole image", res["whole_image"][f"digit_{k}"]) for k in range(1, 5)],
        ("number ≥ 5000 (yes/no), whole image", res["whole_image"]["number_ge_5000"]),
        ("digit (10-way), per-box crop", res["crops"]["digit"])]
fig, ax = plt.subplots(figsize=(9, 3.8))
y = list(range(len(rows)))[::-1]; h = 0.27
ax.barh([v + h for v in y], [r["summary"]["zero_shot"]["acc"]["mean"] for _, r in rows], h, color="#9FB0C6", label="zero-shot")
ax.barh(y, [r["summary"]["calibrated_fewshot"]["acc"]["mean"] for _, r in rows], h, color="#2F6F8F", label="calibrated + few-shot",
        xerr=[r["summary"]["calibrated_fewshot"]["acc"]["std"] for _, r in rows], error_kw={"lw": 1})
ax.barh([v - h for v in y], [r["summary"]["majority_acc"] for _, r in rows], h, color="#D9772B", label="majority")
ax.set_yticks(y, [n for n, _ in rows]); ax.set_xlim(0, 1); ax.set_xlabel("accuracy, held-out 70%, mean of 5 seeds")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "results.png", dpi=150)
print("wrote samples.png results.png")
