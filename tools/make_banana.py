"""Export BananaImageBD ripeness classes to root/<class>/*.png.

    python -m tools.make_banana --out ~/Downloads/data/banana --per-class 150
"""
import argparse
import pathlib

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--per-class", type=int, default=150)
    a = ap.parse_args()
    ds = load_dataset("Project-AgML/BananaImageBD_ripeness_classification", "raw", split="train").shuffle(seed=0)
    names = ds.features["label"].names
    counts = {n: 0 for n in names}
    for i, ex in enumerate(ds):
        c = names[ex["label"]]
        if counts[c] >= a.per_class:
            if all(v >= a.per_class for v in counts.values()):
                break
            continue
        d = a.out / c
        d.mkdir(parents=True, exist_ok=True)
        ex["image"].convert("RGB").save(d / f"{i:05d}.png")
        counts[c] += 1
    print(counts)


if __name__ == "__main__":
    main()
