"""Export MVTec AD objects to root/<object>/<defect>/*.png ("good" = defect-free).

    python -m tools.make_mvtec --out ~/Downloads/data/mvtec --objects bottle screw hazelnut carpet --good-train 100
Uses the TheoM55/mvtec_all_objects_split mirror: config "default", splits "<object>.train" / "<object>.test".
"""
import argparse
import collections
import pathlib

from datasets import load_dataset

REPO = "TheoM55/mvtec_all_objects_split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--objects", nargs="+", default=["bottle", "screw", "hazelnut", "carpet"])
    ap.add_argument("--good-train", type=int, default=100, help="defect-free train images to keep per object")
    a = ap.parse_args()
    for obj in a.objects:
        counts = collections.Counter()
        for split in ("train", "test"):
            ds = load_dataset(REPO, split=f"{obj}.{split}")
            for i, ex in enumerate(ds):
                defect = ex["defect"] or "good"
                if split == "train" and counts["good"] >= a.good_train:
                    continue
                d = a.out / obj / defect
                d.mkdir(parents=True, exist_ok=True)
                ex["image_path"].convert("RGB").save(d / f"{split}_{i:04d}.png")
                counts[defect] += 1
        print(obj, dict(counts))


if __name__ == "__main__":
    main()
