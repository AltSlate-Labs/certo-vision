"""Export an STL-10 subset to root/<class>/*.png plus classes.json (class -> description).

    python -m tools.make_stl10 --out ~/Downloads/data/stl10 --per-class 100
"""
import argparse
import json
import pathlib

from datasets import load_dataset

DESC = {
    "airplane": "an airplane, an aircraft in the sky or on a runway",
    "bird": "a bird",
    "car": "a car, an automobile on a road",
    "cat": "a domestic cat",
    "deer": "a deer in a field or forest",
    "dog": "a pet dog",
    "horse": "a horse",
    "monkey": "a monkey or ape",
    "ship": "a ship or boat on water",
    "truck": "a truck or lorry",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    ds = load_dataset("tanganke/stl10", split=a.split)
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
    (a.out / "classes.json").write_text(json.dumps({n: DESC.get(n, n) for n in names}, indent=1))
    print(counts)


if __name__ == "__main__":
    main()
