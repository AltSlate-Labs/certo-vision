"""Export small out-of-scope image sets (streamed, no full download) to root/<set>/*.png.

    python -m tools.make_ood --out ~/Downloads/data/ood --n 200
Sets: dtd (textures, far OOD), eurosat (satellite, far OOD), pets (cats/dogs breeds, near / in-scope shift).
"""
import argparse
import pathlib

from datasets import load_dataset

SETS = {
    "dtd": ("tanganke/dtd", "train"),
    "eurosat": ("tanganke/eurosat", "train"),
    "pets": ("timm/oxford-iiit-pet", "test"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    a = ap.parse_args()
    for name, (repo, split) in SETS.items():
        if name not in a.sets:
            continue
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(repo, split=split, streaming=True).shuffle(seed=0, buffer_size=2000)
        k = 0
        labels = []
        for ex in ds:
            img = ex.get("image")
            if img is None:
                continue
            img.convert("RGB").save(d / f"{k:05d}.png")
            labels.append(str(ex.get("label", "")))
            k += 1
            if k >= a.n:
                break
        (d / "labels.txt").write_text("\n".join(labels))
        print(name, k)


if __name__ == "__main__":
    main()
