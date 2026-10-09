"""Synthetic form snippets: a printed English field name followed by four boxes, each holding one
handwritten MNIST digit (so the number is 0000..9999). Labels per image go to labels.json.

    python examples/forms/make_forms.py --out examples/forms/data --n 600
MNIST comes from the Hugging Face Hub (ylecun/mnist); set HF_HOME / HF_DATASETS_CACHE to control where it lands.
"""
import argparse
import json
import pathlib
import random

from PIL import Image, ImageDraw, ImageFont

FIELDS = ["item", "quantity", "price", "weight", "count", "total"]
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/System/Library/Fonts/Supplemental", "/Library/Fonts"]
FONT_NAMES = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSansMono.ttf",
              "Arial.ttf", "Georgia.ttf", "Courier New.ttf", "Verdana.ttf", "Times New Roman.ttf"]
SIZE, BOX, GAP = 384, 50, 8


def fonts():
    out = []
    for d in FONT_DIRS:
        for n in FONT_NAMES:
            p = pathlib.Path(d) / n
            if p.exists():
                out.append(str(p))
    if not out:
        raise SystemExit("no TrueType fonts found; add a path to FONT_DIRS")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    mn = load_dataset("ylecun/mnist", split="train")
    by_digit = {d: [] for d in range(10)}
    for i, lab in enumerate(mn["label"]):
        by_digit[lab].append(i)
    rng = random.Random(a.seed)
    fl = fonts()
    a.out.mkdir(parents=True, exist_ok=True)
    labels = []
    for k in range(a.n):
        field = FIELDS[k % len(FIELDS)]
        digits = [rng.randrange(10) for _ in range(4)]
        im = Image.new("L", (SIZE, SIZE), 255)
        d = ImageDraw.Draw(im)
        fp, size = rng.choice(fl), rng.randint(26, 36)
        while True:  # shrink the font until the word and the four boxes fit side by side
            font = ImageFont.truetype(fp, size)
            tw = d.textlength(field + ":", font=font)
            if 14 + tw + 12 + 4 * BOX + 3 * GAP <= SIZE - 10 or size <= 16:
                break
            size -= 2
        y_row = rng.randint(120, 200)
        d.text((14, y_row + rng.randint(-4, 4)), field + ":", fill=0, font=font, anchor="lm")
        x = 14 + tw + rng.randint(12, 24)
        x = min(x, SIZE - 4 * BOX - 3 * GAP - 10)
        boxes = []
        for j in range(4):
            x0, y0 = int(x + j * (BOX + GAP)), y_row - BOX // 2
            d.rectangle([x0, y0, x0 + BOX, y0 + BOX], outline=0, width=2)
            g = mn[rng.choice(by_digit[digits[j]])]["image"].convert("L")
            s = rng.randint(36, 46)
            g = g.resize((s, s), Image.BILINEAR).rotate(rng.uniform(-8, 8), fillcolor=0)
            g = Image.eval(g, lambda v: 255 - v)              # MNIST is white-on-black
            ox, oy = x0 + (BOX - s) // 2 + rng.randint(-3, 3), y0 + (BOX - s) // 2 + rng.randint(-3, 3)
            im.paste(g, (ox, oy), mask=Image.eval(g, lambda v: 255 - v))
            boxes.append([x0, y0, x0 + BOX, y0 + BOX])
        # mild scanner-style noise
        px = im.load()
        for _ in range(rng.randint(100, 400)):
            px[rng.randrange(SIZE), rng.randrange(SIZE)] = rng.randint(120, 230)
        name = f"{k:04d}.png"
        im.convert("RGB").save(a.out / name)
        labels.append({"file": name, "field": field, "digits": digits, "number": int("".join(map(str, digits))), "boxes": boxes})
    (a.out / "labels.json").write_text(json.dumps(labels, indent=0))
    print(f"wrote {a.n} forms to {a.out}")


if __name__ == "__main__":
    main()
