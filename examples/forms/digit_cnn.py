"""A dedicated digit reader for the crops: a small CNN trained on MNIST (with augmentation), applied to the
box crops after MNIST-style normalisation; then the same CNN fine-tuned on the calibration forms' crops.

    HF_HOME=~/Downloads/data/mnist python examples/forms/digit_cnn.py --data examples/forms/data
Appends "digit_cnn" to examples/forms/results.json.
"""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from certo_vision.eval import mean_std, split


def to_mnist(im: Image.Image, box) -> np.ndarray:
    """Box interior -> 28x28 float array, white ink on black, centred by mass, like MNIST."""
    x0, y0, x1, y1 = box
    g = 255 - np.asarray(im.crop((x0 + 4, y0 + 4, x1 - 4, y1 - 4)).convert("L"), dtype=np.float32)
    g[g < 70] = 0
    ys, xs = np.nonzero(g)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    s = 20 / max(h, w)
    g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR), np.float32)
    out = np.zeros((28, 28), np.float32)
    h, w = g.shape
    out[(28 - h) // 2:(28 - h) // 2 + h, (28 - w) // 2:(28 - w) // 2 + w] = g
    cy, cx = np.array(np.nonzero(out)).mean(1) if out.any() else (14, 14)
    out = np.roll(out, (round(14 - cy), round(14 - cx)), axis=(0, 1))
    return out / 255.0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 3), nn.Conv2d(32, 64, 3)
        self.f1, self.f2 = nn.Linear(9216, 128), nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c2(F.relu(self.c1(x)))), 2)
        return self.f2(F.dropout(F.relu(self.f1(torch.flatten(x, 1))), 0.3, self.training))


def augment(x):
    """Random rotation / scale / shift, matching how the forms were generated."""
    n = x.shape[0]
    ang = (torch.rand(n, device=x.device) - 0.5) * 2 * np.deg2rad(12)
    sc = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.3
    tx, ty = (torch.rand(n, device=x.device) - 0.5) * 0.25, (torch.rand(n, device=x.device) - 0.5) * 0.25
    theta = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                         torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def train(net, X, Y, epochs, dev, lr=1e-3, bs=128, aug=True):
    opt = torch.optim.Adam(net.parameters(), lr)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            xb = X[idx]
            if aug:
                xb = augment(xb)
            loss = F.cross_entropy(net(xb), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()


@torch.no_grad()
def predict(net, X, bs=512):
    return torch.cat([net(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = json.loads((a.data / "labels.json").read_text())
    crops = np.stack([to_mnist(Image.open(a.data / l["file"]), b) for l in labels for b in l["boxes"]])
    y = np.array([d for l in labels for d in l["digits"]])
    Xc = torch.tensor(crops[:, None], device=dev); Yc = torch.tensor(y, device=dev)

    from datasets import load_dataset
    mn = load_dataset("ylecun/mnist", split="train")
    Xm = torch.tensor(np.stack([np.asarray(im, np.float32) / 255 for im in mn["image"]])[:, None], device=dev)
    Ym = torch.tensor(np.array(mn["label"]), device=dev)
    mt = load_dataset("ylecun/mnist", split="test")
    Xt = torch.tensor(np.stack([np.asarray(im, np.float32) / 255 for im in mt["image"]])[:, None], device=dev)
    Yt = np.array(mt["label"])

    out = {}
    zero, fine_d, fine_w, zero_w = [], [], [], []
    for s in a.seeds:
        torch.manual_seed(s)
        net = Net().to(dev)
        train(net, Xm, Ym, a.epochs, dev)
        mnist_acc = float((predict(net, Xt) == Yt).mean())
        cal_f, test_f = split(len(labels), 0.3, s)
        cal = np.array([4 * f + j for f in cal_f for j in range(4)]); test = np.array([4 * f + j for f in test_f for j in range(4)])
        p = predict(net, Xc[test])
        zero.append(float((p == y[test]).mean())); zero_w.append(float((p.reshape(-1, 4) == y[test].reshape(-1, 4)).all(1).mean()))
        train(net, Xc[cal], Yc[cal], 15, dev, lr=3e-4)       # fine-tune on the calibration forms' crops
        p = predict(net, Xc[test])
        fine_d.append(float((p == y[test]).mean())); fine_w.append(float((p.reshape(-1, 4) == y[test].reshape(-1, 4)).all(1).mean()))
        print(f"seed {s}: mnist test {mnist_acc:.4f}  crops: mnist-only {zero[-1]:.3f} (whole {zero_w[-1]:.3f})  fine-tuned {fine_d[-1]:.3f} (whole {fine_w[-1]:.3f})")
    out = {"mnist_only": {"digit_acc": mean_std(zero), "whole_number_acc": mean_std(zero_w)},
           "finetuned_on_calib_crops": {"digit_acc": mean_std(fine_d), "whole_number_acc": mean_std(fine_w)},
           "params": sum(p.numel() for p in Net().parameters()), "epochs_mnist": a.epochs}
    print(json.dumps(out, indent=1))
    rp = a.data.parent / "results.json"
    r = json.loads(rp.read_text()); r["digit_cnn"] = out; rp.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
