"""Pipeline tests on the deterministic stub embedder: no model weights needed."""
import json
import pathlib

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from certo_vision import Decider, StubEmbedder
from certo_vision.cli import main

CLASSES = {"Green": "unripe, fully green banana", "Ripe": "fully yellow banana", "Overripe": "yellow with brown spots"}
RECIPE = {"name": "toy", "questions": {
    "stage": {"type": "choice", "instructions": "Which ripeness stage?", "criteria": CLASSES, "labels": "folder"},
    "overripe": {"type": "noul", "instructions": "Is the banana overripe?",
                 "criteria": {"true": "yes, brown spots", "false": "no, green or yellow"},
                 "labels": {"Overripe": "true", "*": "false"}},
    "ripeness": {"type": "score", "instructions": "How ripe?", "criteria": list(CLASSES.values()),
                 "labels": {"Green": "0", "Ripe": "1", "Overripe": "2"}}}}


def stub_image(concept):
    im = Image.new("RGB", (8, 8))
    im.info["concept"] = concept
    return im


def write_folder(root: pathlib.Path, per_class=40):
    for c, desc in CLASSES.items():
        (root / c).mkdir(parents=True)
        for i in range(per_class):
            meta = PngImagePlugin.PngInfo()
            meta.add_text("concept", f"{c}: {desc}")
            Image.new("RGB", (8, 8)).save(root / c / f"{i:03d}.png", pnginfo=meta)


def test_calibrate_save_load_roundtrip(tmp_path):
    emb = StubEmbedder(noise=1.0)
    dec = Decider(emb)
    spec = {"type": "choice", "instructions": "Which stage?", "criteria": CLASSES}
    S = np.stack([emb.embed_state(stub_image(f"{c}: {d}")) for c, d in CLASSES.items() for _ in range(30)])
    y = [c for c in CLASSES for _ in range(30)]
    fit = dec.calibrate("stage", spec, S, y, few_shot_alpha=0.3)
    assert 0 < fit["abstain_threshold"] <= 1
    dec.save(tmp_path / "c.npz")
    dec2 = Decider(StubEmbedder(noise=1.0))
    assert dec2.load(tmp_path / "c.npz") == ["stage"]
    img = stub_image("Ripe: fully yellow banana")
    a1 = dec.decide({"image": img}, {"stage": spec})["answers"]["stage"]
    a2 = dec2.decide({"image": img}, {"stage": spec})["answers"]["stage"]
    assert a1 == a2 and a1["calibrated"] and a1["choice"] == "Ripe" and not a1["abstain"]
    ood = dec2.decide({"image": stub_image("a satellite photo of farmland")}, {"stage": spec})["answers"]["stage"]
    assert ood["abstain"] and ood["confidence"] < a1["confidence"]


def test_cli_calibrate_decide(tmp_path, capsys):
    root = tmp_path / "imgs"
    write_folder(root)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(RECIPE))
    calib = tmp_path / "calib.npz"
    main(["calibrate", "--backbone", "stub", "--recipe", str(recipe), "--images", str(root), "--out", str(calib)])
    rep = json.loads(capsys.readouterr().out)
    assert set(rep["questions"]) == {"stage", "overripe", "ripeness"}
    assert rep["questions"]["stage"]["holdout"]["acc"] > 0.9
    assert rep["questions"]["ripeness"]["holdout"]["mae"] < 0.3
    main(["decide", "--backbone", "stub", "--recipe", str(recipe), "--calib", str(calib), str(root / "Overripe" / "000.png")])
    out = json.loads(capsys.readouterr().out)["answers"]
    assert out["stage"]["choice"] == "Overripe" and out["overripe"]["noul"] > 0.5 and out["ripeness"]["score"] > 1.5
    assert all(a["calibrated"] and not a["abstain"] for a in out.values())


def test_thin_option_warning(tmp_path, capsys):
    root = tmp_path / "imgs"
    write_folder(root, per_class=6)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(RECIPE))
    main(["calibrate", "--backbone", "stub", "--recipe", str(recipe), "--images", str(root), "--out", str(tmp_path / "c.npz"), "--holdout", "0"])
    rep = json.loads(capsys.readouterr().out)
    assert "warning" in rep["questions"]["stage"]


def test_server_roundtrip(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from certo_vision.server import create_app
    root = tmp_path / "imgs"
    write_folder(root)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(RECIPE))
    main(["calibrate", "--backbone", "stub", "--recipe", str(recipe), "--images", str(root), "--out", str(tmp_path / "c.npz"), "--holdout", "0"])
    client = TestClient(create_app("stub", calib=str(tmp_path / "c.npz"), recipe=str(recipe)))
    assert client.get("/health").json()["calibrated_questions"] == ["stage", "overripe", "ripeness"]
    import base64
    b64 = base64.b64encode((root / "Green" / "001.png").read_bytes()).decode()
    r = client.post("/v1/systemone", json={"state": {"image_b64": b64}, "questions": ["stage", "overripe"]}).json()
    # the stub reads the concept from PNG metadata, which survives the b64 round trip
    assert r["answers"]["stage"]["choice"] == "Green" and r["answers"]["stage"]["calibrated"]
    assert client.post("/v1/systemone", json={"state": {"image_b64": b64}, "questions": ["nope"]}).status_code == 400
