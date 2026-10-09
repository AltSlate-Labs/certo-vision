"""Jev-shaped HTTP endpoint.

POST /v1/systemone
{"state": {"image_url" | "image_b64": "...", "text": "optional"},
 "questions": {"qid": {"type": "noul|choice|score", "instructions": "...", "criteria": ...}}
             | ["qid", ...]            # names from the loaded recipe
}
Each answer carries "confidence" (nearest-neighbour scope score), "calibrated" (a saved fit matched
this qid + spec) and "abstain" (confidence below the fitted threshold).

    certo-vision serve --calib demo/banana/calib.npz --recipe recipes/banana.json --port 8080
or  CV_CALIB=... CV_RECIPE=... uvicorn certo_vision.server:app --port 8080
"""
from __future__ import annotations

import base64
import io
import os

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from .decider import Decider
from .recipe import load_recipe, make_embedder, specs


class Req(BaseModel):
    model: str = "certo-vision"
    state: dict
    questions: dict | list[str]


def create_app(backbone="siglip", model=None, calib=None, recipe=None, device=None) -> FastAPI:
    dec = Decider(make_embedder(backbone, model, device))
    loaded = dec.load(calib) if calib else []
    named = specs(load_recipe(recipe)) if recipe else {}
    print(f"[certo-vision] backbone={backbone} calibrated questions: {loaded} recipe questions: {list(named)}")
    app = FastAPI(title="certo-vision")

    @app.get("/health")
    def health():
        return {"backbone": backbone, "calibrated_questions": loaded, "recipe_questions": list(named)}

    @app.post("/v1/systemone")
    def systemone(r: Req):
        if isinstance(r.questions, list):
            missing = [q for q in r.questions if q not in named]
            if missing:
                raise HTTPException(400, f"unknown recipe questions: {missing}")
            questions = {q: named[q] for q in r.questions}
        else:
            questions = r.questions
        img = None
        if "image_b64" in r.state:
            img = Image.open(io.BytesIO(base64.b64decode(r.state["image_b64"]))).convert("RGB")
        elif "image_url" in r.state:
            import urllib.request
            with urllib.request.urlopen(r.state["image_url"], timeout=10) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
        try:
            return dec.decide({"image": img, "text": r.state.get("text")}, questions)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


app = create_app(os.environ.get("CV_BACKBONE", "siglip"), os.environ.get("CV_MODEL"),
                 os.environ.get("CV_CALIB"), os.environ.get("CV_RECIPE")) if os.environ.get("CV_CALIB") or os.environ.get("CV_RECIPE") else None
