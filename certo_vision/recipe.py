"""A recipe is a JSON file declaring the questions for one deployment and, for calibration and
trials, how labelled image folders (root/<folder>/*.png) map onto option keys:

{
 "name": "banana",
 "questions": {
   "<qid>": {"type": "noul|choice|score", "instructions": "...", "criteria": ...,
             "labels": {"<folder>": "<key>", "*": "<key for every other folder>"} | "folder"}
 },
 "ood": {"<name>": "path/to/folder/of/images"}      # optional, trials only
}
"labels": "folder" means the folder name is the option key; folders that are not keys are skipped.
"""
from __future__ import annotations

import json
import pathlib

SPEC_KEYS = ("type", "instructions", "criteria")
DEFAULT_MODELS = {"siglip": "google/siglip2-so400m-patch14-384", "gemma": "google/embeddinggemma-2"}


def load_recipe(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def question_spec(qcfg: dict) -> dict:
    """The part of a question the decider sees. Must match the served request exactly."""
    return {k: qcfg[k] for k in SPEC_KEYS if k in qcfg}


def specs(recipe: dict) -> dict:
    return {qid: question_spec(q) for qid, q in recipe["questions"].items()}


def option_keys(spec: dict) -> list[str]:
    if spec["type"] == "noul":
        return ["true", "false"]
    if spec["type"] == "choice":
        return list(spec["criteria"])
    return [str(i) for i in range(len(spec["criteria"]))]


def folder_label(folder: str, qcfg: dict, keys: list[str]) -> str | None:
    """Option key of an image folder under this question, or None if the folder is unlabelled."""
    lab = qcfg.get("labels", "folder")
    k = folder if lab == "folder" else lab.get(folder, lab.get("*"))
    return k if k in keys else None


def labelled(items, qcfg: dict, keys: list[str]):
    """Indices into items and their option keys for the images this question has labels for."""
    idx, y = [], []
    for i, (_, folder) in enumerate(items):
        k = folder_label(folder, qcfg, keys)
        if k is not None:
            idx.append(i)
            y.append(k)
    return idx, y


def make_embedder(backbone: str = "siglip", model: str | None = None, device: str | None = None):
    if backbone == "siglip":
        from .embedder import SiglipEmbedder
        return SiglipEmbedder(model or DEFAULT_MODELS["siglip"], device)
    if backbone == "gemma":
        from .embedder import GemmaEmbedder
        return GemmaEmbedder(model or DEFAULT_MODELS["gemma"], device)
    if backbone == "stub":
        from .embedder import StubEmbedder
        return StubEmbedder(noise=1.0)
    raise ValueError(f"unknown backbone {backbone}")
