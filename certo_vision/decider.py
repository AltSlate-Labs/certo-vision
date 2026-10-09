"""Typed image decisions (noul / choice / score) over a frozen image-text embedding model.

State is embedded once; every question is a set of cached text prototypes, so all
questions are answered by dot products against the same state vector — parallel
and isolated, like Jev's single-pass design.

Question types mirror the Jev API (TypeSafe AI):
  noul   -> {"type": "noul", "noul": p_true}
  choice -> {"type": "choice", "choice": k, "probabilities": {...}, "confidence": c}
  score  -> {"type": "score", "score": E[level], "legend": {...}, "probabilities": {...}, "confidence": c}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .embedder import l2n

DEFAULT_LOGIT_SCALE = 50.0  # zero-shot prior; replaced by fitted temperature


@dataclass
class Question:
    qid: str
    type: str                      # noul | choice | score
    instructions: str
    keys: list[str]                # option ids / level ids / ["true","false"]
    texts: list[str]               # text that gets embedded for each key
    legend: dict | None = None
    protos: np.ndarray | None = None        # K x D (unit)
    scale: float = DEFAULT_LOGIT_SCALE      # fitted temperature (1/T)
    bias: np.ndarray | None = None          # K, fitted
    calib_states: np.ndarray | None = None  # N x D, for OOD-based confidence
    calib_sim_ref: float | None = None      # typical NN similarity in calibration set
    abstain_threshold: float | None = None  # confidence below this => abstain (set by calibrate)

    @classmethod
    def from_spec(cls, qid: str, spec: dict) -> "Question":
        t, ins, crit = spec["type"], spec["instructions"], spec.get("criteria")
        if t == "noul":
            crit = crit or {"true": f"Yes: {ins}", "false": f"No, the opposite of: {ins}"}
            keys = ["true", "false"]
            texts = [f"{ins} — {crit['true']}", f"{ins} — {crit['false']}"]
            return cls(qid, t, ins, keys, texts)
        if t == "choice":
            if not isinstance(crit, dict) or not 2 <= len(crit) <= 255:
                raise ValueError(f"{qid}: choice needs 2..255 criteria")
            keys = list(crit)
            texts = [f"{ins} — {k.replace('_', ' ')}: {d}" for k, d in crit.items()]
            return cls(qid, t, ins, keys, texts)
        if t == "score":
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                raise ValueError(f"{qid}: score needs 2..10 levels")
            keys = [str(i) for i in range(len(crit))]
            texts = [f"{ins} — {c}" for c in crit]
            return cls(qid, t, ins, keys, texts, legend=dict(zip(keys, crit)))
        raise ValueError(f"{qid}: unknown type {t}")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _trunc(v: np.ndarray, d: int) -> np.ndarray:
    return l2n(v[..., :d])


class Decider:
    def __init__(self, embedder, mrl_dim: int | None = None):
        self.emb = embedder
        self.mrl_dim = mrl_dim or embedder.dim  # truncation only matters for Matryoshka backbones
        self.questions: dict[str, Question] = {}

    # ---------- schema ----------
    def compile(self, questions: dict) -> dict[str, Question]:
        """Embed question prototypes once and cache them (schemas are static)."""
        out = {}
        for qid, spec in questions.items():
            key = qid + json.dumps(spec, sort_keys=True)
            if key in self.questions:
                out[qid] = self.questions[key]
                continue
            q = Question.from_spec(qid, spec)
            q.protos = self.emb.embed_texts(q.texts)
            self.questions[key] = q
            out[qid] = q
        return out

    # ---------- core math ----------
    def _probs(self, q: Question, s: np.ndarray, d: int | None = None) -> np.ndarray:
        d = d or self.mrl_dim
        logits = q.scale * (_trunc(q.protos, d) @ _trunc(s, d))
        if q.bias is not None:
            logits = logits + q.bias
        return _softmax(logits)

    def _confidence(self, q: Question, s: np.ndarray, p_full: np.ndarray | None = None) -> float:
        """Scope confidence, separate from the answer probability: the state's nearest-neighbour
        similarity to the calibration set, relative to the calibration set's own typical NN
        similarity. 1.0 = as close as calibration images are to each other; below
        q.abstain_threshold => out of scope. (Matryoshka agreement was removed: it does not
        separate in-scope from out-of-scope images.) Without calibration data: 1.0."""
        if q.calib_states is None or not q.calib_sim_ref:
            return 1.0
        nn = float((q.calib_states @ l2n(s)).max())
        return round(float(np.clip(nn / q.calib_sim_ref, 0.0, 1.0)), 4)

    # ---------- public API (Jev-shaped) ----------
    def decide(self, state: dict, questions: dict) -> dict:
        """state: {"image": PIL.Image|path|None, "text": str|None}"""
        t0 = time.perf_counter()
        qs = self.compile(questions)
        s = self.emb.embed_state(image=state.get("image"), text=state.get("text"))
        t1 = time.perf_counter()
        answers = {qid: self._answer(q, s) for qid, q in qs.items()}
        t2 = time.perf_counter()
        return {"model": "certo-vision-0.1", "answers": answers,
                "timing_ms": {"embed": round((t1 - t0) * 1e3, 2), "decide": round((t2 - t1) * 1e3, 3)}}

    def _answer(self, q: Question, s: np.ndarray) -> dict:
        p = self._probs(q, s)
        conf = self._confidence(q, s, p)
        meta = {"confidence": conf, "calibrated": q.calib_states is not None,
                "abstain": bool(q.abstain_threshold is not None and conf < q.abstain_threshold)}
        if q.type == "noul":
            return {"type": "noul", "noul": round(float(p[0]), 4)} | meta
        probs = {k: round(float(v), 4) for k, v in zip(q.keys, p)}
        if q.type == "choice":
            return {"type": "choice", "choice": q.keys[int(p.argmax())], "probabilities": probs} | meta
        score = float((np.arange(len(p)) * p).sum())
        return {"type": "score", "score": round(score, 3), "legend": q.legend, "probabilities": probs} | meta

    # ---------- calibration (the RLCD stand-in) ----------
    def calibrate(self, qid: str, spec: dict, states: np.ndarray, labels: list[str],
                  few_shot_alpha: float = 0.0, fit_bias: bool = True, iters: int = 400,
                  coverage: float = 0.95) -> dict:
        """Fit temperature (+ per-option bias) by minimising NLL on labelled states.
        few_shot_alpha > 0 blends each text prototype with the mean embedding of
        that option's labelled images (Tip-Adapter-style prototype refinement).
        Fit on a calibration split; evaluate on a separate test split."""
        q = self.compile({qid: spec})[qid]
        S = l2n(np.asarray(states, np.float32))
        y = np.array([q.keys.index(str(l)) for l in labels])
        if few_shot_alpha > 0:
            P = q.protos.copy()
            for k in range(len(q.keys)):
                if (y == k).any():
                    P[k] = l2n((1 - few_shot_alpha) * P[k] + few_shot_alpha * l2n(S[y == k].mean(0)))
            q.protos = P
        sims = _trunc(S, self.mrl_dim) @ _trunc(q.protos, self.mrl_dim).T
        log_a, b = np.log(DEFAULT_LOGIT_SCALE), np.zeros(len(q.keys))
        lr = 0.05
        for _ in range(iters):  # plain gradient descent; tiny problem
            a = np.exp(log_a)
            p = _softmax(a * sims + b)
            g = p.copy()
            g[np.arange(len(y)), y] -= 1
            g /= len(y)
            log_a -= lr * float((g * sims).sum() * a)
            if fit_bias:
                b -= lr * 10 * g.sum(0)
                b -= b.mean()
        q.scale, q.bias = float(np.exp(log_a)), b
        q.calib_states = S
        G = S @ S.T
        np.fill_diagonal(G, -1)
        loo_nn = G.max(1)                                   # leave-one-out NN similarity per calib image
        q.calib_sim_ref = float(np.median(loo_nn))
        q.abstain_threshold = float(np.quantile(np.clip(loo_nn / q.calib_sim_ref, 0, 1), 1 - coverage))
        p = _softmax(q.scale * sims + q.bias)
        nll = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
        return {"qid": qid, "scale": round(q.scale, 3), "bias": np.round(b, 3).tolist(), "train_nll": round(nll, 4),
                "abstain_threshold": round(q.abstain_threshold, 4)}

    # ---------- persistence ----------
    def save(self, path) -> None:
        """Save every compiled (and calibrated) question: prototypes, fit, calibration states."""
        meta, arrays = {}, {}
        for i, (key, q) in enumerate(self.questions.items()):
            tag = f"q{i}"
            meta[tag] = {"key": key, "qid": q.qid, "type": q.type, "instructions": q.instructions,
                         "keys": q.keys, "texts": q.texts, "legend": q.legend, "scale": q.scale,
                         "calib_sim_ref": q.calib_sim_ref, "abstain_threshold": q.abstain_threshold}
            arrays[f"{tag}__protos"] = q.protos
            if q.bias is not None:
                arrays[f"{tag}__bias"] = q.bias
            if q.calib_states is not None:
                arrays[f"{tag}__calib_states"] = q.calib_states
        np.savez(path, meta=json.dumps({"mrl_dim": self.mrl_dim, "questions": meta}), **arrays)

    def load(self, path) -> list[str]:
        """Load questions saved by save(); returns the loaded qids. Keys must match compile()'s."""
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        for tag, m in meta["questions"].items():
            q = Question(m["qid"], m["type"], m["instructions"], m["keys"], m["texts"], legend=m["legend"],
                         protos=z[f"{tag}__protos"], scale=m["scale"],
                         bias=z[f"{tag}__bias"] if f"{tag}__bias" in z else None,
                         calib_states=z[f"{tag}__calib_states"] if f"{tag}__calib_states" in z else None,
                         calib_sim_ref=m["calib_sim_ref"], abstain_threshold=m["abstain_threshold"])
            self.questions[m["key"]] = q
        return [m["qid"] for m in meta["questions"].values()]

    def confidence_batch(self, qid: str, spec: dict, states: np.ndarray) -> np.ndarray:
        """Scope confidence per state (see _confidence), for evaluation."""
        q = self.compile({qid: spec})[qid]
        return np.array([self._confidence(q, s) for s in np.asarray(states, np.float32)])

    def probs_batch(self, qid: str, spec: dict, states: np.ndarray) -> np.ndarray:
        """Vectorised probabilities for evaluation (no per-call embedding)."""
        q = self.compile({qid: spec})[qid]
        S = _trunc(np.asarray(states, np.float32), self.mrl_dim)
        logits = q.scale * (S @ _trunc(q.protos, self.mrl_dim).T)
        if q.bias is not None:
            logits = logits + q.bias
        return _softmax(logits)
