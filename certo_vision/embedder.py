"""Embedders. SiglipEmbedder (SigLIP 2, the default backbone) and GemmaEmbedder (EmbeddingGemma 2)
wrap the real models; StubEmbedder is a deterministic fake used to test the pipeline without weights."""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

CLS_PREFIX = "task: classification | query: "


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class GemmaEmbedder:
    """EmbeddingGemma 2 via sentence-transformers >= 6.1.

    Interleaved image input follows the model card:
        model.encode({"text": "... <|image|>", "image": ["a.jpg"]})
    Check the card if the call signature differs in your installed version.
    """

    dim = 768

    def __init__(self, model_id: str = "google/embeddinggemma-2", device: str | None = None,
                 vision_budget: int | None = None, prefix: str = CLS_PREFIX):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id, device=device)
        self.prefix = prefix
        self.vision_budget = vision_budget  # 70..1120; set via processor kwargs if exposed

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        v = self.model.encode([self.prefix + t for t in texts], normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def embed_state(self, image=None, text: str | None = None) -> np.ndarray:
        """Encode the state ONCE. image: path or PIL.Image; text: optional context."""
        if image is None:
            return self.embed_texts([text or ""])[0]
        body = (text + " " if text else "") + "<|image|>"
        v = self.model.encode({"text": self.prefix + body, "image": [image]},
                              normalize_embeddings=True)
        v = np.asarray(v, dtype=np.float32)
        return v.reshape(-1)[: self.dim] if v.ndim > 1 and v.shape[0] == 1 else v.reshape(-1)


class StubEmbedder:
    """Deterministic fake. Text -> hashed bag-of-words vector. Images -> a vector made
    from a hidden 'concept' string attached to the image (for synthetic tests),
    plus noise. Lets us verify calibration/metrics/latency plumbing end-to-end."""

    dim = 768

    def __init__(self, noise: float = 0.9, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def _word(self, w: str) -> np.ndarray:
        h = int(hashlib.md5(w.lower().encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h).standard_normal(self.dim).astype(np.float32)

    def _text(self, t: str) -> np.ndarray:
        words = [w.strip(".,:;!?'\"()") for w in t.split()]
        words = [w for w in words if len(w) > 2]
        return l2n(sum((self._word(w) for w in words), np.zeros(self.dim, np.float32)))

    def embed_texts(self, texts):
        return np.stack([self._text(t) for t in texts])

    def embed_state(self, image=None, text=None):
        concept = getattr(image, "concept", None) or (image.info.get("concept") if image is not None else None)
        base = self._text(concept or text or "")
        return l2n(base + self.noise * l2n(self.rng.standard_normal(self.dim).astype(np.float32)))


def _feat(out):
    """transformers >=5 returns BaseModelOutputWithPooling from get_*_features; older returns a tensor."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class SiglipEmbedder:
    """SigLIP 2 via transformers. Default backbone (same interface as GemmaEmbedder; no task prefix)."""

    def __init__(self, model_id: str = "google/siglip2-so400m-patch14-384", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = self.model.config.projection_dim if hasattr(self.model.config, "projection_dim") else 1152

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 64):
                inp = self.proc(text=list(texts[i:i + 64]), padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
                out.append(_feat(self.model.get_text_features(**inp)).float().cpu().numpy())
        return l2n(np.concatenate(out).astype(np.float32))

    def embed_state(self, image=None, text: str | None = None) -> np.ndarray:
        if image is None:
            return self.embed_texts([text or ""])[0]
        from PIL import Image
        img = Image.open(image).convert("RGB") if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__") else image
        with self.torch.no_grad():
            inp = self.proc(images=[img], return_tensors="pt").to(self.device)
            v = _feat(self.model.get_image_features(**inp)).float().cpu().numpy()
        return l2n(v.reshape(-1).astype(np.float32))
