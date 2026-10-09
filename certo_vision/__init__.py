"""certo-vision: calibrated typed image decisions (noul / choice / score) with abstention,
on a frozen SigLIP 2 backbone and a few dozen labelled images per question."""
from .decider import Decider, Question  # noqa: F401
from .embedder import GemmaEmbedder, SiglipEmbedder, StubEmbedder  # noqa: F401

__version__ = "0.1.0"
