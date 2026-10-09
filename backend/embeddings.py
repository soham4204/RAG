"""Turn text into vectors with one shared embedding model.

An embedding is a list of numbers whose direction represents meaning.
Two texts about the same idea point a similar way, even when they share
few words. Cosine similarity measures that angle. After both vectors are
scaled to length 1, cosine similarity is just their dot product.

The query and the documents must use the same model. Vectors from two
different models do not live in the same space, so comparing them is
meaningless.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

# Small, local model. No API key. The first run downloads the weights.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # fastembed runs an ONNX model on CPU. BGE expects a short
        # instruction prefix on queries; query_embed adds it, embed does
        # not. Documents and queries therefore use different methods of
        # the same model.
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages that will be stored and searched."""
        if not texts:
            raise ValueError("no texts to embed")
        return _normalize(list(self._model.embed(texts)))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one user question into the same space as the documents."""
        matrix = _normalize(list(self._model.query_embed([text])))
        return matrix[0]


def _normalize(vectors: list[np.ndarray]) -> np.ndarray:
    """Scale each vector to length 1 so a dot product is cosine similarity."""
    matrix = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
