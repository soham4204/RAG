"""In-memory store of vectors plus the metadata that belongs to them.

A vector database record has three parts: an id, a vector, and a payload
(the original text and where it came from). This class is the small-corpus
version of that record.

Search compares the query with every stored vector and sorts by cosine
similarity. That exact scan is fine for a few dozen vectors. At a million
vectors a database uses an approximate index such as HNSW so it does not
look at every row. The trade-off is speed against a chance of missing the
true nearest neighbor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Record:
    """One stored item. `chunk_index` is None when the whole document was embedded."""

    id: str
    title: str
    section: str
    source: str
    page: int
    text: str
    chunk_index: int | None = None


@dataclass
class Hit:
    rank: int
    score: float
    record: Record


class VectorIndex:
    def __init__(self) -> None:
        self.records: list[Record] = []
        self.vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def add(self, records: list[Record], vectors: np.ndarray) -> None:
        if len(records) != len(vectors):
            raise ValueError("each record needs exactly one vector")
        if self.vectors is None:
            self.records = list(records)
            self.vectors = vectors
            return
        self.records.extend(records)
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Hit]:
        """Return the nearest records.

        This always returns something when the index is not empty. A low
        score still comes back. The index does not know whether the corpus
        actually contains an answer.
        """
        if self.vectors is None or not self.records:
            return []
        # Dot product of unit vectors = cosine similarity.
        scores = self.vectors @ query_vector
        k = min(top_k, len(scores))
        # argpartition finds the top k without sorting the whole list.
        best = np.argpartition(-scores, k - 1)[:k]
        ordered = best[np.argsort(-scores[best])]
        return [
            Hit(rank=position, score=float(scores[index]), record=self.records[index])
            for position, index in enumerate(ordered, start=1)
        ]
