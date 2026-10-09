"""Retrieval metrics for a ranked list of chunks.

A chunk is relevant when its parent document is in the gold set. Chunk
indexes are not labeled, because a different chunk size cuts the same
document in a different place.

Precision@K asks how many of the K retrieved chunks were relevant.
Recall@K asks how many of the gold documents showed up at least once.
MRR asks how high the first relevant chunk was. Unanswerable questions
have no gold document, so they are not included in these averages.
"""

from __future__ import annotations


def parent_document_id(record_id: str) -> str:
    """Recover the corpus id stored on the front of a chunk id."""
    if record_id.endswith("_doc"):
        return record_id[: -len("_doc")]
    head, separator, _tail = record_id.rpartition("_chunk_")
    if separator:
        return head
    return record_id


def precision_at_k(document_ids: list[str], gold: set[str], k: int) -> float:
    """Share of the top K chunks that belong to a gold document."""
    if k <= 0:
        return 0.0
    top = document_ids[:k]
    hits = sum(1 for document_id in top if document_id in gold)
    return hits / k


def recall_at_k(document_ids: list[str], gold: set[str], k: int) -> float:
    """Share of gold documents that appear at least once in the top K."""
    if not gold:
        return 0.0
    found = {document_id for document_id in document_ids[:k] if document_id in gold}
    return len(found) / len(gold)


def reciprocal_rank(document_ids: list[str], gold: set[str]) -> float:
    """1 / rank of the first gold chunk. Zero when none of the hits match."""
    for rank, document_id in enumerate(document_ids, start=1):
        if document_id in gold:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
