"""Two pipelines over the same documents.

Semantic search (retrieval only)
    documents -> embed each document -> store
    query -> embed -> cosine search -> top documents and scores

    It stops at the passages. Nothing is written as an answer. The nearest
    document is returned even when the question is not covered.

RAG (retrieval, then generation)
    documents -> chunk -> embed each chunk -> store
    query -> embed -> retrieve top chunks -> build a prompt -> LLM answer

    The model is told to use only the retrieved text. If that text does not
    contain the answer, the reply should say so. Retrieval can still fail:
    the right chunk may be missing, or the model may misread a chunk that
    was found. RAG does not remove those failure modes.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from chunking import chunk_text
from corpus import DOCUMENTS
from embeddings import EmbeddingModel
from vector_store import Hit, Record, VectorIndex

# Below this cosine score the nearest chunk is treated as not relevant
# enough to answer from. On this model and corpus the demo questions land
# above about 0.72, while a question the documents do not cover (stock
# price) still has a nearest neighbor around 0.59. The cutoff sits in
# that gap. It is a heuristic, not a property of cosine similarity.
SIMILARITY_THRESHOLD = 0.65

REFUSAL = (
    "I don't have enough information in the provided documents to answer that."
)

# Indexing is offline relative to a single question: chunks are embedded
# once per (chunk_size, overlap) pair and reused for later questions.
ChunkKey = tuple[int, int]


@dataclass
class SearchResult:
    hits: list[Hit]
    threshold: float

    @property
    def low_confidence(self) -> bool:
        return not self.hits or self.hits[0].score < self.threshold


@dataclass
class RagResult:
    answer: str
    hits: list[Hit]
    prompt: str
    generation_mode: str
    refused: bool
    threshold: float
    chunk_size: int
    chunk_overlap: int


class KnowledgeBase:
    def __init__(self) -> None:
        self.documents = DOCUMENTS
        self.model = EmbeddingModel()
        self.document_index = VectorIndex()
        self._chunk_indexes: dict[ChunkKey, VectorIndex] = {}
        self._lock = threading.Lock()

    def startup(self) -> None:
        """Build the semantic-search index and the default RAG chunk index."""
        self._build_document_index()
        self.chunk_index(500, 50)

    def semantic_search(self, query: str, top_k: int) -> SearchResult:
        """Embed the question and return the nearest whole documents."""
        hits = self.document_index.search(self.model.embed_query(query), top_k)
        return SearchResult(hits=hits, threshold=SIMILARITY_THRESHOLD)

    def answer(self, query: str, top_k: int, chunk_size: int, overlap: int) -> RagResult:
        """Retrieve chunks, then generate an answer grounded in those chunks."""
        index = self.chunk_index(chunk_size, overlap)
        hits = index.search(self.model.embed_query(query), top_k)
        prompt = build_prompt(query, hits)
        answer, mode, refused = generate(prompt, hits)
        return RagResult(
            answer=answer,
            hits=hits,
            prompt=prompt,
            generation_mode=mode,
            refused=refused,
            threshold=SIMILARITY_THRESHOLD,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk_index(self, chunk_size: int, overlap: int) -> VectorIndex:
        key = (chunk_size, overlap)
        with self._lock:
            cached = self._chunk_indexes.get(key)
            if cached is not None:
                return cached
            index = self._build_chunk_index(chunk_size, overlap)
            self._chunk_indexes[key] = index
            return index

    def _build_document_index(self) -> None:
        # One vector per document. This is the Day 3 semantic-search shape:
        # no chunking and no language model.
        records = [_record(document, text=document["text"], chunk_index=None) for document in self.documents]
        vectors = self.model.embed_documents([record.text for record in records])
        self.document_index = VectorIndex()
        self.document_index.add(records, vectors)

    def _build_chunk_index(self, chunk_size: int, overlap: int) -> VectorIndex:
        records: list[Record] = []
        for document in self.documents:
            pieces = chunk_text(document["text"], chunk_size, overlap)
            for index, piece in enumerate(pieces):
                records.append(_record(document, text=piece, chunk_index=index))
        index = VectorIndex()
        vectors = self.model.embed_documents([record.text for record in records])
        index.add(records, vectors)
        return index


def _record(document: dict, text: str, chunk_index: int | None) -> Record:
    suffix = "doc" if chunk_index is None else f"chunk_{chunk_index:02d}"
    return Record(
        id=f"{document['id']}_{suffix}",
        title=document["title"],
        section=document["section"],
        source=document["source"],
        page=document["page"],
        text=text,
        chunk_index=chunk_index,
    )


def build_prompt(query: str, hits: list[Hit]) -> str:
    """Put the question and the retrieved chunks into one grounded prompt.

    The instruction to answer only from the context is what keeps the model
    from filling gaps with its own memory. If the chunks are the wrong ones,
    the model should refuse rather than guess.
    """
    blocks = []
    for hit in hits:
        record = hit.record
        location = record.title
        if record.chunk_index is not None:
            location = f"{location}, chunk {record.chunk_index}"
        blocks.append(
            f"[{hit.rank}] {location} ({record.source}, page {record.page})\n{record.text}"
        )
    context = "\n\n".join(blocks) if blocks else "(no documents retrieved)"
    return (
        "You are an assistant answering questions using the provided context.\n\n"
        "Context:\n"
        "------------------\n"
        f"{context}\n"
        "------------------\n\n"
        f"Question:\n{query}\n\n"
        "Answer using only the provided context.\n"
        "If the answer cannot be found in the context, say that you don't "
        "have enough information.\n"
        "Cite the source title you used."
    )


def generate(prompt: str, hits: list[Hit]) -> tuple[str, str, bool]:
    """Return (answer, mode, refused).

    A Gemini API key turns the prompt into a real generation step.
    Without a key, the fallback quotes the best chunk or refuses when the
    score is weak. Quoting is not a language model. It only keeps the demo
    runnable and still shows the retrieved context.
    """
    if not hits or hits[0].score < SIMILARITY_THRESHOLD:
        # Do not send a weak context to a model that might invent an answer.
        return REFUSAL, "refused_low_similarity", True

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        try:
            return _complete_gemini(prompt, api_key, model), "gemini", False
        except Exception as exc:
            answer = _quote_best_chunk(hits)
            return answer, f"extractive_fallback (gemini error: {exc})", False

    return _quote_best_chunk(hits), "extractive_fallback", False


def _quote_best_chunk(hits: list[Hit]) -> str:
    """Quote every retrieved chunk that cleared the similarity bar.

    This is the no-model fallback. A language model would synthesize one
    answer; here the passages are shown directly so the reply cannot add
    facts that were not retrieved.
    """
    best = hits[0]
    # Keep chunks from the same document that are close to the best score.
    # A weaker neighbor can still clear the absolute threshold; it stays in
    # the source list, but it is not pasted into the quoted answer.
    kept = [
        hit
        for hit in hits
        if hit.score >= SIMILARITY_THRESHOLD
        and hit.record.title == best.record.title
        and best.score - hit.score <= 0.05
    ]
    if not kept:
        return REFUSAL
    parts = []
    for hit in kept:
        record = hit.record
        where = record.title
        if record.chunk_index is not None:
            where = f"{where}, chunk {record.chunk_index}"
        parts.append(f"From {where} ({record.source}): {record.text}")
    return "\n\n".join(parts)


def _complete_gemini(prompt: str, api_key: str, model: str) -> str:
    # Imported here so the API still starts when the Gemini package is absent
    # and the user is on the extractive fallback.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    # Low thinking fits a short grounded answer. Gemini 3 is tuned for its
    # default temperature, so that setting is left alone.
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        ),
    )
    text = (response.text or "").strip()
    return text or REFUSAL
