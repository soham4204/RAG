"""HTTP API for semantic search and RAG.

Run from the backend folder:

    uvicorn main:app --reload --port 8000

POST /search embeds the question and returns the nearest whole documents.
POST /rag chunks the corpus, retrieves passages, and returns an answer
built only from those passages. Both routes share the embedding model.
They do not share an index: search stores one vector per document, and
RAG stores one vector per chunk.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from embeddings import MODEL_NAME
from evaluate import evaluate_all
from pipelines import KnowledgeBase

# .env lives in the project root, one level above this file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("knowledge")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class RagRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

    @model_validator(mode="after")
    def overlap_fits_window(self) -> "RagRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class HitOut(BaseModel):
    rank: int
    score: float
    id: str
    title: str
    section: str
    source: str
    page: int
    text: str
    chunk_index: int | None = None


class SearchResponse(BaseModel):
    mode: str
    query: str
    threshold: float
    low_confidence: bool
    results: list[HitOut]


class RagResponse(BaseModel):
    mode: str
    query: str
    answer: str
    refused: bool
    generation_mode: str
    threshold: float
    chunk_size: int
    chunk_overlap: int
    prompt: str
    sources: list[HitOut]


def _hits(hits) -> list[HitOut]:
    return [
        HitOut(
            rank=hit.rank,
            score=round(hit.score, 4),
            id=hit.record.id,
            title=hit.record.title,
            section=hit.record.section,
            source=hit.record.source,
            page=hit.record.page,
            text=hit.record.text,
            chunk_index=hit.record.chunk_index,
        )
        for hit in hits
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The model download and the first embeddings happen before the server
    # accepts traffic, so the first question does not pay that cost.
    log.info("Loading embedding model %s", MODEL_NAME)
    base = KnowledgeBase()
    base.startup()
    app.state.base = base
    log.info(
        "Ready: %s documents, %s document vectors",
        len(base.documents),
        len(base.document_index),
    )
    yield


app = FastAPI(title="Semantic Search and RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    base: KnowledgeBase = app.state.base
    return {
        "status": "ok",
        "embedding_model": MODEL_NAME,
        "documents": len(base.documents),
        "semantic_vectors": len(base.document_index),
        "rag_indexes": {
            f"{size}/{overlap}": len(index)
            for (size, overlap), index in base._chunk_indexes.items()
        },
    }


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")
    found = app.state.base.semantic_search(query, body.top_k)
    return SearchResponse(
        mode="semantic_search",
        query=query,
        threshold=found.threshold,
        low_confidence=found.low_confidence,
        results=_hits(found.hits),
    )


@app.post("/evaluate")
def evaluate() -> dict:
    """Score the labeled question set at three chunk sizes.

    This calls the same RAG path as /rag, once per question per preset.
    With a Gemini key that is one model call each, so the request can
    take a few minutes.
    """
    return evaluate_all(app.state.base)


@app.post("/rag", response_model=RagResponse)
def rag(body: RagRequest) -> RagResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")
    result = app.state.base.answer(query, body.top_k, body.chunk_size, body.chunk_overlap)
    return RagResponse(
        mode="rag",
        query=query,
        answer=result.answer,
        refused=result.refused,
        generation_mode=result.generation_mode,
        threshold=result.threshold,
        chunk_size=result.chunk_size,
        chunk_overlap=result.chunk_overlap,
        prompt=result.prompt,
        sources=_hits(result.hits),
    )
