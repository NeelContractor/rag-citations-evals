from __future__ import annotations

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc: str
    page: int | None = None
    heading: str | None = None
    text: str
    score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
