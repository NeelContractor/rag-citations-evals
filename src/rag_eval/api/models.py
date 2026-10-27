from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from rag_eval.api.highlight import highlight_quote_html
from rag_eval.generation.pipeline import RagResponse


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    hybrid: bool = False
    rerank: bool = False

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class SourceOut(BaseModel):
    chunk_id: str
    doc: str
    page: int | None = None
    heading: str | None = None
    chunk_text: str
    quote: str
    highlighted_html: str
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    model: str
    sources: list[SourceOut]
    invalid_citation_count: int = 0
    latency_ms: float

    @classmethod
    def from_rag_response(cls, response: RagResponse, latency_ms: float) -> AskResponse:
        scores = {chunk.chunk_id: chunk.score for chunk in response.retrieved}
        sources = [
            SourceOut(
                chunk_id=citation.chunk_id,
                doc=citation.doc,
                page=citation.page,
                heading=citation.heading,
                chunk_text=citation.chunk_text,
                quote=citation.quote,
                highlighted_html=highlight_quote_html(citation.chunk_text, citation.quote),
                score=scores.get(citation.chunk_id),
            )
            for citation in response.citations
        ]
        return cls(
            question=response.question,
            answer=response.answer,
            refused=response.refused,
            refusal_reason=response.refusal_reason,
            model=response.model,
            sources=sources,
            invalid_citation_count=len(response.invalid_citations),
            latency_ms=latency_ms,
        )
