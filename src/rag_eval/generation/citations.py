from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag_eval.retrieval.types import RetrievedChunk

REFUSAL_ANSWER = "I don't have enough information in the provided sources to answer that question."
MIN_QUOTE_CHARS = 3


class Citation(BaseModel):
    chunk_id: str
    quote: str


class LLMAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class InvalidCitation(BaseModel):
    citation: Citation
    reason: Literal["unknown_chunk", "empty_quote", "quote_not_found"]


class VerifiedCitation(BaseModel):
    chunk_id: str
    quote: str
    doc: str
    page: int | None = None
    heading: str | None = None
    chunk_text: str


class VerifiedAnswer(BaseModel):
    answer: str
    citations: list[VerifiedCitation]
    invalid_citations: list[InvalidCitation]
    refused: bool = False
    refusal_reason: str | None = None


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    normalized_quote = normalize_whitespace(quote).casefold()
    normalized_chunk = normalize_whitespace(chunk_text).casefold()
    return bool(normalized_quote) and normalized_quote in normalized_chunk


def is_refusal(answer: str) -> bool:
    normalized = normalize_whitespace(answer).casefold()
    return normalized.startswith(
        (
            "i don't have enough information",
            "i do not have enough information",
            "insufficient information",
        )
    )


def verify_citations(
    llm_answer: LLMAnswer,
    retrieved: list[RetrievedChunk],
    require_citations: bool = True,
) -> VerifiedAnswer:
    if is_refusal(llm_answer.answer):
        return VerifiedAnswer(
            answer=REFUSAL_ANSWER,
            citations=[],
            invalid_citations=[],
            refused=True,
            refusal_reason="model_refusal",
        )

    by_id = {chunk.chunk_id: chunk for chunk in retrieved}
    valid: list[VerifiedCitation] = []
    invalid: list[InvalidCitation] = []
    seen: set[str] = set()

    for citation in llm_answer.citations:
        chunk = by_id.get(citation.chunk_id)
        if chunk is None:
            invalid.append(InvalidCitation(citation=citation, reason="unknown_chunk"))
            continue

        quote = normalize_whitespace(citation.quote)
        if len(quote) < MIN_QUOTE_CHARS:
            invalid.append(InvalidCitation(citation=citation, reason="empty_quote"))
            continue

        if not quote_in_chunk(quote, chunk.text):
            invalid.append(InvalidCitation(citation=citation, reason="quote_not_found"))
            continue

        if citation.chunk_id in seen:
            continue
        seen.add(citation.chunk_id)

        valid.append(
            VerifiedCitation(
                chunk_id=chunk.chunk_id,
                quote=quote,
                doc=chunk.doc,
                page=chunk.page,
                heading=chunk.heading,
                chunk_text=chunk.text,
            )
        )

    if require_citations and not valid:
        return VerifiedAnswer(
            answer=REFUSAL_ANSWER,
            citations=[],
            invalid_citations=invalid,
            refused=True,
            refusal_reason="no_valid_citations",
        )

    return VerifiedAnswer(
        answer=llm_answer.answer,
        citations=valid,
        invalid_citations=invalid,
    )
