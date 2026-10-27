from __future__ import annotations

from rag_eval.generation.citations import (
    REFUSAL_ANSWER,
    Citation,
    LLMAnswer,
    is_refusal,
    normalize_whitespace,
    quote_in_chunk,
    verify_citations,
)
from rag_eval.retrieval.types import RetrievedChunk


def chunk(
    chunk_id: str = "c1",
    text: str = "Retrieval-augmented generation combines retrieval and generation.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc="guide.md",
        page=2,
        heading="Intro",
        text=text,
    )


def test_quote_matches_after_whitespace_normalization():
    assert quote_in_chunk("combines   retrieval\nand", chunk().text)


def test_quote_matching_is_case_insensitive():
    assert quote_in_chunk("RETRIEVAL-AUGMENTED GENERATION", chunk().text)


def test_paraphrased_quote_is_rejected():
    assert not quote_in_chunk("retrieval makes generation better", chunk().text)


def test_valid_citation_is_kept():
    answer = LLMAnswer(
        answer="RAG combines retrieval and generation.",
        citations=[Citation(chunk_id="c1", quote="combines retrieval and generation")],
    )
    verified = verify_citations(answer, [chunk()])

    assert not verified.refused
    assert len(verified.citations) == 1
    assert verified.citations[0].doc == "guide.md"
    assert verified.citations[0].page == 2
    assert verified.invalid_citations == []


def test_unknown_chunk_is_flagged():
    answer = LLMAnswer(
        answer="RAG combines retrieval and generation.",
        citations=[Citation(chunk_id="missing", quote="retrieval")],
    )
    verified = verify_citations(answer, [chunk()])

    assert verified.refused
    assert verified.refusal_reason == "no_valid_citations"
    assert verified.invalid_citations[0].reason == "unknown_chunk"


def test_quote_not_in_cited_chunk_is_flagged():
    answer = LLMAnswer(
        answer="RAG combines retrieval and generation.",
        citations=[Citation(chunk_id="c1", quote="this text is absent")],
    )
    verified = verify_citations(answer, [chunk()])

    assert verified.refused
    assert verified.invalid_citations[0].reason == "quote_not_found"


def test_empty_quote_is_flagged():
    answer = LLMAnswer(
        answer="RAG combines retrieval and generation.",
        citations=[Citation(chunk_id="c1", quote="  ")],
    )
    verified = verify_citations(answer, [chunk()])

    assert verified.invalid_citations[0].reason == "empty_quote"


def test_mixed_citations_keep_valid_and_flag_invalid():
    answer = LLMAnswer(
        answer="RAG combines retrieval and generation.",
        citations=[
            Citation(chunk_id="c1", quote="combines retrieval and generation"),
            Citation(chunk_id="c1", quote="not in the chunk"),
            Citation(chunk_id="nope", quote="combines retrieval"),
        ],
    )
    verified = verify_citations(answer, [chunk()])

    assert not verified.refused
    assert len(verified.citations) == 1
    assert len(verified.invalid_citations) == 2


def test_duplicate_valid_citations_are_deduplicated():
    answer = LLMAnswer(
        answer="RAG combines retrieval and generation.",
        citations=[
            Citation(chunk_id="c1", quote="combines retrieval and generation"),
            Citation(chunk_id="c1", quote="retrieval and generation"),
        ],
    )
    verified = verify_citations(answer, [chunk()])

    assert len(verified.citations) == 1


def test_model_refusal_clears_citations():
    answer = LLMAnswer(
        answer=REFUSAL_ANSWER,
        citations=[Citation(chunk_id="c1", quote="retrieval")],
    )
    verified = verify_citations(answer, [chunk()])

    assert verified.refused
    assert verified.refusal_reason == "model_refusal"
    assert verified.citations == []


def test_refusal_is_detected_from_paraphrase():
    assert is_refusal("I don't have enough information to answer that.")
    assert is_refusal("Insufficient information in the sources.")
    assert not is_refusal("RAG combines retrieval and generation.")


def test_require_citations_false_allows_uncited_answer():
    answer = LLMAnswer(answer="RAG combines retrieval and generation.", citations=[])
    verified = verify_citations(answer, [chunk()], require_citations=False)

    assert not verified.refused
    assert verified.answer == "RAG combines retrieval and generation."


def test_normalize_whitespace_collapses_all_gaps():
    assert normalize_whitespace(" a\n\tb   c ") == "a b c"
