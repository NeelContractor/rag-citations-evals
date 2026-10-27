from __future__ import annotations

import json

import pytest

from rag_eval.eval.generate_golden import (
    _items_from_response,
    build_golden,
    first_sentence,
    heuristic_item,
)
from rag_eval.retrieval.types import RetrievedChunk


def make_chunk(
    chunk_id: str,
    text: str = "Retrieval-augmented generation combines search with generation. It cites sources.",
    heading: str = "Intro",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc="guide.md",
        page=None,
        heading=heading,
        text=text,
    )


def test_first_sentence_returns_first_long_sentence():
    assert first_sentence("Short. This is the first sufficiently long sentence here.") == (
        "This is the first sufficiently long sentence here."
    )


def test_heuristic_item_uses_heading_and_exact_quote():
    chunk = make_chunk("c1")
    item = heuristic_item(chunk)

    assert item.answerable
    assert item.source_chunk_ids == ["c1"]
    assert item.evidence[0].quote in chunk.text
    assert "Intro" in item.question


def test_build_golden_adds_unanswerable_items_and_ids():
    chunks = [
        make_chunk("c1"),
        make_chunk(
            "c2",
            text=(
                "Second chunk has other material here that is sufficiently long "
                "for evaluation purposes."
            ),
            heading="Other",
        ),
    ]

    items = build_golden(chunks, count=2, unanswerable_count=2, llm=None)

    assert len(items) == 4
    assert [item.id for item in items] == ["q001", "q002", "q003", "q004"]
    assert sum(1 for item in items if item.answerable) == 2
    assert sum(1 for item in items if not item.answerable) == 2


def test_build_golden_deduplicates_questions():
    chunk = make_chunk("c1")
    chunks = [chunk, chunk]

    items = build_golden(chunks, count=2, unanswerable_count=0, llm=None)

    assert len(items) == 1


def test_items_from_llm_response_accepts_valid_quote():
    chunk = make_chunk("c1")
    raw = json.dumps(
        {
            "items": [
                {
                    "chunk_id": "c1",
                    "question": "What does RAG combine?",
                    "reference_answer": "Search and generation.",
                    "evidence_quote": "combines search with generation",
                }
            ]
        }
    )

    items = _items_from_response(raw, [chunk])

    assert list(items) == ["c1"]
    assert items["c1"].question == "What does RAG combine?"


def test_items_from_llm_response_rejects_paraphrased_quote():
    chunk = make_chunk("c1")
    raw = json.dumps(
        {
            "items": [
                {
                    "chunk_id": "c1",
                    "question": "What does RAG combine?",
                    "reference_answer": "Search and generation.",
                    "evidence_quote": "RAG merges search and generation",
                }
            ]
        }
    )

    assert _items_from_response(raw, [chunk]) == {}


def test_items_from_llm_response_ignores_unknown_chunk_id():
    chunk = make_chunk("c1")
    raw = json.dumps(
        {
            "items": [
                {
                    "chunk_id": "missing",
                    "question": "What?",
                    "reference_answer": "Something.",
                    "evidence_quote": "combines search with generation",
                }
            ]
        }
    )

    assert _items_from_response(raw, [chunk]) == {}


def test_items_from_response_requires_items_array():
    with pytest.raises(ValueError):
        _items_from_response('{"questions": []}', [make_chunk("c1")])
