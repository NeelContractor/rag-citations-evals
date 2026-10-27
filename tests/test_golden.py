from __future__ import annotations

import pytest

from rag_eval.eval.golden import (
    Evidence,
    GoldenItem,
    load_golden,
    map_evidence_to_chunks,
    map_gold_chunks,
    save_golden,
)
from rag_eval.retrieval.types import RetrievedChunk


def make_chunk(
    chunk_id: str,
    doc: str = "guide.md",
    page: int | None = None,
    text: str = "Retrieval combines search and generation.",
) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc=doc, page=page, text=text)


def test_answerable_item_requires_evidence():
    with pytest.raises(ValueError):
        GoldenItem(id="q1", question="What is RAG?", answerable=True)


def test_unanswerable_item_can_have_no_evidence():
    item = GoldenItem(id="q1", question="What is the capital of France?", answerable=False)

    assert item.evidence == []


def test_save_and_load_roundtrip(tmp_path):
    items = [
        GoldenItem(
            id="q1",
            question="What is retrieval?",
            reference_answer="Search plus generation.",
            evidence=[Evidence(doc="guide.md", quote="Retrieval combines search")],
            source_chunk_ids=["c1"],
        ),
        GoldenItem(id="q2", question="Unknown?", answerable=False),
    ]
    path = tmp_path / "golden.jsonl"

    save_golden(path, items)
    loaded = load_golden(path)

    assert loaded == items


def test_load_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id":"q1","question":"What is RAG?","evidence":[{"doc":"a.md","quote":"abc"}]}\n'
        '{"id":"q1","question":"Again?","evidence":[{"doc":"a.md","quote":"abc"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate id"):
        load_golden(path)


def test_map_evidence_exact_match():
    chunks = [make_chunk("c1"), make_chunk("c2", text="Something else entirely.")]
    evidence = Evidence(doc="guide.md", quote="combines search and generation")

    assert map_evidence_to_chunks(evidence, chunks) == {"c1"}


def test_map_evidence_requires_page_match():
    chunks = [
        make_chunk("c1", doc="paper.pdf", page=2, text="The answer is here."),
        make_chunk("c2", doc="paper.pdf", page=3, text="The answer is here."),
    ]
    evidence = Evidence(doc="paper.pdf", page=3, quote="The answer is here.")

    assert map_evidence_to_chunks(evidence, chunks) == {"c2"}


def test_map_evidence_falls_back_to_token_coverage():
    chunks = [
        make_chunk("c1", text="generation retrieval combines search"),
        make_chunk("c2", text="unrelated material"),
    ]
    evidence = Evidence(doc="guide.md", quote="retrieval combines search and generation")

    assert map_evidence_to_chunks(evidence, chunks) == {"c1"}


def test_map_evidence_matches_across_pdf_hyphenated_line_break():
    chunks = [
        make_chunk(
            "c1",
            doc="paper.pdf",
            page=2,
            text="Dense: Qwen3-Embedding-8B; BGE: bge-reranker-\nv2-m3.",
        )
    ]
    evidence = Evidence(
        doc="paper.pdf",
        page=2,
        quote="Dense: Qwen3-Embedding-8B; BGE: bge-reranker-v2-m3.",
    )

    assert map_evidence_to_chunks(evidence, chunks) == {"c1"}


def test_map_evidence_rejects_low_coverage():
    chunks = [make_chunk("c1", text="only retrieval appears here")]
    evidence = Evidence(doc="guide.md", quote="retrieval combines search and generation")

    assert map_evidence_to_chunks(evidence, chunks) == set()


def test_map_gold_chunks_combines_evidence_and_skips_unanswerable():
    chunks = [
        make_chunk("c1", text="First fact."),
        make_chunk("c2", text="Second fact."),
    ]
    item = GoldenItem(
        id="q1",
        question="What are the facts?",
        evidence=[
            Evidence(doc="guide.md", quote="First fact."),
            Evidence(doc="guide.md", quote="Second fact."),
        ],
    )
    unanswerable = GoldenItem(id="q2", question="Unknown?", answerable=False)

    assert map_gold_chunks(item, chunks) == {"c1", "c2"}
    assert map_gold_chunks(unanswerable, chunks) == set()
