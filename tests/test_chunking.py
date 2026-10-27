from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_eval.config import ChunkingConfig
from rag_eval.ingest.chunking import chunk_document
from rag_eval.models import RawDocument


def test_fixed_chunks_overlap_at_character_level():
    text = "".join(f"{index:04d}" for index in range(250))
    config = ChunkingConfig(strategy="fixed", chunk_size=200, chunk_overlap=50)
    chunks = chunk_document(RawDocument(doc="a.md", page=None, text=text), config)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    assert chunks[0].text[0] == text[0]
    assert chunks[-1].text[-1] == text[-1]


def test_fixed_chunks_cover_words_without_mid_word_starts():
    words = [f"word{index}" for index in range(300)]
    text = " ".join(words)
    config = ChunkingConfig(strategy="fixed", chunk_size=120, chunk_overlap=30)
    chunks = chunk_document(RawDocument(doc="a.md", page=None, text=text), config)

    seen = set(" ".join(chunk.text for chunk in chunks).split())
    assert set(words) <= seen
    assert all(chunk.text == chunk.text.strip() for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_fixed_chunks_keep_page_metadata():
    text = "Page content. " * 50
    config = ChunkingConfig(strategy="heading", chunk_size=100, chunk_overlap=20)
    chunks = chunk_document(RawDocument(doc="paper.pdf", page=3, text=text), config)

    assert chunks
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.heading is None for chunk in chunks)
    assert all(chunk.chunk_id.startswith("paper-pdf-p3-") for chunk in chunks)


def test_heading_aware_chunks_carry_heading_path():
    markdown = (
        "# Intro\n\nIntro text.\n\n"
        "## Setup\n\nInstall it.\n\n"
        "### macOS\n\nUse brew.\n\n"
        "## Usage\n\nRun it.\n"
    )
    config = ChunkingConfig(strategy="heading", chunk_size=500, chunk_overlap=50)
    chunks = chunk_document(RawDocument(doc="guide.md", page=None, text=markdown), config)

    assert [chunk.heading for chunk in chunks] == [
        "Intro",
        "Intro > Setup",
        "Intro > Setup > macOS",
        "Intro > Usage",
    ]
    assert chunks[0].text == "Intro text."
    assert chunks[1].text == "Install it."
    assert chunks[2].text == "Use brew."
    assert chunks[3].text == "Run it."


def test_heading_aware_splits_large_sections_with_overlap():
    body = "x" * 1000
    markdown = f"## Big section\n\n{body}\n"
    config = ChunkingConfig(strategy="heading", chunk_size=200, chunk_overlap=20)
    chunks = chunk_document(RawDocument(doc="guide.md", page=None, text=markdown), config)

    assert len(chunks) == 6
    assert all(chunk.heading == "Big section" for chunk in chunks)
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert chunks[0].text[-20:] == chunks[1].text[:20]
    assert chunks[4].text[-20:] == chunks[5].text[:20]


def test_heading_aware_ignores_headings_inside_code_fences():
    markdown = "# One\n\n```bash\n# not a heading\n```\n\n## Two\n\nbody\n"
    config = ChunkingConfig(strategy="heading", chunk_size=500, chunk_overlap=50)
    chunks = chunk_document(RawDocument(doc="guide.md", page=None, text=markdown), config)

    assert [chunk.heading for chunk in chunks] == ["One", "One > Two"]
    assert "# not a heading" in chunks[0].text


def test_heading_strategy_without_headings_returns_plain_chunk():
    config = ChunkingConfig(strategy="heading", chunk_size=500, chunk_overlap=50)
    chunks = chunk_document(
        RawDocument(doc="notes.md", page=None, text="Plain text without headings."),
        config,
    )

    assert len(chunks) == 1
    assert chunks[0].heading is None
    assert chunks[0].text == "Plain text without headings."


def test_chunk_ids_are_stable_and_unique():
    text = "alpha beta gamma delta epsilon " * 20
    config = ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=20)
    raw = RawDocument(doc="docs/a.md", page=None, text=text)

    first = chunk_document(raw, config)
    second = chunk_document(raw, config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_start_index_is_respected():
    config = ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=10)
    raw = RawDocument(doc="a.md", page=None, text="some text " * 30)
    chunks = chunk_document(raw, config, start_index=7)

    assert chunks[0].chunk_index == 7


def test_empty_text_produces_no_chunks():
    config = ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=10)
    chunks = chunk_document(RawDocument(doc="a.md", page=None, text="   \n\n  "), config)

    assert chunks == []


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValidationError):
        ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=100)
