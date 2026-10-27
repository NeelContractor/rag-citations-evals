from __future__ import annotations

import json

import pytest

from rag_eval.config import GenerationConfig
from rag_eval.generation.citations import REFUSAL_ANSWER
from rag_eval.generation.output import OutputParseError, parse_llm_answer
from rag_eval.generation.pipeline import GenerationError, RagPipeline
from rag_eval.generation.prompts import build_user_prompt
from rag_eval.retrieval.types import RetrievedChunk


def make_chunk(chunk_id: str = "c1", text: str = "Retrieval combines search and generation."):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc="guide.md",
        page=1,
        heading="Intro",
        text=text,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks

    def search(self, question: str) -> list[RetrievedChunk]:
        return self.chunks


class FakeLLM:
    model = "fake-model"

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.responses.pop(0)


def valid_response() -> str:
    return json.dumps(
        {
            "answer": "Retrieval combines search and generation.",
            "citations": [{"chunk_id": "c1", "quote": "combines search and generation"}],
        }
    )


def test_prompt_numbers_sources_and_includes_chunk_ids():
    prompt = build_user_prompt(
        "What is retrieval?", [make_chunk(), make_chunk("c2", "Other text.")]
    )

    assert "[1] chunk_id: c1" in prompt
    assert "[2] chunk_id: c2" in prompt
    assert "Question: What is retrieval?" in prompt


def test_parse_plain_json():
    parsed = parse_llm_answer(
        '{"answer": "ok", "citations": [{"chunk_id": "c1", "quote": "search"}]}'
    )

    assert parsed.answer == "ok"
    assert parsed.citations[0].chunk_id == "c1"


def test_parse_json_inside_markdown_fence():
    raw = '```json\n{"answer": "ok", "citations": []}\n```'
    parsed = parse_llm_answer(raw)

    assert parsed.answer == "ok"


def test_parse_rejects_invalid_json():
    with pytest.raises(OutputParseError):
        parse_llm_answer("I cannot produce JSON today")


def test_pipeline_returns_verified_citations():
    llm = FakeLLM([valid_response()])
    pipeline = RagPipeline(FakeRetriever([make_chunk()]), llm)

    response = pipeline.ask("What is retrieval?")

    assert not response.refused
    assert response.answer == "Retrieval combines search and generation."
    assert len(response.citations) == 1
    assert response.citations[0].quote == "combines search and generation"
    assert response.invalid_citations == []


def test_pipeline_refuses_when_citations_are_invalid():
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "answer": "Made up answer.",
                    "citations": [{"chunk_id": "c1", "quote": "absent quote"}],
                }
            )
        ]
    )
    pipeline = RagPipeline(FakeRetriever([make_chunk()]), llm)

    response = pipeline.ask("What is retrieval?")

    assert response.refused
    assert response.refusal_reason == "no_valid_citations"
    assert response.answer == REFUSAL_ANSWER
    assert response.invalid_citations[0].reason == "quote_not_found"


def test_pipeline_refuses_without_retrieved_context():
    llm = FakeLLM([])
    pipeline = RagPipeline(FakeRetriever([]), llm)

    response = pipeline.ask("What is retrieval?")

    assert response.refused
    assert response.refusal_reason == "no_context"
    assert llm.prompts == []


def test_pipeline_retries_after_invalid_json():
    llm = FakeLLM(["not json", valid_response()])
    pipeline = RagPipeline(
        FakeRetriever([make_chunk()]),
        llm,
        GenerationConfig(max_retries=1),
    )

    response = pipeline.ask("What is retrieval?")

    assert not response.refused
    assert len(llm.prompts) == 2


def test_pipeline_raises_after_retries_are_exhausted():
    llm = FakeLLM(["not json", "still not json"])
    pipeline = RagPipeline(
        FakeRetriever([make_chunk()]),
        llm,
        GenerationConfig(max_retries=1),
    )

    with pytest.raises(GenerationError):
        pipeline.ask("What is retrieval?")


def test_pipeline_keeps_valid_and_flags_invalid_citations():
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "answer": "Retrieval combines search and generation.",
                    "citations": [
                        {"chunk_id": "c1", "quote": "combines search and generation"},
                        {"chunk_id": "c1", "quote": "not present"},
                    ],
                }
            )
        ]
    )
    pipeline = RagPipeline(FakeRetriever([make_chunk()]), llm)

    response = pipeline.ask("What is retrieval?")

    assert not response.refused
    assert len(response.citations) == 1
    assert len(response.invalid_citations) == 1
