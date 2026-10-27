from __future__ import annotations

import pytest

from rag_eval.eval.judge import FaithfulnessJudge, JudgeError
from rag_eval.generation.citations import REFUSAL_ANSWER
from rag_eval.retrieval.types import RetrievedChunk


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc="guide.md",
        text="Retrieval combines search and generation.",
    )


def test_judge_parses_valid_faithfulness_output():
    llm = FakeLLM('{"faithful": true, "score": 1.0, "unsupported_claims": []}')
    judgement = FaithfulnessJudge(llm).judge("What is RAG?", "It combines search.", [make_chunk()])

    assert judgement.faithful
    assert judgement.score == 1.0
    assert len(llm.calls) == 1


def test_judge_parses_partial_score_and_unsupported_claims():
    llm = FakeLLM('{"faithful": false, "score": 0.5, "unsupported_claims": ["unsupported claim"]}')
    judgement = FaithfulnessJudge(llm).judge("What is RAG?", "Mixed answer.", [make_chunk()])

    assert not judgement.faithful
    assert judgement.score == 0.5
    assert judgement.unsupported_claims == ["unsupported claim"]


def test_judge_short_circuits_on_refusal():
    llm = FakeLLM("not used")
    judgement = FaithfulnessJudge(llm).judge("What is RAG?", REFUSAL_ANSWER, [make_chunk()])

    assert judgement.faithful
    assert judgement.score == 1.0
    assert llm.calls == []


def test_judge_raises_on_invalid_json():
    llm = FakeLLM("not json")

    with pytest.raises(JudgeError):
        FaithfulnessJudge(llm).judge("What is RAG?", "Answer.", [make_chunk()])


def test_judge_raises_on_invalid_schema():
    llm = FakeLLM('{"faithful": true, "score": 2.0}')

    with pytest.raises(JudgeError):
        FaithfulnessJudge(llm).judge("What is RAG?", "Answer.", [make_chunk()])
