from __future__ import annotations

from fastapi.testclient import TestClient

from rag_eval.api.app import create_app
from rag_eval.api.highlight import highlight_quote_html
from rag_eval.generation.citations import VerifiedCitation
from rag_eval.generation.openrouter import LLMError
from rag_eval.generation.pipeline import GenerationError, RagResponse
from rag_eval.retrieval.types import RetrievedChunk

CHUNK_TEXT = "Safe <script>alert(1)</script> retrieval combines search and generation."


def make_response() -> RagResponse:
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc="guide.md",
        page=2,
        heading="Intro",
        text=CHUNK_TEXT,
        score=0.91,
    )
    citation = VerifiedCitation(
        chunk_id="c1",
        quote="Safe <script>alert(1)</script> retrieval",
        doc="guide.md",
        page=2,
        heading="Intro",
        chunk_text=CHUNK_TEXT,
    )
    return RagResponse(
        question="What is retrieval?",
        answer="Retrieval combines search and generation.",
        citations=[citation],
        invalid_citations=[],
        retrieved=[chunk],
        model="fake-model",
    )


class FakePipeline:
    def __init__(self, response: RagResponse | None = None, error: Exception | None = None):
        self.response = response or make_response()
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def ask(self, question: str, top_k: int | None = None) -> RagResponse:
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        return self.response


def make_client(pipeline: FakePipeline) -> TestClient:
    return TestClient(create_app(lambda hybrid, rerank: pipeline))


def test_ui_is_served():
    response = make_client(FakePipeline()).get("/")

    assert response.status_code == 200
    assert "RAG Eval" in response.text
    assert "Verified sources" in response.text


def test_health_endpoint():
    response = make_client(FakePipeline()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_highlighted_sources():
    response = make_client(FakePipeline()).post(
        "/ask",
        json={"question": "What is retrieval?", "top_k": 3, "hybrid": True, "rerank": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Retrieval combines search and generation."
    assert data["model"] == "fake-model"
    assert data["refused"] is False
    assert data["latency_ms"] >= 0
    assert len(data["sources"]) == 1

    source = data["sources"][0]
    assert source["chunk_id"] == "c1"
    assert source["doc"] == "guide.md"
    assert source["page"] == 2
    assert source["chunk_text"] == CHUNK_TEXT
    assert source["score"] == 0.91
    assert "<mark>" in source["highlighted_html"]
    assert "&lt;script&gt;" in source["highlighted_html"]
    assert "<script>" not in source["highlighted_html"]


def test_ask_passes_question_top_k_and_flags():
    pipeline = FakePipeline()
    seen_flags: list[tuple[bool, bool]] = []
    client = TestClient(
        create_app(lambda hybrid, rerank: (seen_flags.append((hybrid, rerank)), pipeline)[1])
    )

    response = client.post(
        "/ask",
        json={"question": "  What is retrieval?  ", "top_k": 7, "hybrid": True},
    )

    assert response.status_code == 200
    assert pipeline.calls == [("What is retrieval?", 7)]
    assert seen_flags == [(True, False)]


def test_blank_question_is_rejected():
    response = make_client(FakePipeline()).post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_top_k_out_of_range_is_rejected():
    response = make_client(FakePipeline()).post(
        "/ask", json={"question": "What is retrieval?", "top_k": 100}
    )

    assert response.status_code == 422


def test_missing_api_key_returns_503():
    client = make_client(FakePipeline(error=ValueError("OPENROUTER_API_KEY is not set")))

    response = client.post("/ask", json={"question": "What is retrieval?"})

    assert response.status_code == 503
    assert "OPENROUTER_API_KEY" in response.json()["detail"]


def test_missing_index_returns_503():
    client = make_client(FakePipeline(error=FileNotFoundError("Run `make ingest` first.")))

    response = client.post("/ask", json={"question": "What is retrieval?"})

    assert response.status_code == 503
    assert "make ingest" in response.json()["detail"]


def test_llm_error_returns_502():
    client = make_client(FakePipeline(error=LLMError("rate limited")))

    response = client.post("/ask", json={"question": "What is retrieval?"})

    assert response.status_code == 502
    assert "rate limited" in response.json()["detail"]


def test_generation_error_returns_502():
    client = make_client(FakePipeline(error=GenerationError("invalid JSON")))

    response = client.post("/ask", json={"question": "What is retrieval?"})

    assert response.status_code == 502
    assert "invalid JSON" in response.json()["detail"]


def test_highlight_marks_case_insensitive_match():
    highlighted = highlight_quote_html("Retrieval combines search.", "retrieval combines")

    assert highlighted == "<mark>Retrieval combines</mark> search."


def test_highlight_normalizes_whitespace():
    highlighted = highlight_quote_html("one two\nthree four", "two three")

    assert "<mark>two\nthree</mark>" in highlighted


def test_highlight_matches_pdf_hyphenated_line_break():
    highlighted = highlight_quote_html("model: bge-reranker-\nv2-m3.", "bge-reranker-v2-m3")

    assert "<mark>bge-reranker-\nv2-m3</mark>" in highlighted


def test_highlight_escapes_html_when_quote_is_missing():
    highlighted = highlight_quote_html("<b>safe</b>", "absent")

    assert highlighted == "&lt;b&gt;safe&lt;/b&gt;"
    assert "<b>" not in highlighted
