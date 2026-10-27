from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import rag_eval.api.app as app_module
from rag_eval.api.app import create_app, get_llm
from rag_eval.api.highlight import highlight_quote_html
from rag_eval.generation.citations import VerifiedCitation
from rag_eval.generation.openrouter import LLMError
from rag_eval.generation.pipeline import GenerationError, RagResponse
from rag_eval.ingest.pipeline import IngestStats
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


def make_client(pipeline: FakePipeline, **kwargs) -> TestClient:
    return TestClient(create_app(lambda hybrid, rerank: pipeline, **kwargs))


def test_ui_is_served():
    response = make_client(FakePipeline()).get("/")

    assert response.status_code == 200
    assert "RAG Eval" in response.text
    assert "Verified sources" in response.text
    assert "Latest run" in response.text
    assert "Upload and index" in response.text


def test_health_endpoint():
    response = make_client(FakePipeline()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_eval_endpoint_reports_missing_results(tmp_path):
    response = make_client(
        FakePipeline(),
        documents_dir=tmp_path / "docs",
        eval_results_path=tmp_path / "latest.json",
    ).get("/evals/latest")

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "generated_at": None,
        "settings": None,
        "best": None,
        "runs": [],
    }


def test_eval_endpoint_summarizes_runs_and_picks_best(tmp_path):
    eval_path = tmp_path / "latest.json"
    eval_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-10-27T07:49:17+00:00",
                "settings": {
                    "question_count": 5,
                    "answerable_count": 4,
                    "unanswerable_count": 1,
                    "retrieval_only": False,
                },
                "runs": [
                    {
                        "config": {"chunk_size": 400, "hybrid": False, "rerank": False},
                        "metrics": {
                            "recall_at_k": 0.4,
                            "mrr": 0.3,
                            "faithfulness": None,
                            "generation_errors": 2,
                            "judge_errors": 1,
                            "retrieval_errors": 0,
                        },
                    },
                    {
                        "config": {"chunk_size": 800, "hybrid": True, "rerank": True},
                        "metrics": {
                            "recall_at_k": 0.8,
                            "mrr": 0.6,
                            "faithfulness": 0.9,
                            "generation_errors": 0,
                            "judge_errors": 0,
                            "retrieval_errors": 0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    data = (
        make_client(
            FakePipeline(),
            documents_dir=tmp_path / "docs",
            eval_results_path=eval_path,
        )
        .get("/evals/latest")
        .json()
    )

    assert data["available"] is True
    assert data["settings"]["question_count"] == 5
    assert len(data["runs"]) == 2
    assert data["runs"][0]["errors"] == 3
    assert data["best"]["config"]["chunk_size"] == 800
    assert data["best"]["metrics"]["recall_at_k"] == 0.8


def test_documents_endpoint_lists_supported_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (docs / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    (docs / "ignored.txt").write_text("no", encoding="utf-8")

    data = make_client(FakePipeline(), documents_dir=docs).get("/documents").json()

    assert [document["name"] for document in data["documents"]] == ["notes.md", "paper.pdf"]
    assert data["total_bytes"] > 0


def test_upload_markdown_saves_file_and_reindexes(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    calls: list[tuple[str, str]] = []

    def fake_ingest(docs_dir, db_path) -> IngestStats:
        calls.append((str(docs_dir), str(db_path)))
        return IngestStats(
            documents=1,
            chunks=2,
            model="fake",
            db_path=str(db_path),
            table_name="chunks",
        )

    client = make_client(
        FakePipeline(),
        documents_dir=docs,
        db_path=tmp_path / "db",
        ingest_service=fake_ingest,
    )
    response = client.post(
        "/documents",
        params={"filename": "notes.md"},
        content=b"# Notes\n\nA new document.",
    )

    assert response.status_code == 201
    data = response.json()
    assert data["document"]["name"] == "notes.md"
    assert data["documents"] == 1
    assert data["chunks"] == 2
    assert calls == [(str(docs), str(tmp_path / "db"))]
    assert (docs / "notes.md").read_text(encoding="utf-8") == "# Notes\n\nA new document."


def test_upload_rejects_duplicate_and_unsupported_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("existing", encoding="utf-8")

    def fake_ingest(docs_dir, db_path) -> IngestStats:
        return IngestStats(
            documents=1,
            chunks=1,
            model="fake",
            db_path=str(db_path),
            table_name="chunks",
        )

    client = make_client(
        FakePipeline(),
        documents_dir=docs,
        db_path=tmp_path / "db",
        ingest_service=fake_ingest,
    )

    duplicate = client.post(
        "/documents",
        params={"filename": "notes.md"},
        content=b"# Different",
    )
    unsupported = client.post(
        "/documents",
        params={"filename": "notes.txt"},
        content=b"text",
    )

    assert duplicate.status_code == 409
    assert unsupported.status_code == 415
    assert (docs / "notes.md").read_text(encoding="utf-8") == "existing"


def test_upload_rejects_invalid_names_content_and_empty_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    client = make_client(FakePipeline(), documents_dir=docs)

    traversal = client.post(
        "/documents",
        params={"filename": "../notes.md"},
        content=b"# Notes",
    )
    invalid_pdf = client.post(
        "/documents",
        params={"filename": "paper.pdf"},
        content=b"not a pdf",
    )
    empty = client.post(
        "/documents",
        params={"filename": "empty.md"},
        content=b"",
    )

    assert traversal.status_code == 400
    assert invalid_pdf.status_code == 422
    assert empty.status_code == 400


def test_upload_removes_file_when_indexing_fails(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()

    def failing_ingest(docs_dir, db_path) -> IngestStats:
        raise ValueError("no chunks")

    client = make_client(
        FakePipeline(),
        documents_dir=docs,
        db_path=tmp_path / "db",
        ingest_service=failing_ingest,
    )
    response = client.post(
        "/documents",
        params={"filename": "empty.md"},
        content=b"# Notes",
    )

    assert response.status_code == 422
    assert not (docs / "empty.md").exists()


def test_upload_rejects_document_without_extractable_text(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    client = make_client(FakePipeline(), documents_dir=docs)

    response = client.post(
        "/documents",
        params={"filename": "blank.md"},
        content=b" \n\t\n",
    )

    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]
    assert not (docs / "blank.md").exists()


def test_api_reads_openrouter_model_on_each_call(monkeypatch):
    models = iter(["first/model", "second/model"])

    class FakeConfig:
        @classmethod
        def from_env(cls):
            return SimpleNamespace(model=next(models))

    monkeypatch.setattr(app_module, "OpenRouterConfig", FakeConfig)

    assert get_llm().model == "first/model"
    assert get_llm().model == "second/model"


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
