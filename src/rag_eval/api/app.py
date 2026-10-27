from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

from rag_eval.api.models import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    DocumentOut,
    DocumentUploadResponse,
    EvalRunOut,
    EvalSummaryOut,
)
from rag_eval.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    OpenRouterConfig,
    RetrievalConfig,
    StoreConfig,
)
from rag_eval.generation.openrouter import LLMError, OpenRouterClient
from rag_eval.generation.pipeline import GenerationError, RagPipeline
from rag_eval.ingest.embedder import Embedder
from rag_eval.ingest.loaders import SUPPORTED_SUFFIXES, discover_documents, load_document
from rag_eval.ingest.pipeline import IngestStats, ingest
from rag_eval.retrieval.retriever import Retriever

PipelineFactory = Callable[[bool, bool], RagPipeline]
DocumentIngestor = Callable[[Path, Path], IngestStats]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(EmbeddingConfig())


def get_llm() -> OpenRouterClient:
    generation = GenerationConfig()
    return OpenRouterClient(OpenRouterConfig.from_env(), generation)


@lru_cache(maxsize=4)
def get_retriever(hybrid: bool, rerank: bool) -> Retriever:
    retrieval = RetrievalConfig(hybrid=hybrid, rerank=rerank)
    return Retriever(
        StoreConfig(),
        EmbeddingConfig(),
        retrieval,
        embedder=get_embedder(),
    )


def build_pipeline(hybrid: bool, rerank: bool) -> RagPipeline:
    return RagPipeline(get_retriever(hybrid, rerank), get_llm(), GenerationConfig())


def ingest_documents(docs_dir: Path, db_path: Path) -> IngestStats:
    return ingest(
        docs_dir,
        ChunkingConfig(strategy="heading", chunk_size=800, chunk_overlap=120),
        EmbeddingConfig(),
        StoreConfig(db_path=db_path),
        embedder=get_embedder(),
    )


def _error_count(metrics: dict) -> int:
    return sum(
        value
        for key in ("retrieval_errors", "generation_errors", "judge_errors")
        if isinstance((value := metrics.get(key)), int)
    )


def _run_rank(run: EvalRunOut) -> tuple[float, float]:
    recall = run.metrics.get("recall_at_k")
    mrr = run.metrics.get("mrr")
    return (
        recall if isinstance(recall, int | float) else -1.0,
        mrr if isinstance(mrr, int | float) else -1.0,
    )


def read_eval_summary(path: Path) -> EvalSummaryOut:
    if not path.exists():
        return EvalSummaryOut(available=False)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="Could not read evaluation results.") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=503, detail="Evaluation results are not a JSON object.")

    runs: list[EvalRunOut] = []
    for raw_run in payload.get("runs", []):
        if not isinstance(raw_run, dict):
            continue
        config = raw_run.get("config")
        metrics = raw_run.get("metrics")
        if not isinstance(config, dict) or not isinstance(metrics, dict):
            continue
        runs.append(EvalRunOut(config=config, metrics=metrics, errors=_error_count(metrics)))

    generated_at = payload.get("generated_at")
    settings = payload.get("settings")
    return EvalSummaryOut(
        available=True,
        generated_at=generated_at if isinstance(generated_at, str) else None,
        settings=settings if isinstance(settings, dict) else {},
        best=max(runs, key=_run_rank) if runs else None,
        runs=runs,
    )


def _safe_document_name(raw_name: str) -> str:
    name = raw_name.strip()
    if (
        not name
        or len(name) > 200
        or name in {".", ".."}
        or name.startswith(".")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise HTTPException(status_code=400, detail="Invalid document filename.")
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PDF or Markdown file.",
        )
    return name


def _validate_document_content(name: str, content: bytes) -> None:
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if Path(name).suffix.lower() == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="The uploaded file is not a valid PDF.")
        return
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=422,
            detail="Markdown files must be UTF-8 encoded.",
        ) from error


def _document_list(path: Path) -> list[DocumentOut]:
    if not path.is_dir():
        return []
    return [
        DocumentOut(name=item.relative_to(path).as_posix(), size_bytes=item.stat().st_size)
        for item in discover_documents(path)
    ]


def create_app(
    pipeline_factory: PipelineFactory | None = None,
    *,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
    eval_results_path: Path | None = None,
    ingest_service: DocumentIngestor | None = None,
) -> FastAPI:
    app = FastAPI(title="RAG Eval", version="0.2.0")
    factory = pipeline_factory or build_pipeline
    docs_dir = documents_dir or Path("data/docs")
    database_path = db_path or StoreConfig().db_path
    eval_path = eval_results_path or Path("eval_results/latest.json")
    index_documents = ingest_service or ingest_documents
    upload_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("rag_eval.api").joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/evals/latest", response_model=EvalSummaryOut)
    def latest_eval() -> EvalSummaryOut:
        return read_eval_summary(eval_path)

    @app.get("/evals/latest/download")
    def download_eval() -> FileResponse:
        if not eval_path.is_file():
            raise HTTPException(status_code=404, detail="No evaluation results are available.")
        return FileResponse(
            eval_path,
            media_type="application/json",
            filename="latest_eval.json",
        )

    @app.get("/documents", response_model=DocumentListResponse)
    def list_documents() -> DocumentListResponse:
        documents = _document_list(docs_dir)
        return DocumentListResponse(
            documents=documents,
            total_bytes=sum(document.size_bytes for document in documents),
        )

    @app.post("/documents", response_model=DocumentUploadResponse, status_code=201)
    async def upload_document(request: Request, filename: str) -> DocumentUploadResponse:
        name = _safe_document_name(filename)
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Invalid Content-Length.") from error
            if declared_size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File is larger than 20 MB.")

        content = await request.body()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File is larger than 20 MB.")
        _validate_document_content(name, content)

        target = docs_dir / name

        def save_and_index() -> IngestStats:
            with upload_lock:
                if target.exists():
                    raise HTTPException(
                        status_code=409,
                        detail="A document with that filename already exists.",
                    )
                docs_dir.mkdir(parents=True, exist_ok=True)
                temporary = docs_dir / f".{name}.uploading"
                try:
                    temporary.write_bytes(content)
                    temporary.replace(target)
                    if not load_document(target, docs_dir):
                        raise ValueError("The document contains no extractable text.")
                    stats = index_documents(docs_dir, database_path)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    target.unlink(missing_ok=True)
                    raise
                get_retriever.cache_clear()
                return stats

        try:
            stats = await run_in_threadpool(save_and_index)
        except HTTPException:
            raise
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail=f"Could not index the document: {error}",
            ) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail="Failed to index the document.") from error

        return DocumentUploadResponse(
            document=DocumentOut(name=name, size_bytes=len(content)),
            documents=stats.documents,
            chunks=stats.chunks,
            table_name=stats.table_name,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        try:
            pipeline = factory(request.hybrid, request.rerank)
            result = pipeline.ask(request.question, top_k=request.top_k)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except LLMError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except GenerationError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        latency_ms = (time.perf_counter() - started) * 1000
        return AskResponse.from_rag_response(result, latency_ms)

    return app


app = create_app()
