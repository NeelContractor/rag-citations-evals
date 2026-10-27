from __future__ import annotations

import time
from collections.abc import Callable
from functools import lru_cache
from importlib.resources import files

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from rag_eval.api.models import AskRequest, AskResponse
from rag_eval.config import (
    EmbeddingConfig,
    GenerationConfig,
    OpenRouterConfig,
    RetrievalConfig,
    StoreConfig,
)
from rag_eval.generation.openrouter import LLMError, OpenRouterClient
from rag_eval.generation.pipeline import GenerationError, RagPipeline
from rag_eval.ingest.embedder import Embedder
from rag_eval.retrieval.retriever import Retriever

PipelineFactory = Callable[[bool, bool], RagPipeline]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(EmbeddingConfig())


@lru_cache(maxsize=1)
def get_llm() -> OpenRouterClient:
    generation = GenerationConfig()
    return OpenRouterClient(OpenRouterConfig.from_env(), generation)


@lru_cache(maxsize=4)
def build_pipeline(hybrid: bool, rerank: bool) -> RagPipeline:
    generation = GenerationConfig()
    retrieval = RetrievalConfig(hybrid=hybrid, rerank=rerank)
    retriever = Retriever(
        StoreConfig(),
        EmbeddingConfig(),
        retrieval,
        embedder=get_embedder(),
    )
    return RagPipeline(retriever, get_llm(), generation)


def create_app(pipeline_factory: PipelineFactory | None = None) -> FastAPI:
    app = FastAPI(title="RAG Eval", version="0.1.0")
    factory = pipeline_factory or build_pipeline

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("rag_eval.api").joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
