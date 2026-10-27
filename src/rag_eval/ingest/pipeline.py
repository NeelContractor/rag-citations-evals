from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_eval.config import ChunkingConfig, EmbeddingConfig, StoreConfig
from rag_eval.ingest.chunking import chunk_document
from rag_eval.ingest.embedder import Embedder
from rag_eval.ingest.loaders import discover_documents, load_document
from rag_eval.ingest.store import ChunkStore
from rag_eval.models import Chunk


@dataclass(frozen=True, slots=True)
class IngestStats:
    documents: int
    chunks: int
    model: str
    db_path: str
    table_name: str


def ingest(
    docs_dir: Path,
    chunking: ChunkingConfig,
    embedding: EmbeddingConfig,
    store: StoreConfig,
    embedder: Embedder | None = None,
) -> IngestStats:
    files = discover_documents(docs_dir)
    if not files:
        raise FileNotFoundError(
            f"No supported documents found in {docs_dir}. Add PDFs or markdown files."
        )

    chunks: list[Chunk] = []
    for path in files:
        start_index = 0
        for raw in load_document(path, docs_dir):
            document_chunks = chunk_document(raw, chunking, start_index=start_index)
            start_index += len(document_chunks)
            chunks.extend(document_chunks)

    if not chunks:
        raise ValueError("Documents were found but produced no chunks")

    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id values generated")

    vectors = (embedder or Embedder(embedding)).embed_documents(
        [chunk.embedding_text for chunk in chunks]
    )
    ChunkStore(store).replace(chunks, vectors)

    return IngestStats(
        documents=len(files),
        chunks=len(chunks),
        model=embedding.model_name,
        db_path=str(store.db_path),
        table_name=store.table_name,
    )
