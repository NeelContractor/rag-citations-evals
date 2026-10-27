from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from rag_eval.config import ChunkingConfig, EmbeddingConfig, StoreConfig
from rag_eval.ingest.pipeline import ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDFs and markdown into LanceDB")
    parser.add_argument("--docs", type=Path, default=Path("data/docs"))
    parser.add_argument("--db", type=Path, default=Path(".lancedb"))
    parser.add_argument("--table", default="chunks")
    parser.add_argument("--strategy", choices=["fixed", "heading"], default="heading")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/fastembed"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        chunking = ChunkingConfig(
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        embedding = EmbeddingConfig(
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
        )
        store = StoreConfig(db_path=args.db, table_name=args.table)
    except ValidationError as error:
        raise SystemExit(f"Invalid configuration:\n{error}") from error

    try:
        stats = ingest(args.docs, chunking, embedding, store)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(
        f"Ingested {stats.documents} document(s) into {stats.chunks} chunk(s) "
        f"using {stats.model} -> {stats.db_path}/{stats.table_name}"
    )
    return 0
