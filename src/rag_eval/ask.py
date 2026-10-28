from __future__ import annotations

import argparse
import json

from rag_eval.config import (
    EmbeddingConfig,
    GenerationConfig,
    LLMConfig,
    RetrievalConfig,
    StoreConfig,
)
from rag_eval.generation.openrouter import LLMClient, LLMError
from rag_eval.generation.pipeline import GenerationError, RagPipeline
from rag_eval.retrieval.retriever import Retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question with cited sources")
    parser.add_argument("question")
    parser.add_argument("--db", default=".lancedb")
    parser.add_argument("--table", default="chunks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        llm_config = LLMConfig.from_env()
        if args.model:
            llm_config = llm_config.model_copy(update={"model": args.model})
        retrieval = RetrievalConfig(
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            hybrid=args.hybrid,
            rerank=args.rerank,
        )
        generation = GenerationConfig()
        store = StoreConfig(db_path=args.db, table_name=args.table)
        embedding = EmbeddingConfig()

        retriever = Retriever(store, embedding, retrieval)
        pipeline = RagPipeline(
            retriever,
            LLMClient(llm_config, generation),
            generation,
        )
        response = pipeline.ask(args.question)
    except (FileNotFoundError, ValueError, LLMError, GenerationError) as error:
        raise SystemExit(str(error)) from error

    if args.json:
        print(json.dumps(response.model_dump(), indent=2))
        return 0

    print(response.answer)
    if response.refused:
        print(f"\nRefused: {response.refusal_reason}")
    if response.citations:
        print("\nSources:")
        for citation in response.citations:
            page = f", page {citation.page}" if citation.page else ""
            print(f'- {citation.doc}{page}: "{citation.quote}"')
    if response.invalid_citations:
        print(f"\nDropped {len(response.invalid_citations)} invalid citation(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
