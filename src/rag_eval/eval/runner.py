from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_eval.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    OpenRouterConfig,
    RetrievalConfig,
    StoreConfig,
)
from rag_eval.eval.golden import GoldenItem, load_chunks, load_golden, map_gold_chunks
from rag_eval.eval.judge import FaithfulnessJudge, JudgeError
from rag_eval.eval.metrics import (
    citation_precision,
    citation_recall,
    citation_validity,
    format_metric,
    format_table,
    mean_or_none,
    recall_at_k,
    reciprocal_rank,
    refusal_rate,
)
from rag_eval.generation.openrouter import LLMError, OpenRouterClient
from rag_eval.generation.pipeline import GenerationError, RagPipeline
from rag_eval.ingest.embedder import Embedder
from rag_eval.ingest.pipeline import ingest
from rag_eval.ingest.store import ChunkStore
from rag_eval.retrieval.reranker import LateInteractionReranker
from rag_eval.retrieval.retriever import Retriever


@dataclass(frozen=True, slots=True)
class RunConfig:
    chunk_size: int
    overlap: int
    strategy: str
    hybrid: bool
    rerank: bool

    @property
    def table_name(self) -> str:
        return f"eval_{self.strategy}_{self.chunk_size}_{self.overlap}"


def run_evaluation(
    docs_dir: Path,
    golden_path: Path,
    db_path: Path,
    chunk_sizes: list[int],
    overlap: int,
    strategy: str,
    top_k: int,
    candidate_k: int,
    output_dir: Path,
    retrieval_only: bool = False,
    judge_model: str | None = None,
    limit: int | None = None,
    reuse_index: bool = False,
    log=print,
) -> dict:
    items = load_golden(golden_path)
    if limit is not None:
        items = items[:limit]

    embedding = EmbeddingConfig()
    embedder = Embedder(embedding)
    reranker = LateInteractionReranker(RetrievalConfig().rerank_model, embedding.cache_dir)
    generation = GenerationConfig()

    llm: OpenRouterClient | None = None
    judge: FaithfulnessJudge | None = None
    if not retrieval_only:
        openrouter = OpenRouterConfig.from_env()
        llm = OpenRouterClient(openrouter, generation)
        judge_config = (
            openrouter.model_copy(update={"model": judge_model}) if judge_model else openrouter
        )
        judge = FaithfulnessJudge(
            OpenRouterClient(
                judge_config,
                GenerationConfig(temperature=0.0, max_tokens=500),
            )
        )

    log(f"Loaded {len(items)} golden items from {golden_path}")
    mappings: dict[int, dict[str, set[str]]] = {}
    runs: list[dict] = []

    for chunk_size in chunk_sizes:
        table_name = RunConfig(chunk_size, overlap, strategy, False, False).table_name
        store = StoreConfig(db_path=db_path, table_name=table_name)
        if reuse_index and ChunkStore(store).exists():
            log(f"Reusing index {table_name}")
        else:
            stats = ingest(
                docs_dir,
                ChunkingConfig(
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                ),
                embedding,
                store,
                embedder=embedder,
            )
            log(
                f"Ingested {stats.documents} document(s) into {stats.chunks} chunk(s) "
                f"for chunk_size={chunk_size} ({table_name})"
            )

        chunks = load_chunks(db_path, table_name)
        mappings[chunk_size] = {item.id: map_gold_chunks(item, chunks) for item in items}
        unmapped = [
            item.id for item in items if item.answerable and not mappings[chunk_size][item.id]
        ]
        if unmapped:
            log(
                f"Warning: {len(unmapped)} answerable golden item(s) could not be mapped "
                f"to any chunk at chunk_size={chunk_size}: {', '.join(unmapped)}"
            )

        for hybrid in (False, True):
            for rerank in (False, True):
                config = RunConfig(chunk_size, overlap, strategy, hybrid, rerank)
                run = _run_config(
                    config,
                    items,
                    mappings[chunk_size],
                    store,
                    embedding,
                    embedder,
                    reranker,
                    llm,
                    judge,
                    generation,
                    top_k,
                    candidate_k,
                    retrieval_only,
                    log,
                )
                runs.append(run)
                _log_metrics(run, log)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "settings": {
            "docs_dir": str(docs_dir),
            "golden_path": str(golden_path),
            "db_path": str(db_path),
            "chunk_sizes": chunk_sizes,
            "overlap": overlap,
            "strategy": strategy,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "retrieval_only": retrieval_only,
            "judge_model": judge_model,
            "question_count": len(items),
            "answerable_count": sum(1 for item in items if item.answerable),
            "unanswerable_count": sum(1 for item in items if not item.answerable),
        },
        "runs": runs,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"eval_{timestamp}.json"
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    output_path.write_text(payload, encoding="utf-8")
    (output_dir / "latest.json").write_text(payload, encoding="utf-8")

    log("")
    log(format_results_table(runs, retrieval_only))
    log("")
    log(f"Saved results to {output_path}")
    return result


def _run_config(
    config: RunConfig,
    items: list[GoldenItem],
    mapping: dict[str, set[str]],
    store: StoreConfig,
    embedding: EmbeddingConfig,
    embedder: Embedder,
    reranker: LateInteractionReranker,
    llm: OpenRouterClient | None,
    judge: FaithfulnessJudge | None,
    generation: GenerationConfig,
    top_k: int,
    candidate_k: int,
    retrieval_only: bool,
    log,
) -> dict:
    log(
        f"Running chunk_size={config.chunk_size} hybrid={config.hybrid} "
        f"rerank={config.rerank} ({len(items)} questions)"
    )
    retrieval = RetrievalConfig(
        top_k=top_k,
        candidate_k=candidate_k,
        hybrid=config.hybrid,
        rerank=config.rerank,
    )
    retriever = Retriever(
        store,
        embedding,
        retrieval,
        embedder=embedder,
        reranker=reranker,
    )
    pipeline = RagPipeline(retriever, llm, generation) if llm is not None else None

    question_results: list[dict] = []
    for index, item in enumerate(items, start=1):
        if index % 10 == 0:
            log(f"  {index}/{len(items)}")
        gold_ids = mapping[item.id]
        started = time.perf_counter()

        try:
            retrieved = retriever.search(item.question, top_k=top_k)
            retrieval_error = None
        except Exception as error:
            retrieved = []
            retrieval_error = str(error)

        retrieved_ids = [chunk.chunk_id for chunk in retrieved]
        record: dict = {
            "id": item.id,
            "question": item.question,
            "answerable": item.answerable,
            "gold_chunk_ids": sorted(gold_ids),
            "retrieved_chunk_ids": retrieved_ids,
            "recall": recall_at_k(gold_ids, retrieved_ids) if item.answerable else None,
            "reciprocal_rank": reciprocal_rank(gold_ids, retrieved_ids)
            if item.answerable
            else None,
            "retrieval_error": retrieval_error,
        }

        if retrieval_only:
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            question_results.append(record)
            continue

        if retrieval_error is not None:
            record.update(
                {
                    "refused": False,
                    "error": None,
                    "retrieval_error": retrieval_error,
                    "answered": False,
                    "citation_precision": 0.0 if item.answerable else None,
                    "citation_recall": 0.0 if item.answerable else None,
                    "citation_validity": None,
                    "faithfulness": None,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            question_results.append(record)
            continue

        try:
            response = pipeline.answer_with_chunks(item.question, retrieved)
        except (LLMError, GenerationError) as error:
            record.update(
                {
                    "refused": False,
                    "error": str(error),
                    "answered": False,
                    "citation_precision": 0.0 if item.answerable else None,
                    "citation_recall": 0.0 if item.answerable else None,
                    "citation_validity": None,
                    "faithfulness": None,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            question_results.append(record)
            continue

        valid_ids = [citation.chunk_id for citation in response.citations]
        invalid_ids = [citation.citation.chunk_id for citation in response.invalid_citations]
        answered = not response.refused

        record.update(
            {
                "answer": response.answer,
                "refused": response.refused,
                "refusal_reason": response.refusal_reason,
                "answered": answered,
                "citations": [
                    {"chunk_id": citation.chunk_id, "quote": citation.quote}
                    for citation in response.citations
                ],
                "invalid_citations": [
                    {
                        "chunk_id": citation.citation.chunk_id,
                        "reason": citation.reason,
                    }
                    for citation in response.invalid_citations
                ],
                "citation_validity": None
                if response.refused
                else citation_validity(valid_ids, invalid_ids),
            }
        )

        if item.answerable:
            record["citation_precision"] = citation_precision(valid_ids, invalid_ids, gold_ids)
            record["citation_recall"] = citation_recall(valid_ids, gold_ids)
        else:
            record["citation_precision"] = None
            record["citation_recall"] = None

        if item.answerable and answered:
            try:
                judgement = judge.judge(item.question, response.answer, response.retrieved)
                record["faithfulness"] = judgement.score
                record["faithfulness_faithful"] = judgement.faithful
                record["unsupported_claims"] = judgement.unsupported_claims
                record["judge_error"] = None
            except (JudgeError, LLMError) as error:
                record["faithfulness"] = None
                record["judge_error"] = str(error)
                log(f"  Judge failed for {item.id}: {error}")
        else:
            record["faithfulness"] = None
            record["judge_error"] = None

        record["latency_ms"] = (time.perf_counter() - started) * 1000
        question_results.append(record)

    return {
        "config": {
            "chunk_size": config.chunk_size,
            "overlap": config.overlap,
            "strategy": config.strategy,
            "hybrid": config.hybrid,
            "rerank": config.rerank,
            "table_name": config.table_name,
        },
        "metrics": aggregate_metrics(question_results, retrieval_only),
        "questions": question_results,
    }


def aggregate_metrics(question_results: list[dict], retrieval_only: bool) -> dict:
    answerable = [record for record in question_results if record["answerable"]]
    unanswerable = [record for record in question_results if not record["answerable"]]
    metrics = {
        "question_count": len(question_results),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "recall_at_k": mean_or_none(record["recall"] for record in answerable),
        "mrr": mean_or_none(record["reciprocal_rank"] for record in answerable),
        "retrieval_errors": sum(1 for record in question_results if record["retrieval_error"]),
    }

    if retrieval_only:
        metrics.update(
            {
                "faithfulness": None,
                "citation_precision": None,
                "citation_recall": None,
                "citation_validity": None,
                "refusal_rate_unanswerable": None,
                "answer_rate_answerable": None,
                "generation_errors": None,
                "judge_errors": None,
                "judged_answers": None,
            }
        )
        return metrics

    metrics.update(
        {
            "faithfulness": mean_or_none(record["faithfulness"] for record in answerable),
            "citation_precision": mean_or_none(
                record["citation_precision"] for record in answerable
            ),
            "citation_recall": mean_or_none(record["citation_recall"] for record in answerable),
            "citation_validity": mean_or_none(
                record["citation_validity"]
                for record in answerable
                if record["citation_validity"] is not None
            ),
            "refusal_rate_unanswerable": refusal_rate(record["refused"] for record in unanswerable),
            "answer_rate_answerable": refusal_rate(record["answered"] for record in answerable),
            "generation_errors": sum(1 for record in question_results if record.get("error")),
            "judge_errors": sum(1 for record in question_results if record.get("judge_error")),
            "judged_answers": sum(
                1 for record in answerable if record.get("faithfulness") is not None
            ),
        }
    )
    return metrics


def format_results_table(runs: list[dict], retrieval_only: bool) -> str:
    headers = [
        "size",
        "hybrid",
        "rerank",
        "recall@k",
        "mrr",
        "faith",
        "cite_p",
        "cite_r",
        "ref_unans",
        "answer_rate",
        "errors",
    ]
    rows = []
    for run in runs:
        config = run["config"]
        metrics = run["metrics"]
        errors = metrics["retrieval_errors"]
        if not retrieval_only:
            errors += metrics["generation_errors"] + metrics["judge_errors"]
        rows.append(
            [
                str(config["chunk_size"]),
                "on" if config["hybrid"] else "off",
                "on" if config["rerank"] else "off",
                format_metric(metrics["recall_at_k"]),
                format_metric(metrics["mrr"]),
                format_metric(metrics["faithfulness"]),
                format_metric(metrics["citation_precision"]),
                format_metric(metrics["citation_recall"]),
                format_metric(metrics["refusal_rate_unanswerable"]),
                format_metric(metrics["answer_rate_answerable"]),
                str(errors),
            ]
        )
    return format_table(headers, rows)


def _log_metrics(run: dict, log) -> None:
    metrics = run["metrics"]
    log(
        "  recall@k={recall} mrr={mrr} faith={faith} cite_p={precision} "
        "cite_r={recall_c} ref_unans={refusal}".format(
            recall=format_metric(metrics["recall_at_k"]),
            mrr=format_metric(metrics["mrr"]),
            faith=format_metric(metrics["faithfulness"]),
            precision=format_metric(metrics["citation_precision"]),
            recall_c=format_metric(metrics["citation_recall"]),
            refusal=format_metric(metrics["refusal_rate_unanswerable"]),
        )
    )


def parse_chunk_sizes(value: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("chunk sizes must be comma-separated integers") from error
    if not sizes:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation matrix")
    parser.add_argument("--docs", type=Path, default=Path("data/docs"))
    parser.add_argument("--golden", type=Path, default=Path("data/evals/golden.jsonl"))
    parser.add_argument("--db", type=Path, default=Path(".lancedb"))
    parser.add_argument("--output", type=Path, default=Path("eval_results"))
    parser.add_argument(
        "--chunk-sizes", type=parse_chunk_sizes, default=parse_chunk_sizes("400,800,1200")
    )
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--strategy", choices=["heading", "fixed"], default="heading")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k < 1 or args.candidate_k < args.top_k:
        raise SystemExit("candidate_k must be greater than or equal to top_k")
    if any(size <= args.overlap for size in args.chunk_sizes):
        raise SystemExit("every chunk size must be larger than the overlap")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive")

    try:
        run_evaluation(
            docs_dir=args.docs,
            golden_path=args.golden,
            db_path=args.db,
            chunk_sizes=args.chunk_sizes,
            overlap=args.overlap,
            strategy=args.strategy,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            output_dir=args.output,
            retrieval_only=args.retrieval_only,
            judge_model=args.judge_model,
            limit=args.limit,
            reuse_index=args.reuse_index,
        )
    except (FileNotFoundError, ValueError) as error:
        message = str(error)
        if "OPENROUTER_API_KEY" in message:
            message += "\nSet the key, or use `make eval EVAL_FLAGS=--retrieval-only`."
        raise SystemExit(message) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
