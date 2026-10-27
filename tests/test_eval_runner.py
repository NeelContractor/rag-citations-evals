from __future__ import annotations

from rag_eval.eval.runner import (
    RunConfig,
    aggregate_metrics,
    format_results_table,
    parse_chunk_sizes,
)


def make_records() -> list[dict]:
    return [
        {
            "id": "q1",
            "answerable": True,
            "recall": 1.0,
            "reciprocal_rank": 1.0,
            "refused": False,
            "answered": True,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "citation_validity": 1.0,
            "faithfulness": 1.0,
            "retrieval_error": None,
            "error": None,
            "judge_error": None,
        },
        {
            "id": "q2",
            "answerable": True,
            "recall": 0.5,
            "reciprocal_rank": 0.5,
            "refused": False,
            "answered": True,
            "citation_precision": 0.5,
            "citation_recall": 0.5,
            "citation_validity": 0.5,
            "faithfulness": 0.5,
            "retrieval_error": None,
            "error": None,
            "judge_error": None,
        },
        {
            "id": "q3",
            "answerable": False,
            "recall": None,
            "reciprocal_rank": None,
            "refused": True,
            "answered": False,
            "citation_precision": None,
            "citation_recall": None,
            "citation_validity": None,
            "faithfulness": None,
            "retrieval_error": None,
            "error": None,
            "judge_error": None,
        },
    ]


def test_aggregate_metrics_full_mode():
    metrics = aggregate_metrics(make_records(), retrieval_only=False)

    assert metrics["question_count"] == 3
    assert metrics["answerable_count"] == 2
    assert metrics["unanswerable_count"] == 1
    assert metrics["recall_at_k"] == 0.75
    assert metrics["mrr"] == 0.75
    assert metrics["faithfulness"] == 0.75
    assert metrics["citation_precision"] == 0.75
    assert metrics["citation_recall"] == 0.75
    assert metrics["refusal_rate_unanswerable"] == 1.0
    assert metrics["answer_rate_answerable"] == 1.0
    assert metrics["judged_answers"] == 2
    assert metrics["generation_errors"] == 0


def test_aggregate_metrics_retrieval_only_has_no_answer_metrics():
    metrics = aggregate_metrics(make_records(), retrieval_only=True)

    assert metrics["recall_at_k"] == 0.75
    assert metrics["mrr"] == 0.75
    assert metrics["faithfulness"] is None
    assert metrics["citation_precision"] is None
    assert metrics["refusal_rate_unanswerable"] is None


def test_aggregate_metrics_counts_errors_and_refusals():
    records = make_records()
    records[0]["error"] = "model failed"
    records[0]["answered"] = False
    records[0]["faithfulness"] = None
    records[1]["judge_error"] = "judge failed"
    records[1]["faithfulness"] = None

    metrics = aggregate_metrics(records, retrieval_only=False)

    assert metrics["generation_errors"] == 1
    assert metrics["judge_errors"] == 1
    assert metrics["answer_rate_answerable"] == 0.5
    assert metrics["faithfulness"] is None
    assert metrics["judged_answers"] == 0


def test_run_config_table_name_is_shared_across_retrieval_flags():
    base = RunConfig(800, 100, "heading", False, False)
    hybrid = RunConfig(800, 100, "heading", True, False)
    rerank = RunConfig(800, 100, "heading", False, True)

    assert base.table_name == hybrid.table_name == rerank.table_name
    assert base.table_name == "eval_heading_800_100"


def test_format_results_table_contains_config_and_metrics():
    run = {
        "config": {
            "chunk_size": 800,
            "hybrid": True,
            "rerank": False,
        },
        "metrics": {
            "recall_at_k": 0.8,
            "mrr": 0.7,
            "faithfulness": None,
            "citation_precision": None,
            "citation_recall": None,
            "refusal_rate_unanswerable": None,
            "answer_rate_answerable": None,
            "retrieval_errors": 0,
            "generation_errors": None,
            "judge_errors": None,
        },
    }

    table = format_results_table([run], retrieval_only=True)

    assert "size" in table
    assert "800" in table
    assert "on" in table
    assert "0.800" in table
    assert "0.700" in table


def test_parse_chunk_sizes():
    assert parse_chunk_sizes("400, 800,1200") == [400, 800, 1200]
