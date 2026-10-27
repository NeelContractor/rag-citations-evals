from __future__ import annotations

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


def test_recall_at_k_perfect_partial_and_missing():
    assert recall_at_k({"a", "b"}, ["a", "b", "c"]) == 1.0
    assert recall_at_k({"a", "b"}, ["a", "c"]) == 0.5
    assert recall_at_k({"a"}, ["b", "c"]) == 0.0


def test_recall_at_k_without_gold_is_zero():
    assert recall_at_k(set(), ["a"]) == 0.0


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank({"a"}, ["a", "b"]) == 1.0
    assert reciprocal_rank({"b"}, ["a", "b"]) == 0.5
    assert reciprocal_rank({"c"}, ["a", "b", "c"]) == 1 / 3
    assert reciprocal_rank({"z"}, ["a", "b"]) == 0.0


def test_reciprocal_rank_without_gold_is_zero():
    assert reciprocal_rank(set(), ["a"]) == 0.0


def test_citation_precision_counts_valid_gold_citations():
    assert citation_precision(["a", "b"], [], {"a", "b"}) == 1.0
    assert citation_precision(["a"], ["bad"], {"a"}) == 0.5
    assert citation_precision(["a"], [], {"b"}) == 0.0


def test_citation_precision_with_no_citations_is_zero():
    assert citation_precision([], [], {"a"}) == 0.0


def test_citation_recall_measures_gold_coverage():
    assert citation_recall(["a", "b"], {"a", "b"}) == 1.0
    assert citation_recall(["a"], {"a", "b"}) == 0.5
    assert citation_recall([], {"a"}) == 0.0


def test_citation_validity_ignores_gold_set():
    assert citation_validity(["a"], []) == 1.0
    assert citation_validity(["a"], ["bad"]) == 0.5
    assert citation_validity([], []) == 1.0


def test_refusal_rate():
    assert refusal_rate([True, False, True]) == 2 / 3
    assert refusal_rate([]) == 0.0


def test_mean_or_none_ignores_missing_values():
    assert mean_or_none([1.0, None, 3.0]) == 2.0
    assert mean_or_none([None, None]) is None
    assert mean_or_none([]) is None


def test_format_metric_handles_none():
    assert format_metric(0.12345) == "0.123"
    assert format_metric(None) == "-"


def test_format_table_aligns_columns():
    table = format_table(["a", "long"], [["x", "y"], ["xx", "z"]])
    lines = table.splitlines()

    assert lines[0].startswith("a ")
    assert "long" in lines[0]
    assert set(lines[1]) == {"-", " "}
    assert lines[2].startswith("x ")
    assert lines[3].startswith("xx")
