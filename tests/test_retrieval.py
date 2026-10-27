from __future__ import annotations

from rag_eval.retrieval.fusion import BM25Index, reciprocal_rank_fusion, tokenize
from rag_eval.retrieval.reranker import late_interaction_score


def test_tokenize_lowercases_and_keeps_alphanumerics():
    assert tokenize("RAG-based retrieval, v2!") == ["rag", "based", "retrieval", "v2"]


def test_bm25_ranks_matching_document_first():
    index = BM25Index(
        [
            ("a", "cats sit on mats"),
            ("b", "dogs chase balls"),
            ("c", "retrieval augmented generation cites sources"),
        ]
    )
    results = index.search("retrieval sources", limit=3)

    assert results
    assert results[0][0] == "c"
    assert results[0][1] > 0


def test_bm25_returns_empty_for_no_matches():
    index = BM25Index([("a", "cats sit on mats"), ("b", "dogs chase balls")])
    assert index.search("quantum", limit=5) == []


def test_bm25_handles_empty_corpus():
    index = BM25Index([])
    assert index.search("anything", limit=5) == []


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "c", "a"]],
        k=60,
    )

    assert list(fused)[:2] == ["b", "a"]
    assert fused["b"] > fused["a"] > fused["c"]


def test_reciprocal_rank_fusion_handles_empty_rankings():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_late_interaction_score_uses_best_document_token_per_query_token():
    query = [[1.0, 0.0], [0.0, 1.0]]
    document = [[1.0, 0.0], [0.0, 2.0]]
    assert late_interaction_score(query, document) == 3.0


def test_late_interaction_score_is_zero_for_empty_input():
    assert late_interaction_score([], [[1.0]]) == 0.0
    assert late_interaction_score([[1.0]], []) == 0.0
