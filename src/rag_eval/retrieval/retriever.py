from __future__ import annotations

from pathlib import Path

import lancedb

from rag_eval.config import EmbeddingConfig, RetrievalConfig, StoreConfig
from rag_eval.ingest.embedder import Embedder
from rag_eval.retrieval.fusion import BM25Index, reciprocal_rank_fusion
from rag_eval.retrieval.reranker import LateInteractionReranker
from rag_eval.retrieval.types import RetrievedChunk


class Retriever:
    def __init__(
        self,
        store_config: StoreConfig,
        embedding_config: EmbeddingConfig,
        retrieval_config: RetrievalConfig,
        embedder: Embedder | None = None,
        reranker: LateInteractionReranker | None = None,
    ) -> None:
        self.store_config = store_config
        self.embedding_config = embedding_config
        self.config = retrieval_config
        self.embedder = embedder or Embedder(embedding_config)
        self.reranker = reranker or LateInteractionReranker(
            retrieval_config.rerank_model,
            embedding_config.cache_dir,
        )
        self._table = None
        self._corpus: dict[str, RetrievedChunk] | None = None
        self._bm25: BM25Index | None = None

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        limit = top_k or self.config.top_k
        fetch_k = max(self.config.candidate_k, limit)
        vector_hits = self._vector_search(query, fetch_k)

        if self.config.hybrid:
            candidates = self._hybrid_candidates(query, vector_hits, fetch_k)
        else:
            candidates = vector_hits

        if self.config.rerank:
            candidates = self._rerank(query, candidates)

        candidates.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return candidates[:limit]

    def _hybrid_candidates(
        self,
        query: str,
        vector_hits: list[RetrievedChunk],
        fetch_k: int,
    ) -> list[RetrievedChunk]:
        bm25_hits = self._get_bm25().search(query, fetch_k)
        fused = reciprocal_rank_fusion(
            [
                [chunk.chunk_id for chunk in vector_hits],
                [chunk_id for chunk_id, _ in bm25_hits],
            ],
            k=self.config.rrf_k,
        )

        corpus = self._get_corpus()
        vector_by_id = {chunk.chunk_id: chunk for chunk in vector_hits}
        bm25_scores = dict(bm25_hits)
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_hits, start=1)}
        vector_ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(vector_hits, start=1)}

        candidates: list[RetrievedChunk] = []
        for chunk_id, score in fused.items():
            base = vector_by_id.get(chunk_id) or corpus.get(chunk_id)
            if base is None:
                continue
            candidates.append(
                base.model_copy(
                    update={
                        "score": score,
                        "vector_rank": vector_ranks.get(chunk_id),
                        "bm25_rank": bm25_ranks.get(chunk_id),
                        "bm25_score": bm25_scores.get(chunk_id),
                    }
                )
            )
        return candidates

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        scores = self.reranker.score(query, [chunk.text for chunk in candidates])
        reranked = [
            chunk.model_copy(update={"rerank_score": score, "score": score})
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return reranked

    def _vector_search(self, query: str, limit: int) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_query(query)
        rows = self._get_table().search(query_vector).metric("cosine").limit(limit).to_list()
        hits: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            score = 1.0 - float(row.get("_distance", 0.0))
            hits.append(
                self._to_chunk(
                    row,
                    score=score,
                    vector_rank=rank,
                    vector_score=score,
                )
            )
        return hits

    def _to_chunk(
        self,
        row: dict,
        score: float,
        vector_rank: int | None = None,
        vector_score: float | None = None,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            doc=row["doc"],
            page=row["page"] or None,
            heading=row["heading"] or None,
            text=row["text"],
            score=score,
            vector_score=vector_score,
            vector_rank=vector_rank,
        )

    def _get_table(self):
        if self._table is None:
            db_path = self.store_config.db_path
            if not Path(db_path).exists():
                raise FileNotFoundError(
                    f"Vector store not found at {db_path}. Run `make ingest` first."
                )
            database = lancedb.connect(str(db_path))
            if self.store_config.table_name not in database.table_names():
                raise FileNotFoundError(
                    f"Table {self.store_config.table_name!r} not found in {db_path}. "
                    "Run `make ingest` first."
                )
            self._table = database.open_table(self.store_config.table_name)
        return self._table

    def _get_corpus(self) -> dict[str, RetrievedChunk]:
        if self._corpus is None:
            rows = (
                self._get_table()
                .to_arrow()
                .select(["chunk_id", "doc", "page", "heading", "text"])
                .to_pylist()
            )
            self._corpus = {row["chunk_id"]: self._to_chunk(row, score=0.0) for row in rows}
        return self._corpus

    def _get_bm25(self) -> BM25Index:
        if self._bm25 is None:
            corpus = self._get_corpus()
            self._bm25 = BM25Index(
                [(chunk_id, chunk.text) for chunk_id, chunk in corpus.items()],
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )
        return self._bm25
