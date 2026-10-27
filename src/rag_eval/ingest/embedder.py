from __future__ import annotations

from rag_eval.config import EmbeddingConfig


class Embedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.embed(texts, batch_size=self.config.batch_size)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        model = self._get_model()
        vector = next(iter(model.query_embed(query)))
        return [float(value) for value in vector]

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.config.model_name,
                cache_dir=str(self.config.cache_dir),
            )
        return self._model
