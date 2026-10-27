from __future__ import annotations

import os

import pytest

from rag_eval.config import OpenRouterConfig, RetrievalConfig, load_env


def test_retrieval_rejects_fewer_candidates_than_results():
    with pytest.raises(ValueError):
        RetrievalConfig(top_k=10, candidate_k=5)


def test_load_env_parses_values_and_does_not_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nOPENROUTER_API_KEY="from-file"\nOPENROUTER_MODEL=model-from-file\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "environ", {"OPENROUTER_MODEL": "existing-model"})

    load_env(env_file)

    assert os.environ["OPENROUTER_API_KEY"] == "from-file"
    assert os.environ["OPENROUTER_MODEL"] == "existing-model"


def test_from_env_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {})

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterConfig.from_env(tmp_path / "missing.env")
