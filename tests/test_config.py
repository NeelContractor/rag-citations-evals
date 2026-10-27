from __future__ import annotations

import os

import pytest

from rag_eval.config import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterConfig,
    RetrievalConfig,
    load_env,
)


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


def test_load_env_can_override_when_requested(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_MODEL=model-from-file\n", encoding="utf-8")
    monkeypatch.setattr(os, "environ", {"OPENROUTER_MODEL": "existing-model"})

    load_env(env_file, override=True)

    assert os.environ["OPENROUTER_MODEL"] == "model-from-file"


def test_from_env_reads_and_refreshes_model(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.setattr(os, "environ", {})

    env_file.write_text(
        "OPENROUTER_API_KEY=key\nOPENROUTER_MODEL=first/model\n",
        encoding="utf-8",
    )
    assert OpenRouterConfig.from_env(env_file).model == "first/model"

    env_file.write_text(
        "OPENROUTER_API_KEY=key\nOPENROUTER_MODEL=second/model\n",
        encoding="utf-8",
    )
    assert OpenRouterConfig.from_env(env_file).model == "second/model"


def test_from_env_uses_default_model_when_unset(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=key\n", encoding="utf-8")
    monkeypatch.setattr(os, "environ", {})

    assert OpenRouterConfig.from_env(env_file).model == DEFAULT_OPENROUTER_MODEL


def test_from_env_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {})

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterConfig.from_env(tmp_path / "missing.env")
