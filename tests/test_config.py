from __future__ import annotations

import os

import pytest

from rag_eval.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    LLMConfig,
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
    assert LLMConfig.from_env(env_file).model == "first/model"

    env_file.write_text(
        "OPENROUTER_API_KEY=key\nOPENROUTER_MODEL=second/model\n",
        encoding="utf-8",
    )
    assert LLMConfig.from_env(env_file).model == "second/model"


def test_from_env_uses_default_openrouter_model_when_unset(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=key\n", encoding="utf-8")
    monkeypatch.setattr(os, "environ", {})

    config = LLMConfig.from_env(env_file)

    assert config.provider == "openrouter"
    assert config.base_url == OPENROUTER_BASE_URL
    assert config.model == DEFAULT_OPENROUTER_MODEL


def test_from_env_auto_detects_groq_key(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"GROQ_API_KEY": "groq-key"})

    config = LLMConfig.from_env(tmp_path / "missing.env")

    assert config.provider == "groq"
    assert config.provider_label == "Groq"
    assert config.api_key == "groq-key"
    assert config.base_url == GROQ_BASE_URL
    assert config.model == DEFAULT_GROQ_MODEL


def test_from_env_accepts_openrouter_style_groq_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "OPENROUTER_API_KEY": "groq-key",
            "OPENROUTER_MODEL": "openai/gpt-oss-120b",
            "OPENROUTER_BASE_URL": GROQ_BASE_URL,
        },
    )

    config = LLMConfig.from_env(tmp_path / "missing.env")

    assert config.provider == "groq"
    assert config.api_key == "groq-key"
    assert config.model == "openai/gpt-oss-120b"
    assert config.base_url == GROQ_BASE_URL


def test_from_env_explicit_provider_and_generic_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "LLM_PROVIDER": "groq",
            "LLM_API_KEY": "generic-key",
            "LLM_MODEL": "groq-model",
            "LLM_BASE_URL": "https://example.test/openai/v1",
            "GROQ_API_KEY": "groq-key",
        },
    )

    config = LLMConfig.from_env(tmp_path / "missing.env")

    assert config.provider == "groq"
    assert config.api_key == "generic-key"
    assert config.model == "groq-model"
    assert config.base_url == "https://example.test/openai/v1"


def test_from_env_explicit_groq_ignores_stale_openrouter_base(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "groq-key",
            "OPENROUTER_BASE_URL": OPENROUTER_BASE_URL,
        },
    )

    config = LLMConfig.from_env(tmp_path / "missing.env")

    assert config.provider == "groq"
    assert config.base_url == GROQ_BASE_URL


def test_from_env_prefers_openrouter_when_both_keys_are_set(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "OPENROUTER_API_KEY": "openrouter-key",
            "GROQ_API_KEY": "groq-key",
        },
    )

    config = LLMConfig.from_env(tmp_path / "missing.env")

    assert config.provider == "openrouter"
    assert config.api_key == "openrouter-key"
    assert config.base_url == OPENROUTER_BASE_URL


def test_from_env_rejects_invalid_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"LLM_PROVIDER": "other"})

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        LLMConfig.from_env(tmp_path / "missing.env")


def test_from_env_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {})

    with pytest.raises(ValueError, match="No LLM API key"):
        LLMConfig.from_env(tmp_path / "missing.env")
