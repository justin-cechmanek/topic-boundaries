import inspect
from pathlib import Path

import pytest

from topic_boundaries import config as cfg
from topic_boundaries.config import Settings


@pytest.fixture
def _no_config_env(monkeypatch):
    """Config precedence tests must ignore any REDIS_URL/EMBEDDING_MODEL the
    caller's shell (or CI) has exported, since env overrides the config file."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)


def test_settings_reads_redis_model_and_kmeans_from_config(_no_config_env, tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text(
        "\n".join(
            [
                "redis_url: redis://localhost:6379/9",
                "embedding_model: sentence-transformers/all-mpnet-base-v2",
                "kmeans:",
                "  random_state: 7",
                "  n_init: 25",
            ]
        ),
        encoding="utf-8",
    )
    s = Settings.from_config(config_path=cfg)
    assert s.embedding_model == "sentence-transformers/all-mpnet-base-v2"
    assert s.kmeans_random_state == 7
    assert s.kmeans_n_init == 25
    assert s.redis_url == "redis://localhost:6379/9"


def test_settings_defaults_without_config_file(_no_config_env, tmp_path: Path):
    missing = tmp_path / "missing.yml"
    s = Settings.from_config(config_path=missing)
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.kmeans_random_state == 42
    assert s.kmeans_n_init == "auto"


def test_env_overrides_config(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("redis_url: redis://from-file:6379/1\n", encoding="utf-8")
    monkeypatch.setenv("REDIS_URL", "redis://from-env:6379/2")
    monkeypatch.setenv("EMBEDDING_MODEL", "env-model")
    s = Settings.from_config(config_path=cfg)
    assert s.redis_url == "redis://from-env:6379/2"  # env wins over file
    assert s.embedding_model == "env-model"


def test_n_init_absent_defaults_to_auto(_no_config_env, tmp_path: Path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("kmeans:\n  random_state: 7\n", encoding="utf-8")
    assert Settings.from_config(config_path=cfg_path).kmeans_n_init == "auto"


def test_from_config_rejects_non_mapping_top_level(tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Settings.from_config(config_path=cfg)


def test_from_config_rejects_non_mapping_kmeans(tmp_path: Path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("kmeans: not_a_map\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Settings.from_config(config_path=cfg_path)


def _default(func, param):
    return inspect.signature(func).parameters[param].default


def test_scattered_defaults_reference_the_central_constants():
    """The chunk/batch defaults must live in one place, not be re-typed per site."""
    from topic_boundaries import embedding, pdf_corpus, pipeline, sources

    for func, param, expected in [
        (pdf_corpus.chunk_text, "max_chars", cfg.DEFAULT_CHUNK_MAX_CHARS),
        (pdf_corpus.chunk_text, "overlap", cfg.DEFAULT_CHUNK_OVERLAP),
        (pdf_corpus.pdf_to_datapoints, "max_chars", cfg.DEFAULT_CHUNK_MAX_CHARS),
        (pdf_corpus.pdf_to_datapoints, "overlap", cfg.DEFAULT_CHUNK_OVERLAP),
        (sources.PdfSource.__init__, "max_chars", cfg.DEFAULT_CHUNK_MAX_CHARS),
        (sources.PdfSource.__init__, "overlap", cfg.DEFAULT_CHUNK_OVERLAP),
        (sources.TextDirSource.__init__, "max_chars", cfg.DEFAULT_CHUNK_MAX_CHARS),
        (sources.TextDirSource.__init__, "overlap", cfg.DEFAULT_CHUNK_OVERLAP),
        (embedding.HFTextEmbedder.__init__, "batch_size", cfg.DEFAULT_EMBED_BATCH_SIZE),
        (pipeline.embed_and_cluster_datapoints, "embed_batch_size", cfg.DEFAULT_EMBED_BATCH_SIZE),
    ]:
        assert _default(func, param) == expected, f"{func.__qualname__}({param})"
