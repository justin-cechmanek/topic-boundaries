from pathlib import Path

import pytest

from src.config import Settings, _parse_n_init


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


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "auto"),
        ("", "auto"),
        ("  ", "auto"),
        (25, 25),
        ("30", 30),
        ("auto", "auto"),
    ],
)
def test_parse_n_init(value, expected):
    assert _parse_n_init(value) == expected


def test_parse_n_init_rejects_bad_type():
    with pytest.raises(ValueError):
        _parse_n_init(3.5)


def test_from_config_rejects_non_mapping_top_level(tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Settings.from_config(config_path=cfg)


def test_from_config_rejects_non_mapping_kmeans(tmp_path: Path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("kmeans: not_a_map\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Settings.from_config(config_path=cfg)
