from pathlib import Path

from src.config import Settings


def test_settings_reads_redis_model_and_kmeans_from_config(tmp_path: Path):
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


def test_settings_defaults_without_config_file(tmp_path: Path):
    missing = tmp_path / "missing.yml"
    s = Settings.from_config(config_path=missing)
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.kmeans_random_state == 42
    assert s.kmeans_n_init == "auto"
