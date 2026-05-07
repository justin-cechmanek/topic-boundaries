from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def _parse_n_init(value: object, default: int | str = "auto") -> int | str:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        if s.isdigit():
            return int(s)
        return s
    raise ValueError(f"Invalid kmeans_n_init value: {value!r}")


@dataclass(frozen=True)
class Settings:
    redis_url: str
    embedding_model: str
    schema_path: Path
    kmeans_random_state: int
    kmeans_n_init: int | str

    @classmethod
    def from_config(
        cls,
        *,
        config_path: Path | None = None,
        schema_path: Path | None = None,
    ) -> Settings:
        root = Path(__file__).resolve().parent
        sp = schema_path or root / "schema.yml"
        cp = config_path or root.parent / "config.yml"
        data: dict = {}
        if cp.is_file():
            with cp.open(encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a top-level map: {cp}")
            data = loaded
        km = data.get("kmeans", {})
        if km and not isinstance(km, dict):
            raise ValueError("Config key 'kmeans' must be a map.")
        return cls(
            redis_url=str(data.get("redis_url", "redis://localhost:6379/0")),
            embedding_model=str(
                data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            schema_path=sp,
            kmeans_random_state=int(km.get("random_state", 42)),
            kmeans_n_init=_parse_n_init(km.get("n_init"), default="auto"),
        )
