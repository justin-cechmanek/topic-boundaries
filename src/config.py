from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    redis_url: str
    embedding_model: str
    schema_path: Path

    @classmethod
    def from_env(cls, schema_path: Path | None = None) -> Settings:
        root = Path(__file__).resolve().parent
        sp = schema_path or root / "schema.yml"
        return cls(
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            schema_path=sp,
        )
