from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# --- Centralized defaults (single source of truth) --------------------------
# These back both the runtime function signatures (chunk/embedding sites import
# them directly) and the Settings fields below. Change a knob here, not in four
# scattered call sites.

DEFAULT_N_CLUSTERS = 10  # KMeans k; dataset-dependent. See TODO PR-T2 (auto-k).

# Chunking. NOTE: keep max_chars under the embedding model's max input length or
# the model silently truncates. all-MiniLM-L6-v2 caps at 256 tokens (~1000
# chars), so 1500 already truncates a little — lower it for that model, or raise
# it for long-context embedders. overlap keeps context across chunk edges.
DEFAULT_CHUNK_MAX_CHARS = 1500
DEFAULT_CHUNK_OVERLAP = 200

DEFAULT_EMBED_BATCH_SIZE = 64  # embeds per forward pass; raise if GPU memory allows.

# Boundary rankers: how many hits to keep per cluster. Sensible values scale with
# cluster size (a handful out of tens-to-hundreds of members).
DEFAULT_TOP_N_PER_CLUSTER = 5
DEFAULT_K_PER_CENTROID = 5
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    redis_url: str
    embedding_model: str
    schema_path: Path
    kmeans_random_state: int
    kmeans_n_init: int | str
    # Tunable defaults for a future analysis entry point (TODO PR-T5). These
    # mirror the module constants above and are read from config.yml when present.
    n_clusters: int
    chunk_max_chars: int
    chunk_overlap: int
    embed_batch_size: int
    top_n_per_cluster: int
    k_per_centroid: int

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
        # Environment variables override config.yml values, matching the old from_env() behaviour.
        return cls(
            redis_url=os.environ.get(
                "REDIS_URL", str(data.get("redis_url", "redis://localhost:6379/0"))
            ),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL",
                str(data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            ),
            schema_path=sp,
            kmeans_random_state=int(km.get("random_state", 42)),
            kmeans_n_init=km.get("n_init") or "auto",
            n_clusters=int(data.get("n_clusters", DEFAULT_N_CLUSTERS)),
            chunk_max_chars=int(data.get("chunk_max_chars", DEFAULT_CHUNK_MAX_CHARS)),
            chunk_overlap=int(data.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
            embed_batch_size=int(data.get("embed_batch_size", DEFAULT_EMBED_BATCH_SIZE)),
            top_n_per_cluster=int(data.get("top_n_per_cluster", DEFAULT_TOP_N_PER_CLUSTER)),
            k_per_centroid=int(data.get("k_per_centroid", DEFAULT_K_PER_CENTROID)),
        )
