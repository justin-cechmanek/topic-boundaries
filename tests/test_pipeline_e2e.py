"""Full real path: embed -> KMeans -> Redis index -> query, no mocks.

Uses the cached MiniLM model and a live Redis Stack; skips if either is absent.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from topic_boundaries.documents import load_jsonl
from topic_boundaries import embed_and_cluster_datapoints, index_clustered_datapoints

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def _needs_redis(redis_url):
    redis = pytest.importorskip("redis")
    try:
        redis.from_url(redis_url).ping()
    except Exception:
        pytest.skip("Redis Stack not reachable")


@pytest.mark.integration
def test_embed_cluster_and_index_end_to_end(_needs_redis, redis_url, isolated_schema):
    dps = load_jsonl(ROOT / "datasets" / "sample.jsonl")
    n_clusters = 2
    vectors, labels, centroids = embed_and_cluster_datapoints(
        dps,
        n_clusters=n_clusters,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    index = index_clustered_datapoints(
        dps,
        vectors,
        labels,
        redis_url=redis_url,
        overwrite_index=True,
        schema_path=str(isolated_schema()),
    )
    assert vectors.shape == (len(dps), 384)
    assert labels.shape == (len(dps),)
    assert centroids.shape == (n_clusters, 384)
    assert set(labels.tolist()) == set(range(n_clusters))
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-3)
    assert index is not None
