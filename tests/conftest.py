"""Shared fixtures. Redis fixtures build a real index and skip when Redis is down.

No mocks: the query tests run against a live Redis Stack (see docker-compose.yml).
Synthetic low-dim unit vectors keep assertions deterministic without an embedder.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from topic_boundaries.indexing import create_and_load, open_index, records_for_redis

DIM = 8  # small dim for hand-built vectors; open_index overrides schema.yml dims


def vec(*components: float) -> np.ndarray:
    """Unit-normalized float32 vector, zero-padded to DIM."""
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(components)] = components
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n else v


@pytest.fixture
def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def make_index(redis_url):
    """Factory: build a real Redis index from given records; drop it on teardown.

    Skips the test if Redis is unreachable. Serial pytest execution means a fixed
    index name is safe; each build recreates with overwrite+drop.
    """
    redis = pytest.importorskip("redis")
    try:
        redis.from_url(redis_url).ping()
    except Exception:
        pytest.skip("Redis Stack not reachable")

    created = []

    def _make(doc_ids, bodies, vectors, cluster_labels, dim: int = DIM):
        vectors = np.asarray(vectors, dtype=np.float32)
        index = open_index(None, redis_url, dim)
        records = records_for_redis(
            list(doc_ids), list(bodies), vectors, np.asarray(cluster_labels)
        )
        create_and_load(index, records, overwrite=True, drop_keys=True)
        created.append(index)
        return index

    yield _make

    for index in created:
        try:
            index.delete(drop=True)
        except Exception:
            pass


# Known geometry: dims 0,1,2 used. Cluster 0 spreads from the x-axis; cluster 1
# sits near z with one doc pulled toward x for the cross-boundary test.
X_CENTROID = vec(1.0, 0.0, 0.0).tolist()  # full DIM-length, along the x-axis

CORPUS = {
    # doc_id, body, cluster, vector
    "c0_near": (0, vec(1.0, 0.0, 0.0)),   # nearest to X within cluster 0
    "c0_mid": (0, vec(0.7, 0.7, 0.0)),    # ~45 deg
    "c0_far": (0, vec(0.0, 1.0, 0.0)),    # 90 deg -> farthest in cluster 0
    "c1_z": (1, vec(0.0, 0.0, 1.0)),      # z-axis
    "c1_nearx": (1, vec(0.9, 0.0, 0.436)),  # cluster 1 doc closest to X
}


@pytest.fixture
def loaded_corpus(make_index):
    """A real index loaded with CORPUS; returns (index, X_centroid)."""
    doc_ids = list(CORPUS)
    labels = [CORPUS[d][0] for d in doc_ids]
    vectors = np.stack([CORPUS[d][1] for d in doc_ids])
    bodies = [f"body of {d}" for d in doc_ids]
    index = make_index(doc_ids, bodies, vectors, labels)
    return index, list(X_CENTROID)
