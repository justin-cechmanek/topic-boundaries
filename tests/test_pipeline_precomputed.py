"""End-to-end proof that precomputed vectors flow through run_pipeline with no
embedding, into a real Redis index, and out through a boundary method."""

import numpy as np
import pytest

from topic_boundaries import Datapoint, PrecomputedEmbedder, max_distance_rankings, run_pipeline


@pytest.fixture
def _needs_redis(redis_url):
    redis = pytest.importorskip("redis")
    try:
        redis.from_url(redis_url).ping()
    except Exception:
        pytest.skip("Redis Stack not reachable")


def _dp(doc_id, vec):
    return Datapoint(doc_id=doc_id, body="", vector=np.asarray(vec, dtype=np.float32))


@pytest.mark.integration
def test_run_pipeline_with_precomputed_vectors(_needs_redis, redis_url):
    # Two clearly separated clusters in 4-D; no text anywhere.
    dps = [
        _dp("a0", [1.0, 0.0, 0.0, 0.0]),
        _dp("a1", [0.9, 0.1, 0.0, 0.0]),
        _dp("b0", [0.0, 0.0, 1.0, 0.0]),
        _dp("b1", [0.0, 0.1, 0.9, 0.0]),
    ]
    state = run_pipeline(
        dps,
        redis_url=redis_url,
        n_clusters=2,
        embedder=PrecomputedEmbedder(),  # <-- no embedding model
        schema_path=None,
        overwrite_index=True,
    )
    # Passthrough: stored vectors equal the inputs (embedding was skipped).
    assert state.vectors.shape == (4, 4)
    np.testing.assert_array_equal(state.vectors[0], dps[0].vector)
    assert set(state.labels.tolist()) == {0, 1}

    hits = max_distance_rankings(state.indexed.index, state.centroids, 2)
    assert len(hits) == 4
    assert {h.cluster_id for h in hits} == {0, 1}
