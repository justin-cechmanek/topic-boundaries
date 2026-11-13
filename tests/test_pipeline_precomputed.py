"""End-to-end proof that precomputed vectors flow through clustering with no
embedding, into a real Redis index, and out through a boundary method."""

import numpy as np
import pytest

from topic_boundaries import (
    Datapoint,
    PrecomputedEmbedder,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
    max_distance_rankings,
)


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
def test_index_with_precomputed_vectors(_needs_redis, redis_url, isolated_schema):
    # Two clearly separated clusters in 4-D; no text anywhere.
    dps = [
        _dp("a0", [1.0, 0.0, 0.0, 0.0]),
        _dp("a1", [0.9, 0.1, 0.0, 0.0]),
        _dp("b0", [0.0, 0.0, 1.0, 0.0]),
        _dp("b1", [0.0, 0.1, 0.9, 0.0]),
    ]
    vectors, labels, centroids = embed_and_cluster_datapoints(
        dps,
        n_clusters=2,
        embedder=PrecomputedEmbedder(),  # <-- no embedding model
    )
    index = index_clustered_datapoints(
        dps,
        vectors,
        labels,
        redis_url=redis_url,
        overwrite_index=True,
        schema_path=str(isolated_schema()),
    )
    # Passthrough: stored vectors equal the inputs (embedding was skipped).
    assert vectors.shape == (4, 4)
    np.testing.assert_array_equal(vectors[0], dps[0].vector)
    assert set(labels.tolist()) == {0, 1}

    hits = max_distance_rankings(index, centroids, 2)
    assert len(hits) == 4
    assert {h.cluster_id for h in hits} == {0, 1}


def test_empty_cluster_is_rejected():
    dps = [_dp("a", [1.0, 0.0]), _dp("b", [1.0, 0.0])]
    with pytest.raises(RuntimeError, match="clusters are empty"):
        embed_and_cluster_datapoints(dps, n_clusters=2, embedder=PrecomputedEmbedder())
