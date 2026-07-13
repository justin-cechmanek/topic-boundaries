"""Full real path: embed -> KMeans -> Redis index -> query, no mocks.

Uses the cached MiniLM model and a live Redis Stack; skips if either is absent.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from topic_boundaries.cli import main
from topic_boundaries.documents import load_jsonl
from topic_boundaries.pipeline import run_pipeline

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
def test_run_pipeline_end_to_end(_needs_redis, redis_url):
    dps = load_jsonl(ROOT / "datasets" / "sample.jsonl")
    n_clusters = 2
    state = run_pipeline(
        dps,
        redis_url=redis_url,
        n_clusters=n_clusters,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        schema_path=None,
        overwrite_index=True,
    )
    assert state.vectors.shape == (len(dps), 384)
    assert state.labels.shape == (len(dps),)
    assert state.centroids.shape == (n_clusters, 384)
    # Guard in run_pipeline forbids empty clusters, so every label is populated.
    assert set(state.labels.tolist()) == set(range(n_clusters))
    assert np.isclose(np.linalg.norm(state.vectors[0]), 1.0, atol=1e-3)


@pytest.mark.integration
@pytest.mark.parametrize(
    "method",
    ["max_distance_sort", "convex_hull", "cross_boundary", "centroid_neighbors"],
)
def test_cli_main_emits_valid_json(_needs_redis, capsys, method):
    rc = main(
        [
            "--data",
            str(ROOT / "datasets" / "sample.jsonl"),
            "--n-clusters",
            "2",
            "--method",
            method,
            "--overwrite-index",
            "--top-n",
            "5",
            "--centroid-neighbors-k",
            "3",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == method
    if method == "centroid_neighbors":
        assert len(out["clusters"]) == 2
        assert out["clusters"][0]["nearest_to_centroid"]
    else:
        # Each of the three boundary methods buckets hits per cluster.
        assert len(out["boundary_by_cluster"]) == 2
        assert any(out["boundary_by_cluster"])


def test_cli_main_rejects_too_few_datapoints(capsys):  # guard runs before Redis
    rc = main(
        [
            "--data",
            str(ROOT / "datasets" / "sample.jsonl"),
            "--n-clusters",
            "100000",
            "--method",
            "convex_hull",
        ]
    )
    assert rc == 1
