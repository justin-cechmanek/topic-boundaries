import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.clustering import cluster_counts, cluster_embeddings


def test_cluster_embeddings_groups_separated_points():
    # Two tight, well-separated blobs -> the two members of each blob share a label.
    vectors = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], dtype=np.float32
    )
    labels, centroids = cluster_embeddings(vectors, 2, random_state=0)
    assert labels.dtype == np.int64
    assert centroids.shape == (2, 2)
    assert centroids.dtype == np.float32
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_embeddings_deterministic_with_fixed_seed():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((30, 5)).astype(np.float32)
    a, _ = cluster_embeddings(vectors, 3, random_state=42)
    b, _ = cluster_embeddings(vectors, 3, random_state=42)
    np.testing.assert_array_equal(a, b)


def test_cluster_counts_reports_zero_for_absent_cluster():
    labels = np.array([0, 0, 1])
    counts = cluster_counts(labels, 3)
    np.testing.assert_array_equal(counts, np.array([2, 1, 0]))
