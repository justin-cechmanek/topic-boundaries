import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.convex_hull.boundaries import (
    boundary_doc_indices_per_cluster,
    hull_boundary_indices,
)


def test_hull_returns_subset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8)).astype(np.float32)
    ix = hull_boundary_indices(X, max_hull_dim=5)
    assert ix.size >= 3
    assert ix.max() < X.shape[0]


def test_hull_two_points_returns_all():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(hull_boundary_indices(X), np.arange(2))


def test_hull_low_dim_returns_all():
    # d=1 -> n_comp < 2 -> early return of all indices.
    X = np.arange(5, dtype=np.float32).reshape(5, 1)
    np.testing.assert_array_equal(hull_boundary_indices(X), np.arange(5))


def test_hull_collinear_falls_back_to_pc1_extrema():
    # Rank-1 points in 3D -> Qhull degenerate -> argmin/argmax(PC1) fallback.
    X = np.array([[i, 2 * i, 3 * i] for i in range(6)], dtype=np.float32)
    ix = hull_boundary_indices(X)
    assert set(ix.tolist()) == {0, 5}


def test_boundary_doc_indices_maps_to_global_and_handles_empty():
    # cluster 0 = rows 0,2,4 ; cluster 1 = rows 1,3 ; cluster 2 = empty
    labels = np.array([0, 1, 0, 1, 0])
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((5, 4)).astype(np.float32)
    out = boundary_doc_indices_per_cluster(labels, vectors, 3)
    # 3 pts in cluster 0 -> triangle -> all 3 are hull vertices, mapped to globals.
    assert set(out[0].tolist()) == {0, 2, 4}
    assert set(out[1].tolist()) == {1, 3}
    assert out[2].size == 0
