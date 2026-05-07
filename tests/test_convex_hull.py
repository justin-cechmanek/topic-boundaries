import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.convex_hull.boundaries import hull_boundary_indices


def test_hull_returns_subset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8)).astype(np.float32)
    ix = hull_boundary_indices(X, max_hull_dim=5)
    assert ix.size >= 3
    assert ix.max() < X.shape[0]
