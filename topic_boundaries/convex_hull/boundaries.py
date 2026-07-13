from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


def hull_boundary_indices(
    vectors: np.ndarray,
    *,
    max_hull_dim: int = 12,
    random_state: int = 0,
) -> np.ndarray:
    """
    Indices of points on an approximate convex hull.

    Full-dimensional Quickhull becomes impractical for typical embedding dims;
    we follow the README's Quickhull idea on a PCA-compressed representation of
    the cluster (still computed via Qhull underneath SciPy's ConvexHull).
    """
    n, d = vectors.shape
    if n <= 2:
        return np.arange(n)
    n_comp = min(max_hull_dim, n - 1, d)
    if n_comp < 2:
        return np.arange(n)
    X = PCA(n_components=n_comp, random_state=random_state).fit_transform(vectors)
    try:
        hull = ConvexHull(X, qhull_options="Qt")
    except Exception:
        # Degenerate configurations (e.g. coplanar); fall back to extrema on first PC.
        pc1 = X[:, 0]
        i_min = int(np.argmin(pc1))
        i_max = int(np.argmax(pc1))
        return np.unique(np.array([i_min, i_max]))
    return np.asarray(hull.vertices, dtype=np.int64)


def boundary_doc_indices_per_cluster(
    labels: np.ndarray,
    vectors: np.ndarray,
    n_clusters: int,
    **kwargs,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for c in range(n_clusters):
        mask = labels == c
        idx_global = np.flatnonzero(mask)
        if idx_global.size == 0:
            out[c] = np.array([], dtype=np.int64)
            continue
        local = vectors[mask]
        hull_local = hull_boundary_indices(local, **kwargs)
        out[c] = idx_global[hull_local]
    return out
