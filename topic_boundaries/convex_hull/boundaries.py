from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


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

    Degenerate clusters (coplanar/coincident after PCA) can't form a hull; those
    fall back to the two PC1 extrema and log a WARNING so the fallback rate is
    visible rather than silent. Non-degenerate Qhull failures now propagate.
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
    except QhullError as e:
        # ponytail: hull still works up to max_hull_dim; "drop hull for high-D"
        # (TODO PR-P3) stays deferred until the logged fallback rate says it hurts.
        logger.warning(
            "convex hull degenerate for n=%d points in %d-D (PCA of %d-D); "
            "falling back to PC1 extrema. Qhull: %s",
            n,
            n_comp,
            d,
            str(e).splitlines()[0] if str(e) else e.__class__.__name__,
        )
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
