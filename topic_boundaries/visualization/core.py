from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from topic_boundaries.documents import Datapoint

# t-SNE tuning. Reduce features to TSNE_PRE_PCA_DIM before t-SNE so the pairwise
# neighbor search is cheaper (sklearn's own guidance); warn past
# TSNE_MAX_RECOMMENDED points since t-SNE is O(n^2).
TSNE_PRE_PCA_DIM = 50
TSNE_MAX_RECOMMENDED = 10_000


def _doc_fingerprint(dp: Datapoint) -> str:
    title = str(dp.meta.get("title") or "").strip()
    return f"{title}\0{dp.body.strip()}"


def _item_fingerprint(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "").strip()
    text = str(item.get("abstract") or item.get("body") or "").strip()
    return f"{title}\0{text}"


def boundary_highlight_indices(
    results: dict[str, Any],
    datapoints: list[Datapoint],
) -> set[int]:
    """Indices of documents that appear in boundary (or neighbor) JSON output."""
    method = str(results.get("method") or "")
    if method == "centroid_neighbors":
        doc_ids: set[str] = set()
        for c in results.get("clusters") or []:
            for row in c.get("nearest_to_centroid") or []:
                did = str(row.get("doc_id") or "")
                if did:
                    doc_ids.add(did)
        return {i for i, d in enumerate(datapoints) if d.doc_id in doc_ids}

    fps: set[str] = set()
    for cluster_list in results.get("boundary_by_cluster") or []:
        for item in cluster_list:
            fps.add(_item_fingerprint(item))
    out: set[int] = set()
    for i, dp in enumerate(datapoints):
        if _doc_fingerprint(dp) in fps:
            out.add(i)
    return out


def project_to_2d(
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    method: str,
    random_state: int,
    tsne_perplexity: float,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project docs + centroids to 2D or 3D.

    Returns (xy_docs, xy_centroids), each shape (n, n_components).
    ``n_components`` must be 2 or 3 (build_plotly_figure picks Scatter vs
    Scatter3d from the coordinate width).

    Methods: "pca", "tsne", "umap" (needs umap-learn), or "precomputed" —
    pass coords that are already reduced (vectors/centroids used as-is).
    """
    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")
    m = method.lower().strip()
    if m == "precomputed":
        for name, arr in (("vectors", vectors), ("centroids", centroids)):
            if arr.shape[1] != n_components:
                raise ValueError(
                    f"precomputed {name} has width {arr.shape[1]}, "
                    f"expected n_components={n_components}"
                )
        return vectors.astype(np.float64), centroids.astype(np.float64)
    if m == "umap":
        try:
            from umap import UMAP
        except ImportError as e:
            raise ImportError(
                "UMAP projection requires umap-learn. Install with: "
                'pip install umap-learn (or pip install -e ".[viz]")'
            ) from e
        reducer = UMAP(n_components=n_components, random_state=random_state)
        xy_docs = reducer.fit_transform(vectors)
        # UMAP (unlike t-SNE) has transform(), so centroids project properly.
        xy_centroids = reducer.transform(centroids)
        return xy_docs.astype(np.float64), xy_centroids.astype(np.float64)
    if m == "pca":
        pca = PCA(n_components=n_components, random_state=random_state)
        xy_docs = pca.fit_transform(vectors)
        xy_centroids = pca.transform(centroids)
        return xy_docs.astype(np.float64), xy_centroids.astype(np.float64)
    if m == "tsne":
        n_docs = len(vectors)
        if n_docs > TSNE_MAX_RECOMMENDED:
            warnings.warn(
                f"t-SNE is O(n^2); projecting {n_docs} points will be slow. "
                "Consider method='umap' or sampling the points first.",
                stacklevel=2,
            )
        # t-SNE has no transform(), so embed docs + centroids jointly: centroids
        # land in the same space as real points (no faked means, no NaN for
        # empty clusters — a KMeans centroid always has a position).
        stacked = np.vstack([vectors, centroids])
        if stacked.shape[1] > TSNE_PRE_PCA_DIM:
            pre_k = min(TSNE_PRE_PCA_DIM, stacked.shape[0] - 1)
            stacked = PCA(
                n_components=pre_k, random_state=random_state
            ).fit_transform(stacked)
        n_join = stacked.shape[0]
        perp = float(min(tsne_perplexity, max(5.0, (n_join - 1) / 3.0)))
        perp = min(perp, max(1.5, n_join - 1 - 1e-6))
        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
        )
        xy = tsne.fit_transform(stacked)
        return xy[:n_docs].astype(np.float64), xy[n_docs:].astype(np.float64)
    raise ValueError(
        f"Unknown projection: {method!r} (use pca, tsne, umap, or precomputed)"
    )
