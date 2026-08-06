from __future__ import annotations

import json
import warnings
from pathlib import Path
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


def boundary_highlight_indices(
    results: dict[str, Any],
    datapoints: list[Datapoint],
) -> set[int]:
    """Indices of documents that appear in boundary (or neighbor) JSON output.

    Every result item carries the document's ``doc_id`` (see
    ``write_boundary_results``), so matching is a doc_id lookup regardless of
    method — centroid_neighbors nests under ``clusters``, the rest under
    ``boundary_by_cluster``.
    """
    doc_ids: set[str] = set()
    if str(results.get("method") or "") == "centroid_neighbors":
        for c in results.get("clusters") or []:
            for row in c.get("nearest_to_centroid") or []:
                doc_ids.add(str(row.get("doc_id") or ""))
    else:
        for cluster_list in results.get("boundary_by_cluster") or []:
            for item in cluster_list:
                doc_ids.add(str(item.get("doc_id") or ""))
    doc_ids.discard("")
    return {i for i, d in enumerate(datapoints) if d.doc_id in doc_ids}


def write_boundary_results(
    path: str | Path,
    *,
    method: str,
    n_clusters: int,
    doc_ids_by_cluster: list[list[str]],
    datapoints: list[Datapoint] | None = None,
) -> dict[str, Any]:
    """Serialize per-cluster boundary hits to the JSON the viz helpers consume.

    ``doc_ids_by_cluster[c]`` is the list of hit doc_ids for cluster ``c`` (the
    boundary methods' hits all carry a doc_id). Each item is written as
    ``{"doc_id": ...}``; when ``datapoints`` is given, ``title``/``abstract`` are
    added for human-readable inspection — ``boundary_highlight_indices`` keys on
    doc_id either way.
    """
    by_id = {d.doc_id: d for d in datapoints} if datapoints is not None else {}
    clusters: list[list[dict[str, str]]] = []
    for ids in doc_ids_by_cluster:
        items: list[dict[str, str]] = []
        for did in ids:
            item = {"doc_id": did}
            dp = by_id.get(did)
            if dp is not None:
                item["title"] = str(dp.meta.get("title") or "")
                item["abstract"] = dp.body
            items.append(item)
        clusters.append(items)
    payload: dict[str, Any] = {
        "method": method,
        "n_clusters": n_clusters,
        "boundary_by_cluster": clusters,
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return payload


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
