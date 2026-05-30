from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.documents import Datapoint


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
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xy_docs, xy_centroids), each shape (n, 2)."""
    m = method.lower().strip()
    if m == "pca":
        pca = PCA(n_components=2, random_state=random_state)
        xy_docs = pca.fit_transform(vectors)
        xy_centroids = pca.transform(centroids)
        return xy_docs.astype(np.float64), xy_centroids.astype(np.float64)
    if m == "tsne":
        n = len(vectors)
        perp = float(min(tsne_perplexity, max(5.0, (n - 1) / 3.0)))
        perp = min(perp, max(1.5, n - 1 - 1e-6))
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
        )
        xy_docs = tsne.fit_transform(vectors)
        # t-SNE has no transform; place centroids at KMeans member means in 2D.
        xy_centroids = np.zeros((centroids.shape[0], 2), dtype=np.float64)
        for c in range(centroids.shape[0]):
            mask = labels == c
            if mask.any():
                xy_centroids[c] = xy_docs[mask].mean(axis=0)
            else:
                xy_centroids[c] = np.nan
        return xy_docs.astype(np.float64), xy_centroids
    raise ValueError(f"Unknown projection: {method!r} (use pca or tsne)")
