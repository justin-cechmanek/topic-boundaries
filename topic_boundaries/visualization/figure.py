from __future__ import annotations

import html
import warnings
from typing import Any

import numpy as np

from topic_boundaries.documents import Datapoint

# Plotly's default qualitative palette; readable for the common small-k case.
_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def _cluster_colors(n_clusters: int) -> list[str]:
    """One distinct color per cluster, scaling past the 10-color palette."""
    if n_clusters <= len(_PALETTE):
        return _PALETTE[:n_clusters]
    # Sample a perceptually-varied continuous scale so any k gets distinct hues.
    from plotly.colors import sample_colorscale

    return sample_colorscale("Turbo", [i / (n_clusters - 1) for i in range(n_clusters)])


def _render_indices(
    n_docs: int, highlight: set[int], max_points: int | None
) -> set[int] | None:
    """Doc indices to draw: all, or a sample capped at max_points (highlights kept).

    Returns None to mean "render everything".
    """
    if max_points is None or n_docs <= max_points:
        return None
    keep = {i for i in highlight if 0 <= i < n_docs}
    others = np.array([i for i in range(n_docs) if i not in keep])
    budget = max(0, max_points - len(keep))
    if budget < len(others):
        rng = np.random.default_rng(0)  # deterministic decimation
        keep.update(int(i) for i in rng.choice(others, size=budget, replace=False))
    else:
        keep.update(int(i) for i in others)
    warnings.warn(
        f"Decimated {n_docs} points to {len(keep)} for rendering "
        f"(max_points={max_points}); highlighted points are always kept.",
        stacklevel=2,
    )
    return keep


def build_plotly_figure(
    datapoints: list[Datapoint],
    labels: np.ndarray,
    highlight: set[int],
    xy_docs: np.ndarray,
    xy_centroids: np.ndarray,
    *,
    title: str,
    max_points: int | None = None,
) -> Any:
    import plotly.graph_objects as go

    # 2D or 3D follows the coordinate width; project_to_2d(..., n_components=3)
    # produces (n, 3) arrays -> Scatter3d. 2D uses Scattergl (WebGL) so large n
    # stays interactive; Scatter3d is already WebGL-backed.
    is_3d = xy_docs.shape[1] >= 3
    Scatter = go.Scatter3d if is_3d else go.Scattergl

    def coords(arr: np.ndarray, subset: Any) -> dict[str, Any]:
        d = {"x": arr[subset, 0], "y": arr[subset, 1]}
        if is_3d:
            d["z"] = arr[subset, 2]
        return d

    n_clusters = int(labels.max()) + 1 if labels.size else 0
    colors = _cluster_colors(n_clusters)
    render = _render_indices(len(datapoints), highlight, max_points)
    traces: list[Any] = []

    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        color = colors[c]
        members = [i for i in idx.tolist() if render is None or i in render]
        regular = [i for i in members if i not in highlight]
        marked = [i for i in members if i in highlight]
        for subset, size, name_suffix in (
            (regular, 6, ""),
            (marked, 11, " (boundary / centroid-nb)"),
        ):
            if not subset:
                continue
            texts = []
            for i in subset:
                dp = datapoints[i]
                t = html.escape(str(dp.meta.get("title") or "")[:200])
                body_esc = html.escape(dp.body[:400])
                if len(dp.body) > 400:
                    body_esc += "…"
                texts.append(
                    f"<b>{t}</b><br>doc_id: {html.escape(dp.doc_id)}<br>"
                    f"cluster: {c}<br>{body_esc}"
                )
            traces.append(
                Scatter(
                    **coords(xy_docs, subset),
                    mode="markers",
                    name=f"cluster {c}{name_suffix}",
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(width=(2 if name_suffix else 0), color="black"),
                    ),
                    text=texts,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    valid_c = ~np.isnan(xy_centroids).any(axis=1)
    if valid_c.any():
        traces.append(
            Scatter(
                **coords(xy_centroids, valid_c),
                mode="markers",
                name=f"centroids ({xy_centroids.shape[1]}D)",
                marker=dict(
                    size=16,
                    symbol="x",
                    color="black",
                    line=dict(width=2, color="white"),
                ),
                text=[f"cluster {i}" for i in range(len(xy_centroids)) if valid_c[i]],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        dragmode="pan",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=40, r=160, t=50, b=40),
    )
    if is_3d:
        fig.update_layout(
            scene=dict(xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="dim 3")
        )
    else:
        fig.update_xaxes(title="dim 1")
        fig.update_yaxes(title="dim 2", scaleanchor="x", scaleratio=1)
    return fig
