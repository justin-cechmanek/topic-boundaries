from __future__ import annotations

import html
from typing import Any

import numpy as np

from src.documents import Datapoint


def _try_plotly(
    datapoints: list[Datapoint],
    labels: np.ndarray,
    highlight: set[int],
    xy_docs: np.ndarray,
    xy_centroids: np.ndarray,
    *,
    title: str,
) -> Any:
    import plotly.graph_objects as go

    palette = [
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
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    traces: list[Any] = []

    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        color = palette[c % len(palette)]
        regular = [i for i in idx.tolist() if i not in highlight]
        marked = [i for i in idx.tolist() if i in highlight]
        for subset, size, name_suffix in (
            (regular, 6, ""),
            (marked, 11, " (boundary / centroid-nb)"),
        ):
            if not subset:
                continue
            xs = xy_docs[subset, 0]
            ys = xy_docs[subset, 1]
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
                go.Scatter(
                    x=xs,
                    y=ys,
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
            go.Scatter(
                x=xy_centroids[valid_c, 0],
                y=xy_centroids[valid_c, 1],
                mode="markers",
                name="centroids (2D)",
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
    fig.update_xaxes(title="dim 1")
    fig.update_yaxes(title="dim 2", scaleanchor="x", scaleratio=1)
    return fig


def build_plotly_figure(
    datapoints: list[Datapoint],
    labels: np.ndarray,
    highlight: set[int],
    xy_docs: np.ndarray,
    xy_centroids: np.ndarray,
    *,
    title: str,
) -> Any:
    return _try_plotly(
        datapoints,
        labels,
        highlight,
        xy_docs,
        xy_centroids,
        title=title,
    )
