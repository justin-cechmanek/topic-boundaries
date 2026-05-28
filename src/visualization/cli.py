from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import Settings, _parse_n_init
from src.documents import load_jsonl
from src.pipeline import embed_and_cluster_datapoints
from src.visualization.core import boundary_highlight_indices, project_to_2d
from src.visualization.figure import build_plotly_figure


def main(argv: list[str] | None = None) -> int:
    try:
        import plotly  # noqa: F401
    except ImportError:
        print(
            "Plotly is required for visualization. Install with: "
            "pip install 'topic-boundaries[viz]'",
            file=sys.stderr,
        )
        return 1

    p = argparse.ArgumentParser(
        description="Interactive 2D view of clusters, centroids, and boundary results.",
    )
    p.add_argument("--data", type=Path, required=True, help="Same JSONL as the pipeline run.")
    p.add_argument(
        "--results",
        type=Path,
        required=True,
        help="JSON file produced by topic-boundaries CLI.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write HTML here (default: results path with .html suffix).",
    )
    p.add_argument(
        "--projection",
        choices=("pca", "tsne"),
        default="pca",
        help="2D reduction method (default pca).",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--kmeans-random-state", type=int, default=None)
    p.add_argument(
        "--kmeans-n-init",
        default=None,
        metavar="VAL",
        help="KMeans n_init, e.g. auto or integer (match your pipeline run).",
    )
    p.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (clamped vs sample size).",
    )
    args = p.parse_args(argv)

    settings = Settings.from_config(config_path=args.config, schema_path=None)
    model = args.embedding_model or settings.embedding_model
    kmeans_rs = (
        args.kmeans_random_state
        if args.kmeans_random_state is not None
        else settings.kmeans_random_state
    )
    kmeans_n_init = _parse_n_init(args.kmeans_n_init, settings.kmeans_n_init)

    results_path = args.results
    with results_path.open(encoding="utf-8") as f:
        results = json.load(f)

    n_clusters = int(results["n_clusters"])
    datapoints = load_jsonl(args.data)
    if len(datapoints) < n_clusters:
        print("Need at least as many datapoints as n_clusters.", file=sys.stderr)
        return 1

    highlight = boundary_highlight_indices(results, datapoints)
    if not highlight:
        print(
            "Warning: no documents matched boundary/results highlights "
            "(check --data matches the run, and kmeans random_state matches).",
            file=sys.stderr,
        )

    vectors, labels, centroids, _ = embed_and_cluster_datapoints(
        datapoints,
        n_clusters=n_clusters,
        embedding_model=model,
        kmeans_random_state=kmeans_rs,
        kmeans_n_init=kmeans_n_init,
    )

    xy_docs, xy_centroids = project_to_2d(
        vectors,
        labels,
        centroids,
        method=args.projection,
        random_state=kmeans_rs,
        tsne_perplexity=args.tsne_perplexity,
    )

    method = str(results.get("method", ""))
    title = (
        f"{method} · {args.projection.upper()} · k={n_clusters} · "
        f"n={len(datapoints)} · highlighted={len(highlight)}"
    )
    fig = build_plotly_figure(
        datapoints,
        labels,
        highlight,
        xy_docs,
        xy_centroids,
        title=title,
    )

    out = args.output
    if out is None:
        out = results_path.with_suffix(".html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
