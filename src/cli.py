from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import Settings
from src.convex_hull.boundaries import boundary_doc_indices_per_cluster
from src.cross_boundary.boundaries import cross_boundary_hits_for_all_clusters
from src.documents import Datapoint, load_jsonl
from src.pdf_corpus import pdf_to_datapoints
from src.centroid_neighbors import nearest_to_centroid
from src.max_distance_sort.boundaries import boundary_rankings_for_all_clusters
from src.voronoi_boundary.boundaries import boundary_rankings_for_all_clusters as voronoi_boundary_rankings_for_all_clusters
from src.pipeline import run_pipeline


def _json_serial(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _parse_kmeans_n_init(arg: str | None, fallback: int | str) -> int | str:
    if arg is None:
        return fallback
    s = arg.strip()
    if s.isdigit():
        return int(s)
    return s


def _doc_id_to_title(datapoints: list[Datapoint]) -> dict[str, str]:
    """Titles from JSONL meta (e.g. arXiv harvest); empty string if absent."""
    out: dict[str, str] = {}
    for d in datapoints:
        t = d.meta.get("title")
        out[d.doc_id] = str(t).strip() if t is not None else ""
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Topic boundaries pipeline and methods.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="JSONL path (body or abstract).")
    src.add_argument("--pdf", type=Path, help="PDF path; text is chunked into datapoints.")
    p.add_argument(
        "--pdf-max-chars",
        type=int,
        default=1500,
        help="Max characters per chunk when using --pdf.",
    )
    p.add_argument("--n-clusters", type=int, required=True)
    p.add_argument("--method", required=True, choices=[
        "max_distance_sort",
        "convex_hull",
        "cross_boundary",
        "centroid_neighbors",
        "voronoi_boundary",
    ])
    p.add_argument("--redis-url", default=None, help="Optional override for config redis_url.")
    p.add_argument("--config", type=Path, default=None, help="Optional config.yml path.")
    p.add_argument("--embedding-model", default=None, help="Optional override for config embedding model.")
    p.add_argument("--schema", type=Path, default=None, help="Optional alternate schema.yml.")
    p.add_argument("--overwrite-index", action="store_true")
    p.add_argument("--top-n", type=int, default=20, help="Per-cluster cap where applicable.")
    p.add_argument("--centroid-neighbors-k", type=int, default=10)
    p.add_argument(
        "--kmeans-random-state",
        type=int,
        default=None,
        help="KMeans random_state override (default from config.yml).",
    )
    p.add_argument(
        "--kmeans-n-init",
        default=None,
        metavar="VAL",
        help="KMeans n_init override, e.g. auto or a positive integer (default from config.yml).",
    )
    args = p.parse_args(argv)

    settings = Settings.from_config(config_path=args.config, schema_path=args.schema)
    redis_url = args.redis_url or settings.redis_url
    model = args.embedding_model or settings.embedding_model

    if args.pdf is not None:
        datapoints = pdf_to_datapoints(args.pdf, max_chars=args.pdf_max_chars)
    else:
        assert args.data is not None
        datapoints = load_jsonl(args.data)
    if len(datapoints) < args.n_clusters:
        print("Need at least as many datapoints as clusters.", file=sys.stderr)
        return 1

    kmeans_rs = (
        args.kmeans_random_state
        if args.kmeans_random_state is not None
        else settings.kmeans_random_state
    )
    kmeans_n_init = _parse_kmeans_n_init(args.kmeans_n_init, settings.kmeans_n_init)

    state = run_pipeline(
        datapoints,
        redis_url=redis_url,
        n_clusters=args.n_clusters,
        embedding_model=model,
        schema_path=str(args.schema) if args.schema else None,
        overwrite_index=args.overwrite_index,
        kmeans_random_state=kmeans_rs,
        kmeans_n_init=kmeans_n_init,
    )

    out: dict = {"method": args.method, "n_clusters": args.n_clusters}
    titles_by_doc = _doc_id_to_title(state.datapoints)

    if args.method == "max_distance_sort":
        hits = boundary_rankings_for_all_clusters(
            state.indexed.index,
            state.centroids,
            state.indexed.n_clusters,
            top_n_per_cluster=args.top_n,
        )
        boundary_by_cluster = [[] for _ in range(state.indexed.n_clusters)]
        for h in hits:
            boundary_by_cluster[h.cluster_id].append(
                {
                    "title": titles_by_doc.get(h.doc_id, ""),
                    "abstract": h.body,
                }
            )
        out["boundary_by_cluster"] = boundary_by_cluster

    elif args.method == "convex_hull":
        idx_map = boundary_doc_indices_per_cluster(
            state.labels,
            state.vectors,
            state.indexed.n_clusters,
        )
        boundary_by_cluster = [[] for _ in range(state.indexed.n_clusters)]
        for cid, gidx in idx_map.items():
            for gi in gidx:
                dp = state.datapoints[int(gi)]
                boundary_by_cluster[int(cid)].append(
                    {"title": titles_by_doc.get(dp.doc_id, ""), "abstract": dp.body}
                )
        out["boundary_by_cluster"] = boundary_by_cluster

    elif args.method == "cross_boundary":
        hits = cross_boundary_hits_for_all_clusters(
            state.indexed.index,
            state.centroids,
            state.indexed.n_clusters,
            k_per_centroid=args.top_n,
        )
        boundary_by_cluster = [[] for _ in range(state.indexed.n_clusters)]
        for h in hits:
            nid = h.neighbor_doc_id
            boundary_by_cluster[h.source_cluster_id].append(
                {
                    "title": titles_by_doc.get(nid, ""),
                    "abstract": h.neighbor_body,
                }
            )
        out["boundary_by_cluster"] = boundary_by_cluster

    elif args.method == "voronoi_boundary":
        hits = voronoi_boundary_rankings_for_all_clusters(
            state.datapoints,
            state.vectors,
            state.labels,
            state.centroids,
            state.indexed.n_clusters,
            top_n_per_cluster=args.top_n,
        )
        boundary_by_cluster = [[] for _ in range(state.indexed.n_clusters)]
        for h in hits:
            boundary_by_cluster[h.cluster_id].append(
                {
                    "title": titles_by_doc.get(h.doc_id, ""),
                    "abstract": h.body,
                    "voronoi_ratio": h.voronoi_ratio,
                    "nearest_other_cluster_id": h.nearest_other_cluster_id,
                }
            )
        out["boundary_by_cluster"] = boundary_by_cluster

    elif args.method == "centroid_neighbors":
        clusters_out = []
        for c in range(state.indexed.n_clusters):
            rows = nearest_to_centroid(
                state.indexed.index,
                state.centroids[c].tolist(),
                c,
                k=args.centroid_neighbors_k,
            )
            clusters_out.append({
                "cluster_id": c,
                "nearest_to_centroid": [
                    {
                        "doc_id": r["doc_id"],
                        "title": titles_by_doc.get(str(r["doc_id"]), ""),
                        "body": str(r.get("body", ""))[:500],
                        "vector_distance": r.get("vector_distance"),
                    }
                    for r in rows
                ],
            })
        out["clusters"] = clusters_out

    print(json.dumps(out, indent=2, default=_json_serial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
