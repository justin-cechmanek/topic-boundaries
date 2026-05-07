from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import Settings
from src.convex_hull.boundaries import boundary_doc_indices_per_cluster
from src.cross_boundary.boundaries import cross_boundary_hits_for_all_clusters
from src.documents import load_jsonl
from src.pdf_corpus import pdf_to_datapoints
from src.max_distance_sort.boundaries import boundary_rankings_for_all_clusters
from src.pipeline import run_pipeline


def _json_serial(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


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
    ])
    p.add_argument("--redis-url", default=None, help="Overrides REDIS_URL env.")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--schema", type=Path, default=None, help="Optional alternate schema.yml.")
    p.add_argument("--overwrite-index", action="store_true")
    p.add_argument("--top-n", type=int, default=20, help="Per-cluster cap where applicable.")
    p.add_argument("--centroid-neighbors-k", type=int, default=10)
    args = p.parse_args(argv)

    settings = Settings.from_env(schema_path=args.schema)
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

    state = run_pipeline(
        datapoints,
        redis_url=redis_url,
        n_clusters=args.n_clusters,
        embedding_model=model,
        schema_path=str(args.schema) if args.schema else None,
        overwrite_index=args.overwrite_index,
    )

    out: dict = {"method": args.method, "n_clusters": args.n_clusters}

    if args.method == "max_distance_sort":
        hits = boundary_rankings_for_all_clusters(
            state.indexed.index,
            state.centroids,
            state.indexed.n_clusters,
            top_n_per_cluster=args.top_n,
        )
        out["hits"] = [h.__dict__ for h in hits]

    elif args.method == "convex_hull":
        idx_map = boundary_doc_indices_per_cluster(
            state.labels,
            state.vectors,
            state.indexed.n_clusters,
        )
        hits = []
        for cid, gidx in idx_map.items():
            for rank, gi in enumerate(gidx):
                dp = state.datapoints[int(gi)]
                hits.append({
                    "cluster_id": int(cid),
                    "doc_id": dp.doc_id,
                    "body": dp.body[:500],
                    "rank_in_cluster": rank,
                })
        out["hits"] = hits

    elif args.method == "cross_boundary":
        hits = cross_boundary_hits_for_all_clusters(
            state.indexed.index,
            state.centroids,
            state.indexed.n_clusters,
            k_per_centroid=args.top_n,
        )
        out["hits"] = [h.__dict__ for h in hits]

    elif args.method == "centroid_neighbors":
        from src.centroid_neighbors import nearest_to_centroid

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
