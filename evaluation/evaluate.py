"""Compare four boundary methods on controlled 2D geometry."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import yaml

from topic_boundaries import (
    Datapoint,
    PrecomputedEmbedder,
    Settings,
    boundary_doc_indices_per_cluster,
    cross_boundary_hits_for_all_clusters,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
    max_distance_rankings,
    voronoi_rankings,
)


METHODS = ("max_distance", "convex_hull", "voronoi", "cross_boundary")


def create_evaluation_index(datapoints, vectors, labels, *, metric, settings):
    """Create a uniquely named index that the caller deletes when finished."""
    schema = yaml.safe_load(settings.schema_path.read_text(encoding="utf-8"))
    suffix = uuid4().hex[:12]
    schema["index"]["name"] = f"tb_evaluation_{suffix}"
    schema["index"]["prefix"] = f"tb_eval_{suffix}"
    next(f for f in schema["fields"] if f["name"] == "embedding")["attrs"]["distance_metric"] = metric
    with TemporaryDirectory() as directory:
        path = Path(directory) / "schema.yml"
        path.write_text(yaml.safe_dump(schema), encoding="utf-8")
        return index_clustered_datapoints(
            datapoints, vectors, labels,
            redis_url=settings.redis_url,
            schema_path=str(path),
            overwrite_index=True,
        )


def geometric_check(output: Path) -> None:
    """Show what each method ranks on a known bridge/outlier/elongation case."""
    points = {
        "A-core": (-3.0, 0.0), "A-upper": (-2.8, 1.3),
        "A-lower": (-2.8, -1.3), "A-bridge": (-0.6, 0.0),
        "A-outlier": (-6.0, 0.5), "A-mid": (-2.2, 0.1),
        "B-core": (3.0, 0.0), "B-upper": (2.8, 1.3),
        "B-lower": (2.8, -1.3), "B-bridge": (0.6, 0.0),
        "B-mid": (2.2, -0.1),
    }
    datapoints = [
        Datapoint(doc_id=name, vector=np.array(xy, dtype=np.float32))
        for name, xy in points.items()
    ]
    vectors, labels, centroids = embed_and_cluster_datapoints(
        datapoints, n_clusters=2, embedder=PrecomputedEmbedder()
    )
    settings = Settings.from_config()
    index = create_evaluation_index(datapoints, vectors, labels, metric="l2", settings=settings)
    try:
        farthest = max_distance_rankings(index, centroids, 2, top_n_per_cluster=3)
        cross = cross_boundary_hits_for_all_clusters(index, centroids, 2, k_per_centroid=3)
    finally:
        index.delete(drop=True)
    hull = boundary_doc_indices_per_cluster(labels, vectors, 2)
    voronoi = voronoi_rankings(datapoints, vectors, labels, centroids, 2, top_n_per_cluster=3)

    results = {
        method: {str(c): [] for c in range(2)}
        for method in METHODS
    }
    for hit in farthest:
        results["max_distance"][str(hit.cluster_id)].append(hit.doc_id)
    for hit in cross:
        results["cross_boundary"][str(hit.source_cluster_id)].append(hit.neighbor_doc_id)
    for hit in voronoi:
        results["voronoi"][str(hit.cluster_id)].append(hit.doc_id)
    for c, indices in hull.items():
        results["convex_hull"][str(c)] = sorted(datapoints[i].doc_id for i in indices)

    a_cluster = int(labels[list(points).index("A-core")])
    bridge_cluster = int(labels[list(points).index("A-bridge")])
    assert results["max_distance"][str(a_cluster)][0] == "A-outlier"
    assert "A-outlier" in results["convex_hull"][str(a_cluster)]
    overlap = {}
    for left, right in combinations(METHODS, 2):
        a = {doc_id for ids in results[left].values() for doc_id in ids}
        b = {doc_id for ids in results[right].values() for doc_id in ids}
        overlap[f"{left}__{right}"] = round(len(a & b) / len(a | b), 3) if a | b else 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"points": points, "cluster_labels": dict(zip(points, map(int, labels))),
                    "rankings": results, "candidate_set_jaccard": overlap,
                    "overlap_selection": "Top three per cluster for ranked methods and all hull vertices.",
                    "observations": {
                        "A_outlier_ranked_farthest": True,
                        "A_outlier_on_hull": True,
                        "A_bridge_reassigned_away_from_A_core": bridge_cluster != a_cluster,
                        "A_bridge_voronoi_rank_in_assigned_cluster":
                            results["voronoi"][str(bridge_cluster)].index("A-bridge") + 1,
                    }}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("geometry",))
    parser.add_argument("--output-dir", type=Path, default=Path("build/evaluation"))
    args = parser.parse_args()
    geometric_check(args.output_dir / "geometry.json")


if __name__ == "__main__":
    main()
