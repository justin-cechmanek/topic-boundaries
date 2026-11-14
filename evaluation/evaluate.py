"""Reproduce geometric checks and prepare a small blinded human review."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations, zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import yaml

from topic_boundaries import (
    Datapoint,
    JsonlSource,
    PrecomputedEmbedder,
    Settings,
    boundary_doc_indices_per_cluster,
    cross_boundary_hits_for_all_clusters,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
    max_distance_rankings,
    voronoi_rankings,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("max_distance", "convex_hull", "voronoi", "cross_boundary", "random")
SHEET_FIELDS = (
    "review_id", "title", "abstract", "source_cluster_core", "other_cluster_core",
    "boundary_relevance_0_2", "cross_topic_value_0_2", "noise_0_1", "notes",
)


@dataclass(frozen=True)
class Candidate:
    method: str
    doc_id: str
    source_cluster: int


@contextmanager
def temporary_index(datapoints, vectors, labels, *, metric, settings):
    """Use a uniquely named index; never overwrite an existing project index."""
    schema = yaml.safe_load(settings.schema_path.read_text(encoding="utf-8"))
    suffix = uuid4().hex[:12]
    schema["index"]["name"] = f"tb_evaluation_{suffix}"
    schema["index"]["prefix"] = f"tb_eval_{suffix}"
    next(f for f in schema["fields"] if f["name"] == "embedding")["attrs"]["distance_metric"] = metric
    with TemporaryDirectory() as directory:
        path = Path(directory) / "schema.yml"
        path.write_text(yaml.safe_dump(schema), encoding="utf-8")
        index = index_clustered_datapoints(
            datapoints, vectors, labels,
            redis_url=settings.redis_url,
            schema_path=str(path),
            overwrite_index=True,
        )
        try:
            yield index
        finally:
            index.delete(drop=True)


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
    with temporary_index(datapoints, vectors, labels, metric="l2", settings=settings) as index:
        farthest = max_distance_rankings(index, centroids, 2, top_n_per_cluster=3)
        cross = cross_boundary_hits_for_all_clusters(index, centroids, 2, k_per_centroid=3)
    hull = boundary_doc_indices_per_cluster(labels, vectors, 2)
    voronoi = voronoi_rankings(datapoints, vectors, labels, centroids, 2, top_n_per_cluster=3)

    results = {
        method: {str(c): [] for c in range(2)}
        for method in METHODS[:-1]
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
    for left, right in combinations(METHODS[:-1], 2):
        a = {doc_id for ids in results[left].values() for doc_id in ids[:3]}
        b = {doc_id for ids in results[right].values() for doc_id in ids[:3]}
        overlap[f"{left}__{right}"] = round(len(a & b) / len(a | b), 3) if a | b else 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"points": points, "cluster_labels": dict(zip(points, map(int, labels))),
                    "rankings": results, "top3_jaccard": overlap,
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


def interleave_by_cluster(items: list[Candidate], n_clusters: int) -> list[Candidate]:
    buckets = [[] for _ in range(n_clusters)]
    for item in items:
        buckets[item.source_cluster].append(item)
    return [item for row in zip_longest(*buckets) for item in row if item is not None]


def select_candidates(
    pools: dict[str, list[Candidate]], n_clusters: int, per_method: int = 8
) -> list[Candidate]:
    """Select unique documents evenly by method and interleaved by cluster."""
    streams = {method: iter(interleave_by_cluster(pools[method], n_clusters)) for method in METHODS}
    selected: list[Candidate] = []
    used: set[str] = set()
    for _ in range(per_method):
        for method in METHODS:
            for item in streams[method]:
                if item.doc_id not in used:
                    selected.append(item)
                    used.add(item.doc_id)
                    break
            else:
                raise ValueError(f"Not enough unique candidates for {method}.")
    return selected


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_review(output: Path) -> None:
    sheet = output / "review_sheet.csv"
    key = output / "review_key.csv"
    if sheet.exists() or key.exists():
        raise FileExistsError("Review files already exist; choose a new output directory.")
    settings = Settings.from_config()
    datapoints = JsonlSource(ROOT / "datasets/arxiv/ml_ai_100.jsonl").load()
    vectors, labels, centroids = embed_and_cluster_datapoints(
        datapoints, n_clusters=5, embedding_model=settings.embedding_model,
        kmeans_random_state=settings.kmeans_random_state,
        kmeans_n_init=settings.kmeans_n_init,
    )
    n_clusters = len(centroids)
    with temporary_index(datapoints, vectors, labels, metric="cosine", settings=settings) as index:
        farthest = max_distance_rankings(index, centroids, n_clusters)
        cross = cross_boundary_hits_for_all_clusters(index, centroids, n_clusters, k_per_centroid=20)
    hull = boundary_doc_indices_per_cluster(labels, vectors, n_clusters)
    voronoi = voronoi_rankings(datapoints, vectors, labels, centroids, n_clusters)

    pools = {
        "max_distance": [Candidate("max_distance", h.doc_id, h.cluster_id) for h in farthest],
        "cross_boundary": [Candidate("cross_boundary", h.neighbor_doc_id, h.source_cluster_id) for h in cross],
        "convex_hull": [Candidate("convex_hull", datapoints[i].doc_id, c)
                        for c, indices in hull.items() for i in sorted(indices)],
        "voronoi": [Candidate("voronoi", h.doc_id, h.cluster_id) for h in voronoi],
        "random": [],
    }
    rng = random.Random(42)
    for c in range(n_clusters):
        indices = list(np.flatnonzero(labels == c))
        rng.shuffle(indices)
        pools["random"].extend(Candidate("random", datapoints[i].doc_id, c) for i in indices)

    selected = select_candidates(pools, n_clusters)
    rng.shuffle(selected)
    by_id = {point.doc_id: i for i, point in enumerate(datapoints)}

    def core_titles(cluster: int, exclude: str) -> str:
        indices = np.flatnonzero(labels == cluster)
        nearest = sorted(indices, key=lambda i: float(np.linalg.norm(vectors[i] - centroids[cluster])))
        titles = [
            str(datapoints[i].meta.get("title", "")) for i in nearest
            if datapoints[i].doc_id != exclude
        ]
        return "; ".join(titles[:2])

    review_rows = []
    key_rows = []
    for number, item in enumerate(selected, 1):
        idx = by_id[item.doc_id]
        own_cluster = int(labels[idx])
        if own_cluster != item.source_cluster:
            other = own_cluster
        else:
            distances = np.linalg.norm(centroids - centroids[item.source_cluster], axis=1)
            distances[item.source_cluster] = np.inf
            other = int(np.argmin(distances))
        review_id = f"R{number:03d}"
        review_rows.append({
            "review_id": review_id,
            "title": datapoints[idx].meta.get("title", ""),
            "abstract": datapoints[idx].body,
            "source_cluster_core": core_titles(item.source_cluster, item.doc_id),
            "other_cluster_core": core_titles(other, item.doc_id),
            "boundary_relevance_0_2": "",
            "cross_topic_value_0_2": "",
            "noise_0_1": "",
            "notes": "",
        })
        key_rows.append({
            "review_id": review_id, "method": item.method, "doc_id": item.doc_id,
            "source_cluster": item.source_cluster, "document_cluster": own_cluster,
        })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(sheet, SHEET_FIELDS, review_rows)
    write_csv(key, ("review_id", "method", "doc_id", "source_cluster", "document_cluster"), key_rows)
    (output / "metadata.json").write_text(
        json.dumps({"dataset": "datasets/arxiv/ml_ai_100.jsonl", "embedding_model": settings.embedding_model,
                    "n_clusters": n_clusters, "random_state": settings.kmeans_random_state,
                    "selection_seed": 42, "items_per_method": 8}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(sheet)


def report_review(output: Path) -> None:
    with (output / "review_sheet.csv").open(encoding="utf-8", newline="") as stream:
        sheet = list(csv.DictReader(stream))
    with (output / "review_key.csv").open(encoding="utf-8", newline="") as stream:
        key = {row["review_id"]: row["method"] for row in csv.DictReader(stream)}
    if len(sheet) != 40 or set(key) != {row["review_id"] for row in sheet}:
        raise ValueError("Expected 40 review rows matching the hidden key.")
    scores = defaultdict(list)
    for row in sheet:
        try:
            relevance = int(row["boundary_relevance_0_2"])
            connection = int(row["cross_topic_value_0_2"])
            noise = int(row["noise_0_1"])
        except ValueError as error:
            raise ValueError(f"Complete all ratings before reporting: {row['review_id']}") from error
        if relevance not in range(3) or connection not in range(3) or noise not in (0, 1):
            raise ValueError(f"Rating out of range: {row['review_id']}")
        scores[key[row["review_id"]]].append((relevance, connection, noise))
    lines = [
        "# Exploratory human review", "",
        "One reviewer rated 40 blinded papers from the 100-paper arXiv snapshot. "
        "These descriptive averages do not establish statistical significance or novelty.", "",
        "| Selection method | n | Mean boundary relevance (0-2) | Mean cross-topic value (0-2) | Noise flags |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        rows = scores[method]
        lines.append(
            f"| {method} | {len(rows)} | {np.mean([x[0] for x in rows]):.2f} | "
            f"{np.mean([x[1] for x in rows]):.2f} | {sum(x[2] for x in rows)} |"
        )
    lines += ["", "Review the individual comments in `review_sheet.csv` before interpreting the averages."]
    path = output / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("geometry", "sample", "report"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/round1"))
    args = parser.parse_args()
    if args.command == "geometry":
        geometric_check(args.output_dir / "geometry.json")
    elif args.command == "sample":
        prepare_review(args.output_dir)
    else:
        report_review(args.output_dir)


if __name__ == "__main__":
    main()
