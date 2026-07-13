"""Investigate boundaries, contours and edges in high-dimensional vector spaces.

Embed (or bring your own vectors) -> KMeans -> Redis vector index -> boundary
methods. Public API:

    from topic_boundaries import Datapoint, CsvSource, PrecomputedEmbedder, run_pipeline
"""

from __future__ import annotations

from topic_boundaries.config import Settings
from topic_boundaries.documents import Datapoint, datapoint_from_record, load_jsonl
from topic_boundaries.embedding import Embedder, HFTextEmbedder, PrecomputedEmbedder
from topic_boundaries.pipeline import (
    PipelineState,
    embed_and_cluster_datapoints,
    run_pipeline,
)
from topic_boundaries.sources import (
    ArxivSource,
    CsvSource,
    DataFrameSource,
    DataSource,
    JsonlSource,
    PdfSource,
    TextDirSource,
    register_source,
    SOURCES,
)

# Boundary methods (the two array-based rankers share a function name upstream,
# so they're aliased here).
from topic_boundaries.centroid_neighbors import nearest_to_centroid
from topic_boundaries.convex_hull.boundaries import boundary_doc_indices_per_cluster
from topic_boundaries.cross_boundary.boundaries import (
    CrossBoundaryHit,
    cross_boundary_hits_for_all_clusters,
)
from topic_boundaries.max_distance_sort.boundaries import (
    BoundaryHit,
    boundary_rankings_for_all_clusters as max_distance_rankings,
)
from topic_boundaries.voronoi_boundary.boundaries import (
    VoronoiBoundaryHit,
    boundary_rankings_for_all_clusters as voronoi_rankings,
)

__all__ = [
    "Settings",
    "Datapoint",
    "datapoint_from_record",
    "load_jsonl",
    "Embedder",
    "HFTextEmbedder",
    "PrecomputedEmbedder",
    "PipelineState",
    "run_pipeline",
    "embed_and_cluster_datapoints",
    "DataSource",
    "JsonlSource",
    "PdfSource",
    "ArxivSource",
    "CsvSource",
    "DataFrameSource",
    "TextDirSource",
    "register_source",
    "SOURCES",
    "nearest_to_centroid",
    "boundary_doc_indices_per_cluster",
    "cross_boundary_hits_for_all_clusters",
    "CrossBoundaryHit",
    "max_distance_rankings",
    "BoundaryHit",
    "voronoi_rankings",
    "VoronoiBoundaryHit",
]
