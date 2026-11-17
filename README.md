# Topic Boundaries

This project allows you to identify the specific content at the edges of topics within a large corpus of text.

## What is a topic boundary?

A **topic boundary** is the set of datapoints (ideas, statements, concepts) that exist at the periphery of our understanding of a specific field of knowledge. It is the edge—or boundary—of our knowledge of a subject. This is generally where new research and innovation is focused. If we consider a specific field of knowledge we can assume the centroid of that field is well explored, so it follows that the remaining interesting areas of future discovery and innovation exist further away. This repository is a collection of tools and methods that attempts to formally quantify and identify these boundaries using multiple different techniques.

The library loads documents or existing vectors, clusters them with K-means,
and offers four boundary methods. Some methods use a Redis vector index; others
work directly on the vectors in memory.

## Quickstart

Run these commands from a checkout with Python 3.10 or newer and Docker:

```bash
pip install -e ".[dev]"
docker compose up -d
```

The sample corpus has 20 records. This example embeds them, forms five clusters,
indexes them, and prints the three farthest documents in each cluster:

```python
from topic_boundaries import (
    JsonlSource,
    Settings,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
    max_distance_rankings,
)

settings = Settings.from_config()
datapoints = JsonlSource("datasets/sample.jsonl").load()
vectors, labels, centroids = embed_and_cluster_datapoints(
    datapoints,
    n_clusters=5,
    embedding_model=settings.embedding_model,
    kmeans_random_state=settings.kmeans_random_state,
    kmeans_n_init=settings.kmeans_n_init,
)
index = index_clustered_datapoints(
    datapoints,
    vectors,
    labels,
    redis_url=settings.redis_url,
    schema_path=str(settings.schema_path),
    overwrite_index=True,
)
for hit in max_distance_rankings(index, centroids, n_clusters=5, top_n_per_cluster=3):
    print(hit.cluster_id, hit.doc_id, hit.distance)
```

The first run may download the embedding model. `overwrite_index=True` replaces
the configured Redis index **and its existing documents**; use a dedicated index
name in a custom schema when keeping existing data matters.

`Settings.from_config()` reads `config.yml` in a source checkout, with
`REDIS_URL` and `EMBEDDING_MODEL` environment variables taking precedence. The
example passes its K-means and schema settings explicitly. To use vectors you
already have, pass `embedder=PrecomputedEmbedder()` instead of
`embedding_model=...`; see [Data Sources](docs/user_guide/data_sources.md).

## Choose a method

| Method | Question it asks | Needs Redis? |
| --- | --- | --- |
| `max_distance_rankings` | Which documents are farthest from their own centroid? | Yes |
| `boundary_doc_indices_per_cluster` | Which documents form a convex hull after PCA compression? | No |
| `voronoi_rankings` | Which documents sit near a K-means decision boundary? | No |
| `cross_boundary_hits_for_all_clusters` | Which other-cluster documents are nearest to a centroid? | Yes |

`nearest_to_centroid` instead finds representative documents near a topic's
core. The methods answer different questions; their scores are not one common
measure of boundary quality. See [Boundary Methods](docs/concepts/boundary_methods.md)
for their definitions and limits.

## Data sources and distance metrics

A source returns `list[Datapoint]`. Each datapoint has a `doc_id` and text in
`body` or a precomputed `vector`. Built-in sources cover JSONL, JSON arrays,
CSV, in-memory records, text and Markdown files, PDFs, and arXiv papers.
PDF ingestion needs `pip install -e ".[corpus]"`; visualization needs
`pip install -e ".[viz]"`. See [Data Sources](docs/user_guide/data_sources.md)
for loader options and the precomputed-vector path.

K-means and the Voronoi method use Euclidean geometry. The default Redis schema
uses cosine distance, and `HFTextEmbedder` normalizes its text vectors.
`PrecomputedEmbedder` preserves supplied vectors. A custom Redis schema can
use Euclidean distance. Choose a metric that fits the vectors and read the
[metric notes](docs/concepts/architecture.md#index-schema) before comparing
scores across methods.

All datapoints and vectors are held in memory while clustering. Full rankings,
and Euclidean farthest rankings, scan a cluster's indexed vectors. This is not
yet a streaming or proven large-corpus workflow.

## Tests and documentation

```bash
pip install -e ".[dev,corpus,viz,docs]"
docker compose up -d
pytest -q
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Redis integration tests skip when Redis is unavailable; CI supplies a Redis
Stack service. The documentation is also hosted at
https://justin-cechmanek.github.io/topic-boundaries-docs/. Only generated docs
are published there; the package repository remains private. Start with
[Getting Started](docs/user_guide/getting_started.md) or the
[Examples](docs/examples/index.md) and [API Reference](docs/api/index.md).
