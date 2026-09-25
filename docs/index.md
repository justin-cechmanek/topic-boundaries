# Topic Boundaries

Find documents near the edges of clusters in a text corpus or another vector space.

Topic Boundaries loads data, creates embeddings or accepts vectors you already have, clusters them with K-means, and offers several ways to rank boundary documents.

## Quick Start

From the repository root, install the package and start the local Redis Stack:

```bash
pip install -e ".[dev]"
docker compose up -d
```

Run a first boundary query with the included sample corpus:

```python
from topic_boundaries import JsonlSource, Settings, max_distance_rankings, run_pipeline

settings = Settings.from_config()
datapoints = JsonlSource("datasets/sample.jsonl").load()
state = run_pipeline(
    datapoints,
    redis_url=settings.redis_url,
    n_clusters=5,
    embedding_model=settings.embedding_model,
    schema_path=None,
    overwrite_index=True,
)
hits = max_distance_rankings(
    state.indexed.index,
    state.centroids,
    state.indexed.n_clusters,
    top_n_per_cluster=3,
)
for hit in hits:
    print(hit.cluster_id, hit.doc_id, hit.distance)
```

The first run may download the embedding model. See [Getting Started](user_guide/getting_started.md) for a walkthrough, or [Precomputed Vectors](examples/precomputed_vectors.md) to use vectors without a model download.

## Explore the Docs

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📖 Concepts
:link: concepts/index
:link-type: doc

Understand the pipeline and compare boundary methods.
:::

:::{grid-item-card} 🚀 User Guides
:link: user_guide/index
:link-type: doc

Install the project, load data, and run an analysis.
:::

:::{grid-item-card} 💡 Examples
:link: examples/index
:link-type: doc

Start with a complete vector-only example.
:::

:::{grid-item-card} 📚 API Reference
:link: api/index
:link-type: doc

Find the main classes, functions, and return values.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

Concepts <concepts/index>
User Guides <user_guide/index>
Examples <examples/index>
API <api/index>
```
