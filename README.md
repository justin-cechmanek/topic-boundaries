# Topic Boundaries

This project allows you to identify the specific content at the edges of topics within a large corpus of text.

## What is a topic boundary?

A **topic boundary** is the set of datapoints (ideas, statements, concepts) that exist at the periphery of our understanding of a specific field of knowledge. It is the edge—or boundary—of our knowledge of a subject. This is generally where new research and innovation is focused. If we consider a specific field of knowledge we can assume the centroid of that field is well explored, so it follows that the remaining interesting areas of future discovery and innovation exist further away. This repository is a collection of tools and methods that attempts to formally quantify and identify these boundaries using multiple different techniques.

## Quickstart

1. **Python 3.10+** and a virtualenv (see `.python-version` if you use pyenv).

2. **Install** (dependencies are defined in `pyproject.toml`; editable install is recommended):

   ```bash
   pip install -e ".[dev]"
   ```

   Optional PDF ingestion: `pip install -e ".[corpus]"` (installs `pypdf`).

3. **Redis Stack** (RediSearch + vector indexing), e.g.:

   ```bash
   docker compose up -d
   ```

4. **Review or edit defaults** in `config.yml`:

   - `redis_url`
   - `embedding_model`
   - `kmeans.random_state`
   - `kmeans.n_init`

5. **Use it from Python.** Load a source, run the pipeline, then apply a boundary method:

   ```python
   from topic_boundaries import JsonlSource, Settings, run_pipeline, max_distance_rankings

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
       state.indexed.index, state.centroids, state.indexed.n_clusters
   )
   ```

   `Settings.from_config()` reads defaults (Redis URL, embedding model, KMeans
   params) from `config.yml`.

The package installs and imports as **`topic_boundaries`** (e.g. `from topic_boundaries import Datapoint, CsvSource, run_pipeline`).

## Testing

```bash
pip install -e ".[dev,corpus]"
docker compose up -d          # Redis Stack, for the integration tests
pytest                         # or: pytest --cov=topic_boundaries --cov-report=term-missing
```

Tests use no mocks — they exercise real code paths. The `integration`-marked
tests run against a live Redis Stack (and the embedding pipeline uses the real
model); they **skip automatically** when Redis is unreachable, so `pytest` still
passes without it. Override the Redis target with `REDIS_URL=...`.

## The process

Each technique lives under its own subdirectory inside **`topic_boundaries`**. They share these steps:

1. Choose a dataset to analyse.
2. Load the raw text and parse it into datapoints.
3. Embed the datapoints with your chosen model and load vectors into a RedisVL vector search index.
4. Run K-means clustering to form topic clusters.
5. Tag each datapoint with its cluster id in the index.
6. Compute cluster centroids (mean of vectors per cluster, assuming roughly spherical clusters).

## Specific techniques

### 1. Maximum distance sorting

For each cluster, rank documents by greatest distance to the centroid (farthest-first boundary listing).

### 2. Convex hull

Use [convex hull algorithms](https://en.wikipedia.org/wiki/Convex_hull_algorithms), specifically [Quickhull](https://en.wikipedia.org/wiki/Quickhull), on a PCA-compressed representation of each cluster to approximate hull vertices (see `topic_boundaries/convex_hull/`).

### 3. Cross boundary

Using one cluster’s centroid, find the nearest datapoints that are **not** in that cluster—neighbors on other clusters’ boundaries. Repeat per centroid.

## Centroid analysis

Topic centroids are useful on their own: the datapoints nearest a centroid summarize the core of each topic. You can pass those texts to an LLM to name or summarize clusters (`centroid_neighbors` method).

## Dependencies

Runtime dependencies are listed in **`pyproject.toml`** (`[project] dependencies`). The root **`requirements.txt`** mirrors those pins for workflows that prefer a flat file; prefer `pip install -e ".[dev]"` when developing.
