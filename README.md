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

5. **Run** (after install, use the `topic-boundaries` console script, or `python find_topics.py` from the repo root with `PYTHONPATH=.`):

   ```bash
   topic-boundaries --data datasets/sample.jsonl --n-clusters 5 --method max_distance_sort --overwrite-index
   ```

   To use a non-default config file, pass `--config /path/to/config.yml`.
   You can still override the configured Redis URL with `--redis-url`.

The installable Python package is named `topic-boundaries` on PyPI metadata but the import path is still **`src`** (e.g. `from src.pipeline import run_pipeline`). Run CLI from the repo root, or use the `topic-boundaries` entry point after `pip install -e .`.

## The process

Each technique lives under its own subdirectory inside **`src`**. They share these steps:

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

Use [convex hull algorithms](https://en.wikipedia.org/wiki/Convex_hull_algorithms), specifically [Quickhull](https://en.wikipedia.org/wiki/Quickhull), on a PCA-compressed representation of each cluster to approximate hull vertices (see `src/convex_hull/`).

### 3. Cross boundary

Using one cluster’s centroid, find the nearest datapoints that are **not** in that cluster—neighbors on other clusters’ boundaries. Repeat per centroid.

## Centroid analysis

Topic centroids are useful on their own: the datapoints nearest a centroid summarize the core of each topic. You can pass those texts to an LLM to name or summarize clusters (`centroid_neighbors` method).

## Dependencies

Runtime dependencies are listed in **`pyproject.toml`** (`[project] dependencies`). The root **`requirements.txt`** mirrors those pins for workflows that prefer a flat file; prefer `pip install -e ".[dev]"` when developing.
