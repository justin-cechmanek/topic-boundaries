# Boundary-method evaluation

Use the geometry check to inspect how the four boundary methods respond to
controlled clusters containing a bridge, an outlier, and elongated shapes.
It helps check rankings and expose sensitivity to cluster assignments before
applying the methods to a text or vector corpus.

## Run the check

From the repository root, install the dependencies and start local Redis Stack:

```bash
pip install -e ".[dev]"
docker compose up -d
python evaluation/evaluate.py geometry
```

The check uses fixed 2D vectors and K-means with random state 42. It does not
need an embedding-model download. Max-distance and cross-boundary queries use
a uniquely named L2 index. The script deletes that index and its documents
when queries finish, including when a query raises an exception.

`Settings.from_config()` supplies the Redis connection and base schema. Set
`REDIS_URL` to use another Redis Stack instance. The temporary schema uses its
own index name and key prefix, leaving the configured index untouched.

The output is `build/evaluation/geometry.json`. The `build` directory is ignored
by Git. Running the command again replaces the file. To choose another output
directory, run:

```bash
python evaluation/evaluate.py geometry --output-dir build/geometry-check
```

## Read the output

| Field | Meaning |
| --- | --- |
| `points` | Document names and their fixed 2D coordinates |
| `cluster_labels` | Cluster assigned to each point by K-means |
| `rankings` | Per-cluster selections from each boundary method |
| `candidate_set_jaccard` | Pairwise intersection size divided by union size for the selected document sets |
| `overlap_selection` | Which results are included in the overlap calculation |
| `observations` | Outlier membership, bridge reassignment, and bridge Voronoi rank |

Max-distance, Voronoi, and cross-boundary return three candidates per cluster.
Cross-boundary candidates belong to other clusters. Convex hull returns all
vertices, which form an unranked set. Overlap uses all returned candidates,
so the compared sets may have different sizes.

The script asserts that the outlier is farthest from its cluster centroid and
is included in the hull. Inspect the bridge's assigned cluster and Voronoi
rank to understand how clustering affects the boundary selections. Different
methods answer different geometric questions, so their candidate sets can
differ without indicating an error.

These checks evaluate geometric behavior. They do not measure whether a
candidate is useful or novel. Assessing those qualities requires human review
with enough topic context to make a defensible judgment.
