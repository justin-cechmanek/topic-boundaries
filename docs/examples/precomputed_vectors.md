# Precomputed Vectors

Use `PrecomputedEmbedder` to analyze vectors from another model or domain. Start the local Redis Stack with `docker compose up -d`, then run this from the repository root:

```python
import numpy as np

from topic_boundaries import Datapoint, PrecomputedEmbedder, max_distance_rankings, run_pipeline

datapoints = [
    Datapoint("east-1", vector=np.array([1.0, 0.0], dtype=np.float32)),
    Datapoint("east-2", vector=np.array([0.9, 0.1], dtype=np.float32)),
    Datapoint("north-1", vector=np.array([0.0, 1.0], dtype=np.float32)),
    Datapoint("north-2", vector=np.array([0.1, 0.9], dtype=np.float32)),
]

state = run_pipeline(
    datapoints,
    redis_url="redis://localhost:6379/0",
    n_clusters=2,
    embedder=PrecomputedEmbedder(),
    schema_path=None,
    overwrite_index=True,
)
hits = max_distance_rankings(
    state.indexed.index,
    state.centroids,
    state.indexed.n_clusters,
    top_n_per_cluster=1,
)
for hit in hits:
    print(hit.cluster_id, hit.doc_id, hit.distance)
```

`PrecomputedEmbedder` checks that every datapoint has a vector and that the vectors form a two-dimensional matrix. The index schema's vector dimension is updated from this data. The default index uses cosine distance; pass a custom `schema_path` if the vectors call for L2 or another supported metric.
