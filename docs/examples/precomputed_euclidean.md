# Precomputed vectors with Euclidean distance

Vectors can represent text, images, or any other numeric features. The
`PrecomputedEmbedder` passes them through unchanged; K-means uses Euclidean
geometry. This example also gives Redis an L2 schema, so the returned
max-distance scores are Euclidean distances. It creates a uniquely named
temporary index and deletes it when finished.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import yaml

from topic_boundaries import (
    Datapoint,
    PrecomputedEmbedder,
    Settings,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
    max_distance_rankings,
)

points = [(0, 0), (0.4, 0), (0, 0.3), (3, 0), (3.2, 0.3), (2.8, -0.2)]
datapoints = [
    Datapoint(doc_id=f"point-{i}", vector=np.array(point, dtype=np.float32))
    for i, point in enumerate(points)
]
vectors, labels, centroids = embed_and_cluster_datapoints(
    datapoints, n_clusters=2, embedder=PrecomputedEmbedder()
)

settings = Settings.from_config()
schema = yaml.safe_load(settings.schema_path.read_text(encoding="utf-8"))
suffix = uuid4().hex[:12]
schema["index"]["name"] = f"topic_boundaries_example_{suffix}"
schema["index"]["prefix"] = f"tb_example_{suffix}"
next(field for field in schema["fields"] if field["name"] == "embedding")["attrs"]["distance_metric"] = "l2"

with TemporaryDirectory() as directory:
    schema_path = Path(directory) / "l2.yml"
    schema_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    index = index_clustered_datapoints(
        datapoints,
        vectors,
        labels,
        redis_url=settings.redis_url,
        schema_path=str(schema_path),
        overwrite_index=True,
    )
    try:
        hits = max_distance_rankings(index, centroids, n_clusters=2, top_n_per_cluster=2)
        for hit in hits:
            print(hit.cluster_id, hit.doc_id, hit.distance)
    finally:
        index.delete(drop=True)
```

Use a schema with a nonzero-vector-compatible metric such as L2 when your
features include zero vectors. For cosine distance, vectors should have
meaningful nonzero directions; text embeddings from `HFTextEmbedder` are
normalized automatically.
