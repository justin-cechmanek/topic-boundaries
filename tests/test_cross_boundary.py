from pathlib import Path

import numpy as np
import pytest

from topic_boundaries.cross_boundary.boundaries import (
    cross_boundary_hits_for_all_clusters,
    nearest_outside_cluster,
)


@pytest.mark.integration
def test_excludes_source_cluster_and_orders_nearest_first(loaded_corpus):
    index, x = loaded_corpus
    hits = nearest_outside_cluster(index, source_cluster_id=0, centroid=x, k=5)
    ids = [h.neighbor_doc_id for h in hits]
    # Only cluster 1 docs may appear (Tag cluster_id != "0").
    assert set(ids) == {"c1_nearx", "c1_z"}
    assert all(h.neighbor_cluster_id != "0" for h in hits)
    # c1_nearx sits near the x-axis centroid -> nearest cross-cluster neighbor.
    assert ids[0] == "c1_nearx"
    assert hits[0].distance == pytest.approx(0.1, abs=0.01)


@pytest.mark.integration
def test_l2_scores_are_euclidean_distances(make_index, tmp_path):
    schema = Path(__file__).resolve().parents[1] / "topic_boundaries" / "schema.yml"
    l2_schema = tmp_path / "l2.yml"
    l2_schema.write_text(
        schema.read_text().replace("distance_metric: cosine", "distance_metric: l2")
    )
    index = make_index(
        ["own", "near", "far"],
        ["", "", ""],
        [[1.0, 0.0], [0.0, 0.0], [1.0, 2.0]],
        [0, 1, 1],
        dim=2,
        schema_path=str(l2_schema),
    )
    hits = nearest_outside_cluster(index, source_cluster_id=0, centroid=[1.0, 0.0], k=2)
    assert [(hit.neighbor_doc_id, hit.distance) for hit in hits] == [
        ("near", pytest.approx(1.0)),
        ("far", pytest.approx(2.0)),
    ]


@pytest.mark.integration
def test_aggregate_across_clusters(loaded_corpus):
    index, x = loaded_corpus
    centroids = np.array([x, x], dtype=np.float32)
    hits = cross_boundary_hits_for_all_clusters(index, centroids, 2, k_per_centroid=5)
    # cluster 0 sees 2 outside docs, cluster 1 sees 3 outside docs.
    assert len(hits) == 5
    assert {h.source_cluster_id for h in hits} == {0, 1}
