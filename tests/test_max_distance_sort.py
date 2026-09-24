from pathlib import Path

import numpy as np
import pytest

from topic_boundaries.max_distance_sort.boundaries import (
    boundary_rankings_for_all_clusters,
    furthest_from_centroid,
)


@pytest.mark.integration
def test_farthest_first_ranking(loaded_corpus):
    index, x = loaded_corpus
    hits = furthest_from_centroid(index, 0, np.array(x, dtype=np.float32))
    ids = [h.doc_id for h in hits]
    assert len(ids) == 3
    # c0_far is 90deg from the x-axis centroid -> farthest -> rank 0.
    assert ids[0] == "c0_far"
    assert ids[-1] == "c0_near"
    assert [h.rank_in_cluster for h in hits] == [0, 1, 2]


@pytest.mark.integration
def test_top_n_slices(loaded_corpus):
    index, x = loaded_corpus
    hits = furthest_from_centroid(index, 0, np.array(x, dtype=np.float32), top_n=1)
    assert [h.doc_id for h in hits] == ["c0_far"]


@pytest.mark.integration
def test_top_n_farthest_query_matches_exhaustive_scan(loaded_corpus):
    """The server-side 'farthest' query (top_n set, negated-centroid) must give
    the same order AND cosine distances as the exhaustive whole-cluster scan."""
    index, x = loaded_corpus
    centroid = np.array(x, dtype=np.float32)
    fast = furthest_from_centroid(index, 0, centroid, top_n=3)  # negated-centroid path
    exhaustive = furthest_from_centroid(index, 0, centroid)      # top_n=None path
    assert [h.doc_id for h in fast] == [h.doc_id for h in exhaustive]
    # c0_far is 90deg from the x-axis centroid -> cosine distance 1 - cos(90) = 1.0.
    assert fast[0].distance == pytest.approx(1.0, abs=1e-4)
    for a, b in zip(fast, exhaustive):
        assert a.distance == pytest.approx(b.distance, abs=1e-4)


@pytest.mark.integration
def test_l2_top_n_ranks_by_distance_from_centroid(make_index, tmp_path):
    schema = Path(__file__).resolve().parents[1] / "topic_boundaries" / "schema.yml"
    l2_schema = tmp_path / "l2.yml"
    l2_schema.write_text(
        schema.read_text().replace("distance_metric: cosine", "distance_metric: l2")
    )
    centroid = np.array([1.0, 0.0], dtype=np.float32)
    index = make_index(
        ["near", "far"], ["", ""], [[0.5, 0.0], [3.0, 0.0]], [0, 0],
        dim=2, schema_path=str(l2_schema),
    )

    hits = furthest_from_centroid(index, 0, centroid, top_n=1)
    assert [(hit.doc_id, hit.distance) for hit in hits] == [("far", pytest.approx(2.0))]
    all_hits = furthest_from_centroid(index, 0, centroid)
    assert [(hit.doc_id, hit.distance) for hit in all_hits] == [
        ("far", pytest.approx(2.0)), ("near", pytest.approx(0.5)),
    ]


@pytest.mark.integration
def test_empty_cluster_returns_empty(loaded_corpus):
    index, x = loaded_corpus
    assert furthest_from_centroid(index, 42, np.array(x, dtype=np.float32)) == []


@pytest.mark.integration
def test_rankings_across_all_clusters(loaded_corpus):
    index, x = loaded_corpus
    centroids = np.array([x, x], dtype=np.float32)  # both clusters ranked vs x-axis
    hits = boundary_rankings_for_all_clusters(index, centroids, 2)
    assert {h.cluster_id for h in hits} == {0, 1}
    assert len(hits) == 5  # 3 in cluster 0 + 2 in cluster 1
