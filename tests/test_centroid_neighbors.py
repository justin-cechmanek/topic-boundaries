import pytest

from src.centroid_neighbors import nearest_to_centroid


@pytest.mark.integration
def test_nearest_first_within_cluster(loaded_corpus):
    index, x = loaded_corpus
    rows = nearest_to_centroid(index, x, cluster_id=0, k=2)
    ids = [str(r["doc_id"]) for r in rows]
    assert len(ids) == 2
    # Nearest-first (opposite of max_distance_sort): closest to x-axis first.
    assert ids[0] == "c0_near"
    assert ids[1] == "c0_mid"
    assert all(str(r["cluster_id"]) == "0" for r in rows)


@pytest.mark.integration
def test_empty_cluster_returns_empty(loaded_corpus):
    index, x = loaded_corpus
    assert nearest_to_centroid(index, x, cluster_id=42) == []
