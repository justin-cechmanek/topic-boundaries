import numpy as np
import pytest

from topic_boundaries.indexing import records_for_redis
from topic_boundaries.max_distance_sort.boundaries import _cluster_doc_count


# --- pure logic: no Redis ---


def test_records_for_redis_serialization():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    recs = records_for_redis(["a", "b"], ["ba", "bb"], vectors, np.array([0, 3]))
    assert recs[0]["doc_id"] == "a"
    assert recs[0]["cluster_id"] == "0"
    assert recs[1]["cluster_id"] == "3"  # str(int(label))
    blob = recs[0]["embedding"]
    assert isinstance(blob, bytes)
    assert len(blob) == 2 * 4  # 2 float32
    np.testing.assert_array_equal(
        np.frombuffer(blob, dtype=np.float32), vectors[0]
    )


# --- integration: real Redis ---


@pytest.mark.integration
def test_cluster_doc_count(loaded_corpus):
    index, _ = loaded_corpus
    assert _cluster_doc_count(index, 0) == 3
    assert _cluster_doc_count(index, 1) == 2
    assert _cluster_doc_count(index, 99) == 0
