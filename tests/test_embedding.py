import numpy as np
import pytest

from topic_boundaries.documents import Datapoint
from topic_boundaries.embedding import PrecomputedEmbedder


def _dp(doc_id, vec):
    return Datapoint(doc_id=doc_id, vector=np.asarray(vec, dtype=np.float32))


def test_precomputed_stacks_vectors():
    emb = PrecomputedEmbedder()
    out = emb.embed([_dp("a", [1.0, 0.0]), _dp("b", [0.0, 1.0])])
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    assert emb.dims == 2
    np.testing.assert_array_equal(out[0], [1.0, 0.0])


def test_precomputed_rejects_missing_vector():
    emb = PrecomputedEmbedder()
    with pytest.raises(ValueError, match="no precomputed vector"):
        emb.embed([_dp("a", [1.0, 0.0]), Datapoint(doc_id="b", body="text")])


def test_precomputed_rejects_ragged_dims():
    emb = PrecomputedEmbedder()
    with pytest.raises(ValueError, match="inconsistent shapes"):
        emb.embed([_dp("a", [1.0, 0.0]), _dp("b", [1.0, 0.0, 0.0])])


def test_precomputed_rejects_empty():
    with pytest.raises(ValueError):
        PrecomputedEmbedder().embed([])
