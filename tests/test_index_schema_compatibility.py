import numpy as np
import pytest
from redisvl.index import SearchIndex
from redisvl.query import CountQuery
from redisvl.schema import IndexSchema

from topic_boundaries.indexing import create_and_load, records_for_redis


@pytest.mark.integration
@pytest.mark.parametrize("change", ["dims", "datatype", "distance_metric", "prefix", "storage_type"])
def test_incompatible_index_load_preserves_existing_document(make_index, redis_url, change):
    index = make_index(["a"], ["original"], [[1.0, 0.0]], [0], dim=2)
    before = index.client.hgetall(index.key("a"))
    raw = index.schema.to_dict()
    field = next(field for field in raw["fields"] if field["name"] == "embedding")
    if change in ("prefix", "storage_type"):
        raw["index"][change] = "different" if change == "prefix" else "json"
    else:
        field["attrs"][change] = {"dims": 3, "datatype": "float64", "distance_metric": "l2"}[change]
    candidate = SearchIndex(IndexSchema.from_dict(raw), redis_url=redis_url)
    records = records_for_redis(["a"], ["replacement"], np.array([[0.0, 1.0]]), np.array([1]))
    with pytest.raises(ValueError, match="Existing index schema"):
        create_and_load(candidate, records, overwrite=False, drop_keys=False)
    assert index.client.hgetall(index.key("a")) == before
    assert index.query(CountQuery()) == 1


@pytest.mark.integration
def test_compatible_index_load_keeps_existing_documents(make_index):
    index = make_index(["a"], ["original"], [[1.0, 0.0]], [0], dim=2)
    records = records_for_redis(["b"], ["new"], np.array([[0.0, 1.0]]), np.array([1]))
    create_and_load(index, records, overwrite=False, drop_keys=False)
    assert index.query(CountQuery()) == 2
