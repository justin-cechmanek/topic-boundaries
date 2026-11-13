import numpy as np
import pytest
from redisvl.query import CountQuery

from topic_boundaries.indexing import create_and_load, open_index, records_for_redis


@pytest.mark.integration
def test_test_indexes_preserve_existing_schema_and_each_other(make_index, isolated_schema, redis_url):
    parent_schema = isolated_schema()
    parent = open_index(str(parent_schema), redis_url, 2)
    records = records_for_redis(["a"], ["keep"], np.array([[1.0, 0.0]]), np.array([0]))
    create_and_load(parent, records, overwrite=False, drop_keys=False)

    first = make_index(["a"], ["first"], [[1.0, 0.0]], [0], dim=2, schema_path=parent_schema)
    second = make_index(["a"], ["second"], [[0.0, 1.0]], [0], dim=2, schema_path=parent_schema)
    assert len({parent.name, first.name, second.name}) == 3
    assert len({parent.schema.index.prefix, first.schema.index.prefix, second.schema.index.prefix}) == 3
    assert parent.client.hget(parent.key("a"), "body") == b"keep"
    assert first.query(CountQuery()) == second.query(CountQuery()) == 1
