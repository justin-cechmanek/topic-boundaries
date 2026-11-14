import numpy as np
import pytest
import yaml

from topic_boundaries import Datapoint, index_clustered_datapoints


@pytest.mark.parametrize("body", [None, 42, ["text"]])
def test_invalid_body_is_rejected_before_connecting(body):
    with pytest.raises(ValueError, match="string document bodies"):
        index_clustered_datapoints(
            [Datapoint("a", body=body)], np.array([[1.0, 0.0]]), np.array([0]),
            redis_url="redis://127.0.0.1:1/0", overwrite_index=True,
        )


@pytest.mark.integration
def test_invalid_replacement_body_preserves_existing_documents(make_index, redis_url, tmp_path):
    index = make_index(["a"], ["original"], [[1.0, 0.0]], [0], dim=2)
    before = index.client.hgetall(index.key("a"))
    schema_path = tmp_path / "existing.yml"
    schema_path.write_text(yaml.safe_dump(index.schema.to_dict()))
    with pytest.raises(ValueError, match="string document bodies"):
        index_clustered_datapoints(
            [Datapoint("a", body=None)], np.array([[0.0, 1.0]]), np.array([1]),
            redis_url=redis_url, overwrite_index=True,
            schema_path=str(schema_path),
        )
    assert index.exists()
    assert index.client.hgetall(index.key("a")) == before
