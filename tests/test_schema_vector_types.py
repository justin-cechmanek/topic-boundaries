from pathlib import Path

import numpy as np
import pytest

from topic_boundaries import Datapoint, index_clustered_datapoints
from topic_boundaries.config import Settings
from topic_boundaries.schema_builder import load_schema_for_dims


@pytest.mark.parametrize("datatype", ["float64", "FLOAT64", "float16", "bfloat16"])
def test_unsupported_vector_storage_is_rejected_before_connecting(tmp_path, datatype):
    path = tmp_path / "schema.yml"
    path.write_text(
        Settings.from_config().schema_path.read_text().replace("datatype: float32", f"datatype: {datatype}")
    )
    with pytest.raises(ValueError, match="FLOAT32"):
        index_clustered_datapoints(
            [Datapoint("a")], np.array([[1.0, 0.0]]), np.array([0]),
            redis_url="redis://127.0.0.1:1/0", overwrite_index=True, schema_path=str(path),
        )


@pytest.mark.parametrize("datatype", ["float32", "FLOAT32"])
def test_float32_schemas_remain_supported(tmp_path: Path, datatype):
    path = tmp_path / "schema.yml"
    path.write_text(
        Settings.from_config().schema_path.read_text().replace("datatype: float32", f"datatype: {datatype}")
    )
    assert load_schema_for_dims(path, 2).fields["embedding"].attrs.datatype.value == "FLOAT32"
