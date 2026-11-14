from __future__ import annotations

from pathlib import Path

import yaml
from redisvl.schema import IndexSchema


def load_schema_for_dims(schema_path: Path, vector_dims: int) -> IndexSchema:
    """Load a FLOAT32 schema and override vector dimensions to match the embedder."""
    with schema_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for field in raw["fields"]:
        if field.get("name") == "embedding" and field.get("type") == "vector":
            field.setdefault("attrs", {})["dims"] = vector_dims
            break
    else:
        raise ValueError("schema.yml must define an 'embedding' vector field")
    schema = IndexSchema.from_dict(raw)
    if schema.fields["embedding"].attrs.datatype.value != "FLOAT32":
        raise ValueError("The 'embedding' vector field must use FLOAT32 storage.")
    return schema
