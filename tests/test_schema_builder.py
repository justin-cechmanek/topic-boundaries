from pathlib import Path

import pytest

pytest.importorskip("redisvl")

from topic_boundaries.schema_builder import load_schema_for_dims

ROOT = Path(__file__).resolve().parents[1]


def test_overrides_embedding_dims():
    schema = load_schema_for_dims(ROOT / "topic_boundaries" / "schema.yml", 128)
    emb = schema.fields["embedding"]
    # redisvl stores vector attrs; dims must reflect the override, not schema.yml's 384.
    assert emb.attrs.dims == 128


def test_missing_embedding_field_raises(tmp_path: Path):
    bad = tmp_path / "schema.yml"
    bad.write_text(
        "index:\n"
        "  name: t\n"
        "  prefix: t\n"
        "fields:\n"
        "  - name: body\n"
        "    type: text\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_schema_for_dims(bad, 384)
