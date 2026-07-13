import numpy as np
import pytest

from topic_boundaries.sources import (
    CsvSource,
    DataFrameSource,
    TextDirSource,
    _parse_vector,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("[0.1, 0.2, 0.3]", [0.1, 0.2, 0.3]),
        ("0.1 0.2 0.3", [0.1, 0.2, 0.3]),
        ("0.1,0.2,0.3", [0.1, 0.2, 0.3]),
        ([1, 2, 3], [1.0, 2.0, 3.0]),
    ],
)
def test_parse_vector(value, expected):
    assert _parse_vector(value) == expected


def test_csv_source_maps_columns(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text(
        "id,text,topic\n"
        "a,hello world,greetings\n"
        "b,second row,other\n",
        encoding="utf-8",
    )
    dps = CsvSource(p, id_col="id", body_col="text", meta_cols=["topic"]).load()
    assert [d.doc_id for d in dps] == ["a", "b"]
    assert dps[0].body == "hello world"
    assert dps[0].meta["topic"] == "greetings"
    assert dps[0].vector is None


def test_csv_source_with_precomputed_vectors(tmp_path):
    p = tmp_path / "vec.csv"
    p.write_text(
        'id,emb\na,"[1.0, 0.0]"\nb,"[0.0, 1.0]"\n',
        encoding="utf-8",
    )
    dps = CsvSource(p, id_col="id", vector_col="emb").load()
    assert dps[0].body == ""  # vector-only
    np.testing.assert_array_equal(dps[0].vector, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(dps[1].vector, np.array([0.0, 1.0], dtype=np.float32))


def test_dataframe_source_duck_typed():
    # Minimal duck-typed stand-in for a DataFrame: .iterrows() yields (idx, row).
    class FakeDF:
        def __init__(self, rows):
            self._rows = rows

        def iterrows(self):
            return enumerate(self._rows)

    df = FakeDF([{"id": "x", "text": "foo"}, {"id": "y", "text": "bar"}])
    dps = DataFrameSource(df, id_col="id", body_col="text").load()
    assert [d.doc_id for d in dps] == ["x", "y"]
    assert dps[1].body == "bar"


def test_textdir_source_chunks_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha paragraph.\n\nbeta paragraph.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("gamma content here.", encoding="utf-8")
    dps = TextDirSource(tmp_path, max_chars=20, overlap=0).load()
    assert len(dps) >= 3  # a.txt splits into >=2, b.txt >=1
    assert all("#chunk" in d.doc_id for d in dps)
    assert all(d.meta["source_file"].endswith(".txt") for d in dps)
    assert any(d.doc_id.startswith("a.txt") for d in dps)
