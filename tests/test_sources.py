import shutil
from pathlib import Path

import numpy as np
import pytest

from topic_boundaries.sources import (
    CsvSource,
    PdfSource,
    TextDirSource,
    _parse_vector,
)


def _real_pdf() -> Path:
    pytest.importorskip("pypdf")
    pdf = Path(__file__).resolve().parents[1] / "knowledge_boundaries_paper_proposal.pdf"
    if not pdf.is_file():
        pytest.skip(f"missing {pdf}")
    return pdf


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


def test_textdir_source_chunks_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha paragraph.\n\nbeta paragraph.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("gamma content here.", encoding="utf-8")
    dps = TextDirSource(tmp_path, max_chars=20, overlap=0).load()
    assert len(dps) >= 3  # a.txt splits into >=2, b.txt >=1
    assert all("#chunk" in d.doc_id for d in dps)
    assert all(d.meta["source_file"].endswith(".txt") for d in dps)
    assert any(d.doc_id.startswith("a.txt") for d in dps)


def test_pdf_source_single_file_unchanged(tmp_path):
    src = _real_pdf()
    dps = PdfSource(src, max_chars=500).load()
    assert len(dps) >= 5
    assert dps[0].doc_id == f"{src.stem}#chunk0"
    assert dps[0].meta["source_pdf"].endswith(src.name)


def test_pdf_source_directory_globs_and_disambiguates(tmp_path):
    src = _real_pdf()
    shutil.copy(src, tmp_path / "one.pdf")
    sub = tmp_path / "nested"
    sub.mkdir()
    shutil.copy(src, sub / "one.pdf")  # same name, different dir

    dps = PdfSource(tmp_path, max_chars=500).load()
    prefixes = {d.doc_id.split("#")[0] for d in dps}
    # relative-path prefixes keep the two same-named files apart
    assert prefixes == {"one", "nested/one"}
    # each chunk carries its own source_pdf pointing at the file it came from
    assert {Path(d.meta["source_pdf"]).parent.name for d in dps} == {
        tmp_path.name,
        "nested",
    }


def test_pdf_source_explicit_list_concatenates(tmp_path):
    src = _real_pdf()
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    shutil.copy(src, a)
    shutil.copy(src, b)

    dps = PdfSource([a, b], max_chars=500).load()
    prefixes = {d.doc_id.split("#")[0] for d in dps}
    assert prefixes == {"a", "b"}
    only_a = PdfSource(a, max_chars=500).load()
    assert len(dps) == 2 * len(only_a)
