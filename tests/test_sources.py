import shutil
from pathlib import Path

import numpy as np
import pytest

from topic_boundaries import collect_datasets
import json

from topic_boundaries.sources import (
    ArxivPaperSource,
    CsvSource,
    JsonSource,
    PdfSource,
    RecordsSource,
    TextDirSource,
    TextFileSource,
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


def test_records_source_maps_dicts_with_native_values(tmp_path):
    records = [
        {"id": "a", "text": "hello world", "topic": "greetings", "emb": [1.0, 0.0]},
        {"id": "b", "text": "second row", "topic": "other", "emb": [0.0, 1.0]},
    ]
    dps = RecordsSource(
        records, id_col="id", body_col="text", meta_cols=["topic"], vector_col="emb"
    ).load()
    assert [d.doc_id for d in dps] == ["a", "b"]
    assert dps[0].body == "hello world"
    assert dps[0].meta["topic"] == "greetings"
    # native list (not a string) parses straight through
    np.testing.assert_array_equal(dps[0].vector, np.array([1.0, 0.0], dtype=np.float32))


def test_json_source_reads_array_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(
        json.dumps([{"id": "x", "text": "alpha"}, {"id": "y", "text": "beta"}]),
        encoding="utf-8",
    )
    dps = JsonSource(p, id_col="id", body_col="text").load()
    assert [d.doc_id for d in dps] == ["x", "y"]
    assert dps[1].body == "beta"


def test_json_source_rejects_non_array(tmp_path):
    p = tmp_path / "obj.json"
    p.write_text('{"id": "x"}', encoding="utf-8")
    with pytest.raises(ValueError):
        JsonSource(p, id_col="id").load()


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
    assert [d.body for d in dps] == ["alpha paragraph.", "beta paragraph.", "gamma content here."]
    assert all("#chunk" in d.doc_id for d in dps)
    assert all(d.meta["source_file"].endswith(".txt") for d in dps)
    assert any(d.doc_id.startswith("a.txt") for d in dps)


def test_textdir_source_covers_markdown(tmp_path):
    (tmp_path / "a.txt").write_text("plain text paragraph.", encoding="utf-8")
    (tmp_path / "b.md").write_text(
        "# Title\n\nBody paragraph.\n\n```python\ncode = 1\n```\n",
        encoding="utf-8",
    )
    dps = TextDirSource(tmp_path, max_chars=200, overlap=0).load()
    prefixes = {d.doc_id.split("#")[0] for d in dps}
    assert prefixes == {"a.txt", "b.md"}
    md_bodies = [d.body for d in dps if d.doc_id.startswith("b.md")]
    assert md_bodies == ["Title", "Body paragraph.", "code = 1"]


def test_textfile_source_single_txt_and_md(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("one paragraph.\n\ntwo paragraph.", encoding="utf-8")
    dps = TextFileSource(txt).load()
    assert [d.body for d in dps] == ["one paragraph.", "two paragraph."]
    assert dps[0].doc_id == "note#chunk0"
    assert dps[0].meta["source_file"].endswith("note.txt")

    md = tmp_path / "readme.md"
    md.write_text("## Heading\n\nsome content here.", encoding="utf-8")
    md_dps = TextFileSource(md).load()
    assert md_dps[0].doc_id == "readme#chunk0"
    assert "#" not in md_dps[0].body
    assert "Heading" in md_dps[0].body


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


def test_pdf_source_explicit_list_disambiguates_same_stem(tmp_path):
    src = _real_pdf()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    shutil.copy(src, first / "paper.pdf")
    shutil.copy(src, second / "paper.pdf")

    dps = PdfSource([first / "paper.pdf", second / "paper.pdf"], max_chars=500).load()
    assert {d.doc_id.split("#")[0] for d in dps} == {"paper", "paper-2"}
    assert len({d.doc_id for d in dps}) == len(dps)


def test_arxiv_paper_source_metadata(monkeypatch):
    # Substitute only the network hop; feed a real parsed record through.
    record = {
        "title": "A Study of Topic Boundaries",
        "abstract": "We investigate topic boundaries in embeddings.",
        "arxiv_link": "http://arxiv.org/abs/2403.15001v1",
    }
    monkeypatch.setattr(collect_datasets, "fetch_paper", lambda _: record)
    dps = ArxivPaperSource("https://arxiv.org/abs/2403.15001").load()
    assert len(dps) == 1
    assert dps[0].body == record["abstract"]


def test_arxiv_paper_source_fetch_pdf(monkeypatch):
    src = _real_pdf()
    # Substitute only the download with a copy of the real fixture PDF.
    monkeypatch.setattr(
        collect_datasets,
        "download_pdf",
        lambda _id, dest: (shutil.copy(src, dest), dest)[1],
    )
    dps = ArxivPaperSource(
        "https://arxiv.org/abs/2403.15001", fetch_pdf=True, max_chars=500
    ).load()
    assert len(dps) >= 5
    assert all(d.doc_id.startswith("2403.15001#chunk") for d in dps)
    assert dps[0].meta["chunk_index"] == 0
