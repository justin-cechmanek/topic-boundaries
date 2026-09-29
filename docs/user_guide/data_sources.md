# Data Sources

A source returns `list[Datapoint]`. Pass that list to `embed_and_cluster_datapoints`, then pass it with the returned vectors and labels to `index_clustered_datapoints`. You can use a built-in loader or create `Datapoint` objects yourself.

| Loader | Input |
| --- | --- |
| `JsonlSource(path)` | One JSON object per line |
| `JsonSource(path, id_col=..., ...)` | A JSON array of objects |
| `CsvSource(path, id_col=..., ...)` | CSV rows |
| `RecordsSource(records, id_col=..., ...)` | An in-memory iterable of dictionaries |
| `TextFileSource(path)` / `TextDirSource(path)` | Text and Markdown, split into chunks |
| `PdfSource(path)` | One PDF, a directory, or a list of PDFs; requires the `corpus` extra |
| `ArxivSource(query)` | arXiv search results |
| `ArxivPaperSource(id_or_url, fetch_pdf=True)` | One paper's abstract or PDF chunks |

## Structured records

For CSV, JSON arrays, and in-memory records, use `id_col` for document identity and either `body_col` for text or `vector_col` for existing vectors. `meta_cols` copies selected fields into each datapoint's metadata.

```python
from topic_boundaries import CsvSource

datapoints = CsvSource(
    "papers.csv",
    id_col="paper_id",
    body_col="abstract",
    meta_cols=["title", "year"],
).load()
```

`JsonlSource` also recognizes records containing `abstract` and `arxiv_link`, as well as generic `doc_id` and `body` records. A record with a `vector` or `embedding` field can omit text.

## Precomputed vectors

Use `PrecomputedEmbedder` as the `embedder` argument to `embed_and_cluster_datapoints` when every datapoint has a vector of the same length. The vectors may come from another embedding model or a non-text domain.

## Text and PDF chunks

Text, Markdown, and PDF loaders split long content into datapoints. `max_chars` and `overlap` control chunk size and context overlap. Each chunk has its source path and chunk index in `meta`. Markdown headings and fence markers are removed before chunking; code inside the fences is retained.
