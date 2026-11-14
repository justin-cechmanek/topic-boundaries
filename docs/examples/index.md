# Examples

Run these from the repository root. The sample JSONL contains 20 illustrative
records; it is a walkthrough, not evaluation data. Start Redis Stack with
`docker compose up -d` for examples that index or query vectors.

- [Text corpus](text_corpus.md) loads, clusters, and ranks the sample records.
- [Precomputed vectors with Euclidean distance](precomputed_euclidean.md) shows
  the non-text path and a custom L2 schema.
- [Compare boundary methods](compare_methods.md) runs all four methods on the
  same clusters.
- [Visualize boundary candidates](visualize.md) saves a static and interactive
  plot using the library's visualization helpers.

```{toctree}
:maxdepth: 1

text_corpus
precomputed_euclidean
compare_methods
visualize
```
