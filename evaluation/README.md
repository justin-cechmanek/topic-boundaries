# Boundary-method evaluation

The completed human review is a five-paper pilot on vector search and visual
retrieval. All five candidates had enough context for the reviewer to assess
them. Mean boundary relevance and cross-topic value were both 1.8 out of 2,
and no candidate was marked as noise.

The proposed 40-paper review was abandoned because the supplied cluster
context was inadequate. It was not completed and will not be carried out.
There are no remaining ratings to collect for that protocol.

## Completed five-paper pilot

The reviewer was familiar with vector search and computer vision. We collected
44 unique arXiv papers on vector indexing, visual retrieval, and image-text
retrieval, then used two clusters with 15 and 29 papers. Three clusters were
explored first. The two-cluster configuration matched the requested subjects
and was chosen before any human scores were collected.

Each topic had a name, a description, and three representative-paper summaries
with original-source links. The representatives excluded all five candidates.
There was one unique candidate per method or random baseline. The order was
shuffled, and the method key stayed hidden until all numerical ratings were
supplied. The exact ratings, paper titles, source links, and setup are recorded
in [pilot/results.json](pilot/results.json).

| Candidate | Selection method | Boundary relevance (0-2) | Cross-topic value (0-2) | Noise (0-1) |
| --- | --- | ---: | ---: | ---: |
| P1 | Random baseline | 2 | 2 | 0 |
| P2 | Cross-boundary | 2 | 2 | 0 |
| P3 | Convex hull | 2 | 2 | 0 |
| P4 | Voronoi | 2 | 1 | 0 |
| P5 | Max distance | 1 | 2 | 0 |

Boundary relevance measures a meaningful extension beyond the source topic's
core. A central or unrelated paper scores 0, a plausible peripheral extension
scores 1, and a clear relevant extension scores 2. Cross-topic value measures
a useful connection to the comparison topic. No apparent connection scores 0,
a speculative connection scores 1, and a concrete credible transfer or research
question scores 2. Noise is 1 for an unusable item and 0 for a usable item.
Uncertainty is not a zero score or an automatic noise flag.

This pilot shows that the revised context was assessable for this reviewer.
The random candidate also received the highest scores. With one observation
per method, these results cannot establish a method ranking, an advantage over
random selection, statistical significance, or research novelty. The corpus
and source-topic choices were tailored to reviewer familiarity. Paper summaries
paraphrased abstracts, and full-paper claims were not independently verified.

## Reproduce the geometry check

Run from the private repository root after installing `pip install -e ".[dev]"`
and starting Redis Stack:

```bash
python evaluation/evaluate.py geometry
```

The command creates a uniquely named temporary index and deletes it when
finished. It writes `round1/geometry.json` with fixed 2D points, the four method
outputs, and candidate-set overlap. Ranked methods return their top three
per cluster, while the unranked hull contributes all of its vertices. The
`candidate_set_jaccard` values measure intersection size divided by union size.
The sets can have different sizes. Hull membership is independent of document
name order.

In the committed geometric result, max-distance ranks `A-outlier` first and
the hull includes it. The outlier also shifts the K-means centroid enough that
`A-bridge` is assigned to the other cluster. Voronoi then ranks that point first
within its assigned cluster. This is a concrete sensitivity of this small
configuration, not evidence that any method finds novel research.

## Historical sampling tools

The `sample` and `report` commands and `round1/review_sheet.csv`,
`round1/review_key.csv`, and `round1/metadata.json` belong to the abandoned
40-paper protocol over the original 100-paper snapshot. They remain as a record
of that protocol. The sheet is unrated. Those commands cannot regenerate or
report the completed five-paper pilot, and running them is not a remaining
review task.

Keep ratings and selection keys in the private repository. Tasks and delivery
status live in [GitHub Issues](https://github.com/justin-cechmanek/topic-boundaries/issues)
and the private [planning Project](https://github.com/users/justin-cechmanek/projects/1).
