# Boundary-method evaluation

The proposal asks whether geometric edges can surface overlooked ideas. These
checks separate geometric behavior from that human judgment. Nothing here
establishes novelty or statistical significance.

The initial 40-paper review was attempted but could not be completed with the
two-title cluster context supplied in the sheet. Preserve that sheet and leave
unjudgeable ratings blank. Follow-up review work is tracked in
[the human evaluation issue](https://github.com/justin-cechmanek/topic-boundaries/issues/33).
Tasks and delivery status live in GitHub Issues and the private
[planning Project](https://github.com/users/justin-cechmanek/projects/1).

Run from the private repository root after installing `pip install -e ".[dev]"`
and starting Redis Stack. The scripts create uniquely named temporary indexes
and delete them after each run:

```bash
python evaluation/evaluate.py geometry
python evaluation/evaluate.py sample
```

`geometry` writes `round1/geometry.json` with fixed 2D points, the four method
outputs, and candidate-set overlap. Ranked methods return their top three
per cluster, while the unranked hull contributes all of its vertices. The
`candidate_set_jaccard` values measure intersection size divided by union size.
The sets can have different sizes. Hull membership is independent of document
name order. The bridge, outlier, and elongated-cluster cases expose different
selections. `sample` embeds the committed 100-paper arXiv
snapshot, then writes 40 unique items to `round1/review_sheet.csv`, eight each
from max-distance, convex hull, Voronoi, cross-boundary, and a random baseline.
The order is shuffled. Methods and document IDs stay in `review_key.csv`.
Do not open the key until ratings are complete. Running `sample` again refuses
to overwrite the sheet. Choose a new `--output-dir` for another round.

In the committed geometric result, max-distance ranks `A-outlier` first and
the hull includes it. The outlier also shifts the K-means centroid enough that
`A-bridge` is assigned to the other cluster. Voronoi then ranks that point first
within its assigned cluster. This is a concrete sensitivity of this small
configuration, not evidence that any method finds novel research.

One reviewer fills three columns in `review_sheet.csv` for every item:

| Column | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `boundary_relevance_0_2` | Unrelated/noisy for the shown cluster context | Plausible peripheral item | Clear edge or bridge |
| `cross_topic_value_0_2` | No useful connection apparent | Possible follow-up | Compelling cross-topic question |

Set `noise_0_1` to `1` for a malformed, irrelevant, or otherwise unusable item,
and `0` otherwise. Add a short note when a rating needs explanation. The
source and other cluster cores provide context, not ground-truth labels.
The other cluster is a comparison topic, not evidence of a useful connection.
Confirm that the topic context is sufficient before scoring. Uncertainty is
not a zero rating or an automatic noise flag.

After all 40 items have ratings, run:

```bash
python evaluation/evaluate.py report
```

The command writes `round1/results.md` with counts, averages, and noise flags
by hidden selection method. Read the individual notes as well. One reviewer,
one small corpus, and eight items per method support descriptive comparison
only. Keep ratings and the key in the private repository.
