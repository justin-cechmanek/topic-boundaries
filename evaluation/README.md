# Boundary-method evaluation

The proposal asks whether geometric edges can surface overlooked ideas. These
checks separate geometric behavior from that human judgment. Nothing here
establishes novelty or statistical significance.

Run from the private repository root after installing `pip install -e ".[dev]"`
and starting Redis Stack. The scripts create uniquely named temporary indexes
and delete them after each run:

```bash
python evaluation/evaluate.py geometry
python evaluation/evaluate.py sample
```

`geometry` writes `round1/geometry.json` with fixed 2D points, the four method
outputs, and top-three overlap. Its bridge, outlier, and elongated-cluster
cases expose different rankings. `sample` embeds the committed 100-paper arXiv
snapshot, then writes 40 unique items to `round1/review_sheet.csv`, eight each
from max-distance, convex hull, Voronoi, cross-boundary, and a random baseline.
The order is shuffled; methods and document IDs stay in `review_key.csv`.
Do not open the key until ratings are complete. Running `sample` again refuses
to overwrite the sheet; choose a new `--output-dir` for another round.

In the committed geometric result, max-distance ranks `A-outlier` first and
the hull includes it. The outlier also shifts the K-means centroid enough that
`A-bridge` is assigned to the other cluster; Voronoi then ranks that point first
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

After all 40 items have ratings, run:

```bash
python evaluation/evaluate.py report
```

The command writes `round1/results.md` with counts, averages, and noise flags
by hidden selection method. Read the individual notes as well; one reviewer,
one small corpus, and eight items per method support descriptive comparison
only. Keep ratings and the key in the private repository.
