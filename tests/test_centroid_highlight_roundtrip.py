import json

from topic_boundaries import Datapoint
from topic_boundaries.visualization import boundary_highlight_indices, write_boundary_results


def test_centroid_neighbor_results_roundtrip_through_disk(tmp_path):
    datapoints = [Datapoint("a"), Datapoint("b"), Datapoint("c")]
    path = tmp_path / "neighbors.json"
    write_boundary_results(
        path, method="centroid_neighbors", n_clusters=2,
        doc_ids_by_cluster=[["a"], ["c"]], datapoints=datapoints,
    )
    assert boundary_highlight_indices(json.loads(path.read_text()), datapoints) == {0, 2}
