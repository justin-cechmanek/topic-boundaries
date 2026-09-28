import topic_boundaries as tb


def test_public_api_names_resolve():
    # Every advertised name must import from the top-level package.
    for name in tb.__all__:
        assert hasattr(tb, name), f"missing public export: {name}"


def test_key_symbols_are_the_right_kind():
    assert callable(tb.embed_and_cluster_datapoints)
    assert callable(tb.index_clustered_datapoints)
    assert not hasattr(tb, "run_pipeline")
    assert callable(tb.CsvSource)
    # aliased rankers from two different modules
    assert tb.max_distance_rankings is not tb.voronoi_rankings
