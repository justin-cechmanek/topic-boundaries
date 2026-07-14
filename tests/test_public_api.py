import topic_boundaries as tb


def test_public_api_names_resolve():
    # Every advertised name must import from the top-level package.
    for name in tb.__all__:
        assert hasattr(tb, name), f"missing public export: {name}"


def test_key_symbols_are_the_right_kind():
    assert callable(tb.run_pipeline)
    assert isinstance(tb.SOURCES, dict) and "csv" in tb.SOURCES
    # aliased rankers from two different modules
    assert tb.max_distance_rankings is not tb.voronoi_rankings
