from evaluation.evaluate import Candidate, METHODS, select_candidates


def test_blinded_sample_has_unique_ids_and_equal_method_counts():
    pools = {
        method: [Candidate(method, f"{method}-{i}", i % 5) for i in range(12)]
        for method in METHODS
    }
    selected = select_candidates(pools, n_clusters=5)
    assert len(selected) == 40
    assert len({item.doc_id for item in selected}) == 40
    assert {method: sum(item.method == method for item in selected) for method in METHODS} == {
        method: 8 for method in METHODS
    }
