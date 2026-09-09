import pytest

from weighted_history.data import build_history_features, load_csv


def test_features_use_only_prior_results():
    rows = load_csv("examples/history.csv")
    features = build_history_features(rows, .7)
    s1 = [row for row in features if row.series_id == "s1"]
    assert s1[0].history == "" and s1[0].history_score == 0 and s1[0].streak == 0
    assert s1[1].history == "W" and s1[1].history_score == 1 and s1[1].streak == 1


@pytest.mark.parametrize("rows", [
    [{"series_id":"x","best_of":3,"set_number":2,"a_won":1,"p0":.5,"a_side":"blue"}],
    [{"series_id":"x","best_of":3,"set_number":1,"a_won":1,"p0":.5,"a_side":"green"}],
    [{"series_id":"x","best_of":3,"set_number":1,"a_won":1,"p0":.5,"a_side":"blue"},
     {"series_id":"x","best_of":3,"set_number":2,"a_won":1,"p0":.6,"a_side":"red"}],
    [{"series_id":"x","best_of":3,"set_number":1,"a_won":1,"p0":.5,"a_side":"blue"},
     {"series_id":"x","best_of":3,"set_number":2,"a_won":1,"p0":.5,"a_side":"red"},
     {"series_id":"x","best_of":3,"set_number":3,"a_won":0,"p0":.5,"a_side":"blue"}],
])
def test_invalid_rows(rows):
    with pytest.raises(ValueError):
        build_history_features(rows, .7)
