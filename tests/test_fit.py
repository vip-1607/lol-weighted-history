import numpy as np
import pytest

from weighted_history import (HistoryCoefficients, bootstrap_fit, cross_validate,
                              fit_weighted_history, predict_from_fit, weighted_probability)


def synthetic_rows(n_series=600, seed=7):
    rng = np.random.default_rng(seed)
    truth = HistoryCoefficients(alpha=.45, decay=.6, gamma=.12, beta=.3)
    rows = []
    for i in range(n_series):
        p0 = float(rng.uniform(.32, .68)); history = ""; a = b = 0
        for set_number in range(1, 6):
            side = "blue" if rng.random() < .5 else "red"
            p = weighted_probability(p0, history, side, truth)
            won = int(rng.random() < p)
            rows.append({"series_id": f"s{i}", "best_of": 5, "set_number": set_number,
                         "a_won": won, "p0": p0, "a_side": side})
            history += "W" if won else "L"; a += won; b += 1 - won
            if a == 3 or b == 3:
                break
    return rows, truth


def test_synthetic_estimation_recovers_direction_and_scale():
    rows, truth = synthetic_rows()
    fit = fit_weighted_history(rows, l2=.01, fixed={"decay": truth.decay})
    assert fit.success and fit.status == "ok"
    assert fit.parameters.alpha > 0 and fit.parameters.gamma > 0 and fit.parameters.beta > 0
    assert fit.parameters.alpha == pytest.approx(truth.alpha, abs=.2)
    assert fit.parameters.decay == pytest.approx(truth.decay, abs=.35)
    assert fit.parameters.beta == pytest.approx(truth.beta, abs=.2)
    predictions = predict_from_fit(fit, rows[:20])
    assert len(predictions) == 20 and all(0 < p < 1 for p in predictions)
    free_decay = fit_weighted_history(rows[:500], l2=.01)
    assert 0 <= free_decay.parameters.decay <= 1


def test_cv_bootstrap_and_unidentifiable_data():
    rows, _ = synthetic_rows(80)
    cv = cross_validate(rows, folds=4, fixed={"decay": .6})
    assert len(cv.fold_log_losses) == 4 and cv.mean_log_loss > 0
    boot = bootstrap_fit(rows, samples=4, fixed={"decay": .6})
    assert boot.successful > 0 and "alpha" in boot.intervals
    one_map = [{"series_id":f"x{i}","best_of":3,"set_number":1,"a_won":i%2,
                "p0":.5,"a_side":"neutral"} for i in range(10)]
    fit = fit_weighted_history(one_map)
    assert not fit.success and fit.status == "unidentifiable"


def test_fixed_parameters_and_invalid_controls():
    rows, _ = synthetic_rows(20)
    fit = fit_weighted_history(rows, fixed={"decay": .7, "gamma": 0})
    assert fit.parameters.decay == .7 and fit.parameters.gamma == 0
    with pytest.raises(ValueError):
        fit_weighted_history(rows, l2=-1)
    with pytest.raises(ValueError):
        fit_weighted_history(rows, fixed={"unknown": 1})
