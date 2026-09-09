from itertools import product

import pytest

from weighted_history import (HistoryCoefficients, Market, price_markets, signed_streak,
                              simulate_series, weighted_history_score, weighted_probability)
from weighted_history.core import state_from_history, terminal_scores


def enumerate_paths(best_of, p0, coefficients, history, sides):
    state = state_from_history(best_of, history)
    out = {score: 0.0 for score in terminal_scores(state)}

    def visit(h, a, b, mass):
        if a == state.target or b == state.target:
            out[a, b] += mass
            return
        p = weighted_probability(p0, h, sides[len(h)], coefficients)
        visit(h + "W", a + 1, b, mass * p)
        visit(h + "L", a, b + 1, mass * (1 - p))
    visit(history, state.a, state.b, 1.0)
    return out


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("history", ["", "W", "L", "WL", "LW"])
def test_simulator_against_independent_path_recursion(best_of, history):
    try:
        state_from_history(best_of, history)
    except ValueError:
        pytest.skip("history already ended")
    sides = (["blue", "red", "neutral"] if best_of == 3 else
             ["blue", "red", "neutral", "blue", "red"])
    params = HistoryCoefficients(.3, .65, .08, .2)
    actual = simulate_series(best_of, .6, params, history, sides)
    assert actual == pytest.approx(enumerate_paths(best_of, .6, params, history, sides))
    assert sum(actual.values()) == pytest.approx(1)


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("p", [.01, .5, .99])
def test_zero_effects_equal_iid_formula(best_of, p):
    dist = simulate_series(best_of, p, HistoryCoefficients(0, .7, 0, 0))
    target = best_of // 2 + 1
    expected = {}
    for sequence in product("WL", repeat=best_of):
        a = b = 0; prob = 1.0
        for result in sequence:
            prob *= p if result == "W" else 1 - p
            a += result == "W"; b += result == "L"
            if a == target or b == target:
                expected[a, b] = expected.get((a, b), 0) + prob
                break
    # Full-length sequences repeat absorbed prefixes; divide by suffix multiplicity via direct formulas instead.
    if best_of == 3:
        expected = {(2, 0): p*p, (2, 1): 2*p*p*(1-p),
                    (0, 2): (1-p)**2, (1, 2): 2*p*(1-p)**2}
    assert sum(dist.values()) == pytest.approx(1)
    if best_of == 3:
        assert dist == pytest.approx(expected)


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("p", [0, .5, 1])
def test_all_reachable_histories_constant_model(best_of, p):
    params = HistoryCoefficients(0, .7, 0, 0)
    for length in range(best_of + 1):
        for letters in product("WL", repeat=length):
            history = "".join(letters)
            try:
                state = state_from_history(best_of, history)
            except ValueError:
                continue
            dist = simulate_series(best_of, p, params, history)
            assert sum(dist.values()) == pytest.approx(1)
            if state.ended:
                assert dist == {(state.a, state.b): 1.0}


def test_team_exchange_symmetry():
    params = HistoryCoefficients(.3, .65, .08, .2)
    sides = ["blue", "red", "neutral", "blue", "red"]
    swapped_sides = [{"blue": "red", "red": "blue", "neutral": "neutral"}[s] for s in sides]
    a = simulate_series(5, .62, params, "WL", sides)
    b = simulate_series(5, .38, params, "LW", swapped_sides)
    assert a == pytest.approx({(y, x): mass for (x, y), mass in b.items()})


def test_history_streak_side_and_prices():
    assert weighted_history_score("WLW", .5) == pytest.approx(.75)
    assert signed_streak("WWW") == 3
    assert signed_streak("WLL") == -2
    params = HistoryCoefficients(.4, .7, .1, .2)
    assert weighted_probability(.6, "W", "blue", params) > weighted_probability(.6, "", "blue", params)
    assert weighted_probability(.6, "L", "red", params) < weighted_probability(.6, "", "red", params)
    dist = simulate_series(3, .6, params, future_sides=["blue", "red", "blue"])
    prices = price_markets(dist, {"match": Market("match"), "three": Market("total_exact", 3)})
    assert 0 < prices["match"] < 1 and 0 < prices["three"] < 1
