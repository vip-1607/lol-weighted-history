"""Weighted-history probability model and series simulator."""

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import exp, log

from .core import Market, aggregate, probability, state_from_history, terminal_scores


@dataclass(frozen=True)
class HistoryCoefficients:
    alpha: float = 0.0
    decay: float = 0.7
    gamma: float = 0.0
    beta: float = 0.0

    def __post_init__(self):
        for name in ("alpha", "gamma", "beta"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not float("-inf") < float(value) < float("inf"):
                raise ValueError(f"{name} must be finite")
        probability(self.decay, "decay")


def weighted_history_score(history, decay):
    probability(decay, "decay")
    if not isinstance(history, str) or any(x not in "WL" for x in history):
        raise ValueError("history must contain only W/L")
    score = 0.0
    for result in history:
        score = decay * score + (1.0 if result == "W" else -1.0)
    return score


def signed_streak(history):
    if not isinstance(history, str) or any(x not in "WL" for x in history):
        raise ValueError("history must contain only W/L")
    if not history:
        return 0
    last = history[-1]
    run = 0
    for result in reversed(history):
        if result != last:
            break
        run += 1
    return run if last == "W" else -run


def side_code(side):
    normalized = str(side).strip().lower()
    values = {"blue": .5, "red": -.5, "neutral": 0.0}
    if normalized not in values:
        raise ValueError("side must be blue, red or neutral")
    return values[normalized]


def _sigmoid(z):
    return 1 / (1 + exp(-z)) if z >= 0 else exp(z) / (1 + exp(z))


def weighted_probability(p0, history="", side="neutral", coefficients=HistoryCoefficients()):
    p0 = probability(p0, "p0")
    if p0 in (0.0, 1.0):
        # Finite log-odds adjustments cannot move an infinite endpoint logit.
        return p0
    z = (log(p0 / (1 - p0))
         + coefficients.alpha * weighted_history_score(history, coefficients.decay)
         + coefficients.gamma * signed_streak(history)
         + coefficients.beta * side_code(side))
    return _sigmoid(z)


def _side_at(schedule, map_index):
    return "neutral" if schedule is None else schedule[map_index]


def simulate_series(best_of, p0, coefficients=HistoryCoefficients(), history="", future_sides=None):
    state = state_from_history(best_of, history)
    if future_sides is not None:
        if len(future_sides) != best_of:
            raise ValueError("future_sides must describe every possible map in the series")
        for side in future_sides:
            side_code(side)
    result = {score: 0.0 for score in terminal_scores(state)}

    def visit(h, a, b, mass):
        if a == state.target or b == state.target:
            result[a, b] += mass
            return
        p = weighted_probability(p0, h, _side_at(future_sides, len(h)), coefficients)
        visit(h + "W", a + 1, b, mass * p)
        visit(h + "L", a, b + 1, mass * (1 - p))

    visit(history, state.a, state.b, 1.0)
    return result


def price_markets(distribution, markets):
    return {name: aggregate(distribution, market) for name, market in markets.items()}


def sensitivity_grid(best_of, p0, coefficients, history, future_sides, grid, markets):
    unknown = set(grid) - {"alpha", "decay", "gamma", "beta"}
    if unknown:
        raise ValueError(f"Unknown sensitivity parameters: {sorted(unknown)}")
    names = tuple(grid)
    rows = []
    for values in product(*(grid[name] for name in names)):
        varied = replace(coefficients, **dict(zip(names, values)))
        distribution = simulate_series(best_of, p0, varied, history, future_sides)
        rows.append({"parameters": asdict(varied), "prices": price_markets(distribution, markets)})
    return rows
