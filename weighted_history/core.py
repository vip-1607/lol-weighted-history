"""Shared series state and settlement primitives for this standalone project."""

from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral


def probability(value, name="probability", *, open_interval=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    valid = 0 < value < 1 if open_interval else 0 <= value <= 1
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return value


@dataclass(frozen=True)
class SeriesState:
    best_of: int = 3
    a: int = 0
    b: int = 0

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (self.best_of, self.a, self.b)):
            raise ValueError("best_of and scores must be integers")
        if self.best_of not in (3, 5):
            raise ValueError("Only Bo3 and Bo5 are supported")
        if not (0 <= self.a <= self.target and 0 <= self.b <= self.target) or self.a == self.b == self.target:
            raise ValueError("Invalid series score")

    @property
    def target(self):
        return self.best_of // 2 + 1

    @property
    def ended(self):
        return self.a == self.target or self.b == self.target


@dataclass(frozen=True)
class Market:
    kind: str
    line: float | None = None

    def __post_init__(self):
        if self.kind not in {"match", "total_exact", "total_over", "handicap"}:
            raise ValueError("Unknown market kind")
        if self.kind == "match" and self.line is not None:
            raise ValueError("match does not take a line")
        if self.kind == "total_exact" and (not isinstance(self.line, Integral) or self.line < 0):
            raise ValueError("total_exact requires a nonnegative integer")
        if self.kind in {"total_over", "handicap"}:
            if not isinstance(self.line, (int, float)) or not isfinite(self.line):
                raise ValueError("line must be finite")
            if float(self.line).is_integer() or not (2 * float(self.line)).is_integer():
                raise ValueError("Use half-point lines")

    @property
    def label(self):
        return self.kind if self.line is None else f"{self.kind}:{self.line}"

    def payout(self, score):
        a, b = score
        if self.kind == "match":
            return int(a > b)
        if self.kind == "total_exact":
            return int(a + b == self.line)
        if self.kind == "total_over":
            return int(a + b > self.line)
        return int(a + self.line > b)


def terminal_scores(state):
    if state.ended:
        return ((state.a, state.b),)
    m = state.target
    return tuple([(m, b) for b in range(state.b, m)] + [(a, m) for a in range(state.a, m)])


def state_from_history(best_of, history):
    if not isinstance(history, str):
        raise ValueError("history must be a W/L string")
    state = SeriesState(best_of)
    for result in history:
        if state.ended or result not in "WL":
            raise ValueError("Invalid history or result after series end")
        state = SeriesState(best_of, state.a + (result == "W"), state.b + (result == "L"))
    return state


def aggregate(distribution, market):
    value = fsum(mass * market.payout(score) for score, mass in distribution.items())
    if not -1e-12 <= value <= 1 + 1e-12:
        raise ValueError("Aggregated probability outside [0,1]")
    return min(1.0, max(0.0, value))
