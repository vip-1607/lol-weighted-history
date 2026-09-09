"""Long-format series data validation and leakage-safe feature building."""

import csv
from dataclasses import dataclass
from pathlib import Path

from .core import SeriesState, probability
from .model import side_code, signed_streak, weighted_history_score


@dataclass(frozen=True)
class Observation:
    series_id: str
    best_of: int
    set_number: int
    a_won: int
    p0: float
    a_side: str


@dataclass(frozen=True)
class FeatureRow:
    series_id: str
    set_number: int
    outcome: int
    p0: float
    history: str
    history_score: float
    streak: int
    side: float


def _bool01(value):
    if value in (1, True, "1", "true", "True", "W", "w"):
        return 1
    if value in (0, False, "0", "false", "False", "L", "l"):
        return 0
    raise ValueError("a_won must be 0/1, true/false or W/L")


def normalize_rows(rows):
    normalized = []
    for row in rows:
        if isinstance(row, Observation):
            obs = row
        else:
            obs = Observation(str(row["series_id"]), int(row["best_of"]), int(row["set_number"]),
                              _bool01(row["a_won"]), float(row["p0"]), str(row["a_side"]).lower())
        if not obs.series_id:
            raise ValueError("series_id must not be empty")
        SeriesState(obs.best_of)
        if obs.set_number < 1:
            raise ValueError("set_number must start at 1")
        probability(obs.p0, "p0", open_interval=True)
        side_code(obs.a_side)
        if obs.a_won not in (0, 1):
            raise ValueError("a_won must be 0 or 1")
        normalized.append(obs)
    if not normalized:
        raise ValueError("at least one observation is required")
    return tuple(normalized)


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"series_id", "best_of", "set_number", "a_won", "p0", "a_side"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        return normalize_rows(reader)


def build_history_features(rows, decay):
    observations = normalize_rows(rows)
    groups = {}
    for obs in observations:
        groups.setdefault(obs.series_id, []).append(obs)
    features = []
    for series_id, group in groups.items():
        ordered = sorted(group, key=lambda row: row.set_number)
        if [row.set_number for row in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError(f"{series_id}: set_number must be unique and contiguous from 1")
        if len({row.best_of for row in ordered}) != 1 or len({row.p0 for row in ordered}) != 1:
            raise ValueError(f"{series_id}: best_of and p0 must be constant")
        history = ""
        for obs in ordered:
            state = SeriesState(obs.best_of, history.count("W"), history.count("L"))
            if state.ended:
                raise ValueError(f"{series_id}: result recorded after series ended")
            features.append(FeatureRow(series_id, obs.set_number, obs.a_won, obs.p0, history,
                                       weighted_history_score(history, decay), signed_streak(history),
                                       side_code(obs.a_side)))
            history += "W" if obs.a_won else "L"
    return tuple(features)
