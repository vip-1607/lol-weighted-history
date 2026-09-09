"""Regularized likelihood estimation with series-level validation utilities."""

from dataclasses import asdict, dataclass, replace
from math import log

import numpy as np
from scipy.optimize import minimize

from .data import Observation, build_history_features, normalize_rows
from .model import HistoryCoefficients, side_code, signed_streak, weighted_history_score


@dataclass(frozen=True)
class FitResult:
    parameters: HistoryCoefficients
    success: bool
    status: str
    message: str
    n_observations: int
    n_series: int
    design_rank: int
    log_loss: float
    objective: float
    l2: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CrossValidationResult:
    fold_log_losses: tuple[float, ...]
    mean_log_loss: float


@dataclass(frozen=True)
class BootstrapResult:
    requested: int
    successful: int
    intervals: dict[str, tuple[float, float]]


PARAMETERS = ("alpha", "decay", "gamma", "beta")
BOUNDS = {"alpha": (-10, 10), "decay": (0, 1), "gamma": (-10, 10), "beta": (-10, 10)}


def _linear_predictors(observations, params):
    features = build_history_features(observations, params.decay)
    return np.array([
        log(row.p0 / (1 - row.p0))
        + params.alpha * row.history_score
        + params.gamma * row.streak
        + params.beta * row.side
        for row in features
    ])


def _outcomes(observations, decay):
    return np.array([row.outcome for row in build_history_features(observations, decay)], dtype=float)


def _loss(observations, params, l2):
    z = _linear_predictors(observations, params)
    y = _outcomes(observations, params.decay)
    nll = np.sum(np.logaddexp(0.0, z) - y * z)
    penalty = .5 * l2 * (params.alpha ** 2 + params.gamma ** 2 + params.beta ** 2)
    return float(nll + penalty)


def fit_weighted_history(rows, l2=1e-3, fixed=None):
    observations = normalize_rows(rows)
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be nonnegative")
    fixed = dict(fixed or {})
    if set(fixed) - set(PARAMETERS):
        raise ValueError("fixed contains an unknown parameter")
    for key, value in fixed.items():
        if not np.isfinite(value) or not BOUNDS[key][0] <= value <= BOUNDS[key][1]:
            raise ValueError(f"fixed {key} is outside its allowed range")
    free = tuple(name for name in PARAMETERS if name not in fixed)

    def unpack(x):
        values = {"alpha": 0.0, "decay": .5, "gamma": 0.0, "beta": 0.0, **fixed}
        values.update(dict(zip(free, x)))
        return HistoryCoefficients(**values)

    if free:
        starts = []
        for decay_start in (.1, .5, .9):
            base = {"alpha": 0.0, "decay": decay_start, "gamma": 0.0, "beta": 0.0, **fixed}
            starts.append([base[name] for name in free])
        results = [minimize(lambda x: _loss(observations, unpack(x), l2), start,
                            method="L-BFGS-B", bounds=[BOUNDS[name] for name in free],
                            options={"maxiter": 3000, "ftol": 1e-12}) for start in starts]
        result = min(results, key=lambda item: item.fun)
        params = unpack(result.x)
        converged, message = bool(result.success), str(result.message)
    else:
        params, converged, message = unpack([]), True, "all parameters fixed"

    features = build_history_features(observations, params.decay)
    linear_free = [name for name in ("alpha", "gamma", "beta") if name in free]
    columns = {"alpha": [r.history_score for r in features],
               "gamma": [r.streak for r in features], "beta": [r.side for r in features]}
    matrix = np.column_stack([columns[name] for name in linear_free]) if linear_free else np.empty((len(features), 0))
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    identifiable = rank == len(linear_free)
    if "decay" in free and not any(row.set_number >= 3 for row in observations):
        identifiable = False
    z = _linear_predictors(observations, params)
    y = _outcomes(observations, params.decay)
    log_loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    warnings = []
    if "decay" in free and (params.decay < 1e-5 or params.decay > 1 - 1e-5):
        warnings.append("decay estimate is on a boundary; history and streak effects may be weakly identified")
    status = "ok" if converged and identifiable else "unidentifiable" if converged else "failed"
    return FitResult(params, converged and identifiable, status, message, len(observations),
                     len({r.series_id for r in observations}), rank, log_loss,
                     _loss(observations, params, l2), float(l2), tuple(warnings))


def predict_from_fit(result, rows):
    observations = normalize_rows(rows)
    z = _linear_predictors(observations, result.parameters)
    return tuple(float(1 / (1 + np.exp(-value))) for value in z)


def cross_validate(rows, folds=5, seed=0, l2=1e-3, fixed=None):
    observations = normalize_rows(rows)
    ids = np.array(sorted({row.series_id for row in observations}))
    if not 2 <= folds <= len(ids):
        raise ValueError("folds must be between 2 and the number of series")
    rng = np.random.default_rng(seed); rng.shuffle(ids)
    losses = []
    for validation_ids in np.array_split(ids, folds):
        held = set(validation_ids.tolist())
        train = [row for row in observations if row.series_id not in held]
        validation = [row for row in observations if row.series_id in held]
        fit = fit_weighted_history(train, l2=l2, fixed=fixed)
        if not fit.success:
            raise ValueError(f"cross-validation training fit failed: {fit.status}")
        z = _linear_predictors(validation, fit.parameters)
        y = _outcomes(validation, fit.parameters.decay)
        losses.append(float(np.mean(np.logaddexp(0.0, z) - y * z)))
    return CrossValidationResult(tuple(losses), float(np.mean(losses)))


def bootstrap_fit(rows, samples, seed=0, l2=1e-3, fixed=None):
    observations = normalize_rows(rows)
    if samples < 1:
        raise ValueError("samples must be positive")
    groups = {}
    for row in observations:
        groups.setdefault(row.series_id, []).append(row)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for sample_index in range(samples):
        sampled = []
        for copy_index, series_id in enumerate(rng.choice(ids, size=len(ids), replace=True)):
            sampled.extend(replace(row, series_id=f"boot{sample_index}-{copy_index}")
                           for row in groups[series_id])
        fitted = fit_weighted_history(sampled, l2=l2, fixed=fixed)
        if fitted.success:
            estimates.append(asdict(fitted.parameters))
    intervals = {}
    for name in PARAMETERS:
        values = [row[name] for row in estimates]
        if values:
            intervals[name] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
    return BootstrapResult(samples, len(estimates), intervals)
