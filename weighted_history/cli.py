"""CLI for weighted-history simulation, estimation and prediction."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .core import Market
from .data import load_csv
from .fit import bootstrap_fit, cross_validate, fit_weighted_history
from .model import (HistoryCoefficients, price_markets, sensitivity_grid, simulate_series,
                    weighted_probability)


def _market(data):
    return Market(data["kind"], data.get("line"))


def _scenario(data, parameters=None):
    coefficients = parameters or HistoryCoefficients(**data.get("parameters", {}))
    markets = {row.get("name", _market(row).label): _market(row) for row in data.get("markets", [])}
    distribution = simulate_series(data["best_of"], data["p0"], coefficients,
                                   data.get("history", ""), data.get("future_sides"))
    next_side = "neutral"
    if data.get("future_sides"):
        next_side = data["future_sides"][len(data.get("history", ""))]
    output = {"parameters": asdict(coefficients),
              "next_probability": None if len(distribution) == 1 else weighted_probability(
                  data["p0"], data.get("history", ""), next_side, coefficients),
              "distribution": {f"{a}:{b}": value for (a, b), value in distribution.items()},
              "prices": price_markets(distribution, markets)}
    if data.get("sensitivity"):
        output["sensitivity"] = sensitivity_grid(data["best_of"], data["p0"], coefficients,
                                                  data.get("history", ""), data.get("future_sides"),
                                                  data["sensitivity"], markets)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(prog="weighted-history")
    sub = parser.add_subparsers(dest="command", required=True)
    simulate = sub.add_parser("simulate"); simulate.add_argument("scenario", type=Path)
    fit = sub.add_parser("fit"); fit.add_argument("csv", type=Path); fit.add_argument("--l2", type=float, default=1e-3)
    fit.add_argument("--fixed-decay", type=float); fit.add_argument("--cv-folds", type=int, default=0)
    fit.add_argument("--bootstrap", type=int, default=0); fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--output", type=Path)
    predict = sub.add_parser("predict"); predict.add_argument("model", type=Path); predict.add_argument("scenario", type=Path)
    args = parser.parse_args(argv)
    if args.command == "simulate":
        output = _scenario(json.loads(args.scenario.read_text(encoding="utf-8")))
    elif args.command == "predict":
        model = json.loads(args.model.read_text(encoding="utf-8"))
        coefficients = HistoryCoefficients(**model["parameters"])
        output = _scenario(json.loads(args.scenario.read_text(encoding="utf-8")), coefficients)
    else:
        rows = load_csv(args.csv); fixed = {} if args.fixed_decay is None else {"decay": args.fixed_decay}
        fitted = fit_weighted_history(rows, l2=args.l2, fixed=fixed)
        output = {"parameters": asdict(fitted.parameters), "fit": asdict(fitted)}
        if args.cv_folds:
            output["cross_validation"] = asdict(cross_validate(rows, args.cv_folds, args.seed, args.l2, fixed))
        if args.bootstrap:
            output["bootstrap"] = asdict(bootstrap_fit(rows, args.bootstrap, args.seed, args.l2, fixed))
        if args.output:
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
