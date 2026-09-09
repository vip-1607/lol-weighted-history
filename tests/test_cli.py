import json

from weighted_history.cli import main


def test_simulate_and_fit_cli(tmp_path, capsys):
    main(["simulate", "examples/scenario.json"])
    simulation = json.loads(capsys.readouterr().out)
    assert sum(simulation["distribution"].values()) == pytest.approx(1)
    output = tmp_path / "model.json"
    main(["fit", "examples/history.csv", "--fixed-decay", ".7", "--output", str(output)])
    fitted = json.loads(capsys.readouterr().out)
    assert output.exists() and fitted["parameters"]["decay"] == .7


import pytest
