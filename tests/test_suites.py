"""Exploratory studies run the actual configured interventions."""
import pytest

from sfl_lab.config import CandidateConfig, EvaluationConfig, ExperimentConfig
from sfl_lab.engine import run_training
from sfl_lab.suites import run_suites


def test_two_dimensional_sweep_and_active_ablations_execute_real_conditions():
    config = ExperimentConfig(model="tiny_cnn", samples=40, clients=2, rounds=1, seeds=[1],
        candidate=CandidateConfig(expression="gaussian(clip(G, C_G), sigma_G, C_G)", parameters={"C_G": 1., "sigma_G": 1.},
                                  activation_clip=1., activation_noise=2., fedprox_mu=.01),
        evaluation=EvaluationConfig(network=False, non_iid=False, attacks=False, ablation=True,
            sweep=True, sweep_values=[.5, 1], sweep_secondary_parameter="sigma_G", sweep_secondary_values=[0, 1]))
    baseline, candidate = run_training(config, "baseline", 1), run_training(config, "candidate", 1)
    suites = run_suites(config, [baseline], [candidate])
    sweep = suites["sweep"]
    assert sweep["status"] == "completed"
    assert len(sweep["rows"]) == 4
    assert {(r["value"], r["secondary_value"]) for r in sweep["rows"]} == {(.5, 0), (.5, 1), (1, 0), (1, 1)}
    assert all(r["source"] == "measured_training" and r["history"] for r in sweep["rows"])
    for row in sweep["rows"]:
        assert row["config_changes"]["candidate"]["parameters"]["C_G"] == row["value"]
        assert row["privacy"]["channels"]["gradient"]["unprotected_releases"] == (row["privacy"]["channels"]["gradient"]["releases"] if row["secondary_value"] == 0 else 0)
    chosen = max(sweep["rows"], key=lambda r: (r["metrics"]["val_accuracy"], -r["metrics"]["val_loss"]))
    assert (sweep["selected_value"], sweep["selected_secondary_value"]) == (chosen["value"], chosen["secondary_value"])
    ablation = {r["label"]: r for r in suites["ablation"]["rows"]}
    for name in ("minus Activation Clipping", "minus Gradient DP", "minus FedProx"):
        assert ablation[name]["source"] == "measured_training"
    assert ablation["minus Dynamic Split"]["status"] == "not_applicable"


def test_sweep_dimensions_must_be_distinct():
    with pytest.raises(ValueError, match="different parameters"):
        ExperimentConfig(evaluation=EvaluationConfig(sweep_secondary_parameter="C_G"))
