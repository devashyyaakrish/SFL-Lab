"""Every advertised executable template reaches its hook in a real model run."""
import math

import pytest

from sfl_lab.config import CandidateConfig, ExperimentConfig
from sfl_lab.engine import run_training
from sfl_lab.formulas import catalog


@pytest.mark.parametrize("template", catalog()["templates"], ids=lambda t: t["id"])
def test_template_real_training(template):
    candidate = CandidateConfig(name=template["name"], hook=template["hook"], expression=template["expression"], parameters=template["parameters"])
    if candidate.hook == "H9":
        candidate.activation_clip = candidate.gradient_clip = 1
        candidate.activation_noise = candidate.gradient_noise = 5
    config = ExperimentConfig(model="tiny_cnn", samples=40, clients=2, rounds=1, seeds=[1], candidate=candidate)
    run = run_training(config, "candidate", 1)
    assert run["completed_steps"] > 0 and math.isfinite(run["metrics"]["test_loss"])
    assert all(c["status"] == "PASS" for c in run["compliance"])
    if candidate.hook == "H9":
        assert run["privacy"]["candidate_diagnostics"][0]["value"] is not None
