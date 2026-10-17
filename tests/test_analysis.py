"""Decisions and inferential statistics must follow the stored observations."""
import copy
import math

import pytest
from scipy.stats import ttest_rel

from sfl_lab.analysis import compare_runs, pareto_front
from sfl_lab.config import AcceptanceConfig, ExperimentConfig
from sfl_lab.reports import make_report


def observations(accuracies, *, epsilon=None):
    return [{"seed": seed, "metrics": {"val_accuracy": accuracy, "test_accuracy": accuracy - 1,
             "runtime_seconds": 10., "communication_bytes": 1000, "epsilon": epsilon},
             "privacy": {"protected_releases": 4 if epsilon is not None else 0},
             "dataset": {"split_hashes": {"train": f"train-{seed}", "test": f"test-{seed}"}},
             "initial_state_hash": f"model-{seed}", "compliance": [{"name": "Protocol", "status": "PASS"}]}
            for seed, accuracy in enumerate(accuracies, 1)]


def test_paired_statistics_match_scipy_and_seed_identifiers():
    baseline = observations([80, 82, 79, 84, 81])
    candidate = observations([81, 82.5, 78.5, 85.5, 82])
    config = ExperimentConfig(seeds=[1, 2, 3, 4, 5])
    result = compare_runs(baseline, list(reversed(candidate)), config)
    test = result["statistics"]["val_accuracy"]
    assert test["n_pairs"] == 5
    assert test["p_value"] == pytest.approx(ttest_rel([81, 82.5, 78.5, 85.5, 82], [80, 82, 79, 84, 81]).pvalue)
    assert test["ci95"][0] < test["mean_difference"] < test["ci95"][1]
    assert test["adjusted_p_value"] >= test["p_value"]
    assert result["deltas"]["val_accuracy"]["absolute"] == pytest.approx(.7)


def test_constant_differences_do_not_invent_significance():
    result = compare_runs(observations([80, 81, 82]), observations([81, 82, 83]), ExperimentConfig())
    test = result["statistics"]["val_accuracy"]
    assert test["p_value"] is None and test["significant"] is False
    assert "constant" in test["status"]
    assert result["recommendation"]["decision"] == "ACCEPT"


def test_accuracy_failure_rejects_and_overhead_is_conditional():
    config = ExperimentConfig()
    assert compare_runs(observations([80, 81, 82]), observations([77, 78, 79]), config)["recommendation"]["decision"] == "REJECT"
    candidate = observations([80, 81, 82])
    for run in candidate:
        run["metrics"]["runtime_seconds"] = 12
    assert compare_runs(observations([80, 81, 82]), candidate, config)["recommendation"]["decision"] == "CONDITIONALLY ACCEPT"


def test_unknown_privacy_reduction_is_undefined():
    config = ExperimentConfig(acceptance=AcceptanceConfig(required_privacy_improvement_pct=20))
    result = compare_runs(observations([80, 81, 82]), observations([80, 81, 82], epsilon=3), config)
    assert result["deltas"]["epsilon"]["relative_pct"] is None
    assert result["hypothesis_status"] == "inconclusive"


def test_exceeded_privacy_budget_rejects_even_when_other_criteria_pass():
    result = compare_runs(observations([80, 81, 82]), observations([80, 81, 82], epsilon=20), ExperimentConfig())
    assert result["recommendation"]["decision"] == "REJECT"
    assert next(row for row in result["criteria"] if row["name"] == "Privacy budget")["status"] == "fail"


@pytest.mark.parametrize("change", ["missing_seed", "initialization", "data", "missing_metric", "missing_compliance"])
def test_incomplete_controlled_evidence_is_inconclusive(change):
    baseline, candidate = observations([80, 81, 82]), observations([80, 81, 82])
    if change == "missing_seed": candidate.pop()
    elif change == "initialization": candidate[0]["initial_state_hash"] = "different"
    elif change == "data": candidate[0]["dataset"]["split_hashes"]["test"] = "different"
    elif change == "missing_metric": candidate[0]["metrics"].pop("runtime_seconds")
    else: candidate[0]["compliance"] = []
    assert compare_runs(baseline, candidate, ExperimentConfig())["recommendation"]["decision"] == "INCONCLUSIVE"


def test_core_violation_rejects():
    candidate = observations([80, 81, 82])
    candidate[0]["compliance"][0]["status"] = "FAIL"
    assert compare_runs(observations([80, 81, 82]), candidate, ExperimentConfig())["recommendation"]["decision"] == "REJECT"


def test_pareto_requires_every_selected_objective_and_correct_direction():
    points = [{"label": "a", "val_accuracy": 90, "epsilon": 3},
              {"label": "b", "val_accuracy": 89, "epsilon": 4},
              {"label": "c", "val_accuracy": 94, "epsilon": None}]
    result = pareto_front(points, {"val_accuracy": "higher", "epsilon": "lower"})
    assert [p["non_dominated"] for p in result] == [True, False, None]


@pytest.mark.parametrize("format", ["json", "csv", "pdf", "markdown"])
def test_reports_handle_single_variant_without_fabricated_comparison(format):
    experiment = {"id": "EXP-test", "name": "Single baseline", "status": "completed", "config": ExperimentConfig().model_dump(),
                  "result": {"baseline": observations([80]), "candidate": [], "comparison": {}, "suites": {}, "environment": {}, "limitations": []}, "logs": []}
    report = make_report(experiment, format)
    assert isinstance(report, bytes) and len(report) > 50
    if format == "pdf": assert report.startswith(b"%PDF")
    if format == "markdown": assert b"INCONCLUSIVE" in report
