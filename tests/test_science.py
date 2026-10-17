import json
import math
from types import SimpleNamespace

import pytest

from sfl_lab.config import EnergyConfig, ExperimentConfig, NetworkConfig
from sfl_lab.science import RDPAccountant, energy_cost, network_cost, run_attacks, select_profiled_split


def test_gaussian_replace_one_accounting_and_composition():
    acc = RDPAccountant(delta=1e-5)
    assert acc.epsilon == 0
    acc.record(clip=2, sigma=3, count=2, mechanism="activation")
    expected = min(a * 4 / 9 + math.log(1e5) / (a - 1) for a in acc.orders)
    assert acc.epsilon == pytest.approx(expected)
    prior = acc.epsilon
    acc.record(clip=7, sigma=3, count=2, mechanism="gradient")
    assert acc.epsilon > prior
    # Scaling clip and Gaussian std together leaves the multiplier bound fixed.
    other = RDPAccountant(delta=1e-5)
    other.record(clip=9, sigma=3, count=4)
    assert other.epsilon == pytest.approx(acc.epsilon)
    noisier = RDPAccountant(delta=1e-5)
    noisier.record(clip=2, sigma=6, count=4)
    assert noisier.epsilon < acc.epsilon
    assert acc.summary()["release_count"] == 4
    assert acc.summary()["end_to_end_dp"] is False


def test_unprotected_channel_prevents_finite_transcript_claim():
    acc = RDPAccountant(1e-5)
    acc.record(1, 2, mechanism="gradient")
    acc.record(None, 0, mechanism="activation")
    assert acc.epsilon is None
    assert acc.summary()["status"] == "UNBOUNDED"
    assert acc.summary()["channels"]["gradient"]["epsilon"] is not None
    assert acc.summary()["unprotected_releases"] == 1
    json.dumps(acc.summary(), allow_nan=False)


def test_extreme_positive_noise_multiplier_never_divides_by_underflowed_square():
    tiny = RDPAccountant(1e-5)
    tiny.record(1, 1e-200)
    assert tiny.epsilon is None
    assert "representability" in tiny.summary()["reason"]
    json.dumps(tiny.summary(), allow_nan=False)
    large = RDPAccountant(1e-5)
    large.record(1, 1e200)
    assert large.epsilon is not None and large.epsilon > 0


@pytest.mark.parametrize("kwargs", [{"delta": 0}, {"delta": 1}, {"delta": float("nan")},
                                    {"delta": .1, "epsilon_max": -1}, {"delta": .1, "warning_fraction": 1}])
def test_accountant_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RDPAccountant(**kwargs)


@pytest.mark.parametrize("args", [(-1, 1, 1), (1, -1, 1), (1, 1, -1), (1, 1, 1.5), (1, float("inf"), 1)])
def test_accountant_invalid_releases(args):
    with pytest.raises(ValueError):
        RDPAccountant(.01).record(*args)


def test_network_units_loss_and_measured_message_count():
    cfg = NetworkConfig(bandwidth_mbps=8, latency_ms=10, packet_loss=.2)
    result = network_cost({"activation": 1_000_000, "intermediate": 500_000}, cfg,
                          message_counts={"activation": 2, "intermediate": 1})
    assert result["transmit_seconds"] == pytest.approx(1.25)
    assert result["receive_seconds"] == pytest.approx(.625)
    assert result["total_seconds"] == pytest.approx(1.905)
    assert result["expected_wire_bytes"] == pytest.approx(1_875_000)
    assert result["expected_dropped_messages"] == 0
    drop = network_cost({"activation": 1_000_000}, cfg.model_copy(update={"retransmission": False}))
    assert drop["total_seconds"] == pytest.approx(1.01)
    assert drop["expected_lost_bytes"] == pytest.approx(200_000)
    assert drop["dropped_rounds"] is None
    assert drop["accuracy_impact_simulated"] is False


def test_network_deterministic_jitter_and_invalid_bytes():
    cfg = NetworkConfig(jitter_ms=5)
    assert network_cost({"activation": 100}, cfg, 3) == network_cost({"activation": 100}, cfg, 3)
    assert network_cost({}, cfg)["total_seconds"] == 0
    with pytest.raises(ValueError):
        network_cost({"activation": -1}, cfg)
    with pytest.raises(ValueError):
        network_cost({"activation": 3.5}, cfg)
    with pytest.raises(ValueError):
        network_cost({"activation": 1}, cfg, message_counts={"activation": 0})


def test_energy_units_and_disabled():
    # 2 GHz, 2 billion assumed cycles/sec, two seconds -> 1.6 Joules compute.
    result = energy_cost(2, {"transmit_seconds": 3, "receive_seconds": 4}, EnergyConfig())
    assert result["compute_joules"] == pytest.approx(1.6)
    assert result["total_joules"] == pytest.approx(1.6 + 4.2 + 3.6)
    assert result["kind"] == "SIMULATED_ESTIMATE"
    assert energy_cost(2, {}, EnergyConfig(enabled=False))["total_joules"] is None
    with pytest.raises(ValueError):
        energy_cost(-1, {}, EnergyConfig())


def test_offline_split_selection_constraints_and_hysteresis():
    profiles = [
        {"split": [2, 4, 2], "metrics": {"val_accuracy": 90, "runtime_seconds": 10, "energy_joules": 8, "communication_bytes": 100, "epsilon": None}},
        {"split": [1, 6, 1], "metrics": {"val_accuracy": 91, "runtime_seconds": 5, "energy_joules": 4, "communication_bytes": 50, "epsilon": 4}},
    ]
    held = select_profiled_split(profiles, [2, 4, 2])
    assert held["selected_split"] == [2, 4, 2]
    changed = select_profiled_split(profiles, [2, 4, 2], bandwidth_changed=True)
    assert changed["selected_split"] == [1, 6, 1]
    assert changed["changed"] is True
    blocked = select_profiled_split(profiles, [2, 4, 2], limits={"max_epsilon": 1})
    assert blocked["status"] == "inconclusive"


def test_profile_selection_supports_disabled_energy_and_rejects_missing_active_metrics():
    profiles = [{"split": [2, 4, 2], "metrics": {"val_accuracy": 90, "runtime_seconds": 10,
                                                "energy_joules": None, "communication_bytes": 100}}]
    assert select_profiled_split(profiles, [2, 4, 2])["status"] == "inconclusive"
    result = select_profiled_split(profiles, [2, 4, 2], weights={"energy": 0})
    assert result["status"] == "selected"
    profiles[0]["metrics"]["val_accuracy"] = "missing"
    assert select_profiled_split(profiles, [2, 4, 2])["status"] == "inconclusive"
    with pytest.raises(ValueError, match="unknown split constraint"):
        select_profiled_split([], [2, 4, 2], limits={"memory_typo": 1})


def test_profile_selection_requires_trigger_and_material_hysteresis_improvement():
    profiles = [
        {"split": [2, 4, 2], "metrics": {"val_accuracy": 90, "runtime_seconds": 10, "energy_joules": 10, "communication_bytes": 100}},
        {"split": [1, 6, 1], "metrics": {"val_accuracy": 90, "runtime_seconds": 9.9, "energy_joules": 9.9, "communication_bytes": 99}},
    ]
    held = select_profiled_split(profiles, [2, 4, 2], battery_fraction=.2)
    assert held["condition_triggered"] is True
    assert held["changed"] is False
    changed = select_profiled_split(profiles, [2, 4, 2], battery_fraction=.2, hysteresis=.001)
    assert changed["changed"] is True
    with pytest.raises(ValueError):
        select_profiled_split(profiles, [2, 4, 2], hysteresis=float("nan"))


def test_attacks_run_real_pixel_optimization_and_serialize():
    import torch
    torch.set_num_threads(2)
    config = ExperimentConfig(model="tiny_cnn", samples=60, clients=2, seeds=[3],
                              evaluation={"attack_steps": 2})
    result = run_attacks(config)
    assert result["status"] == "completed"
    assert len(result["rows"]) == 5
    assert result["samples"] == 3
    assert len({example["training_index"] for example in result["examples"]}) == 3
    assert result["checkpoint_unchanged"] is True
    trained = result["reference_training"]
    assert trained["variant"] == "baseline"
    assert trained["completed_steps"] > 0
    assert trained["final_state_hash"] != trained["initial_state_hash"]
    assert result["success_rate"] is None
    for row in result["rows"]:
        assert len(row["sample_results"]) == 3
        for attack in ("feature", "gradient"):
            measurements = row[attack]
            assert measurements["steps"] == 2
            assert measurements["final_objective"] <= measurements["initial_objective"]
            assert 0 <= measurements["mse"] <= 1
            assert -1 <= measurements["ssim"] <= 1
            assert measurements["reconstruction_png"].startswith("data:image/png;base64,")
            assert measurements["samples"] == 3
            samples = [sample[attack] for sample in row["sample_results"]]
            assert measurements["mse"] == pytest.approx(sum(sample["mse"] for sample in samples) / 3)
            assert measurements["mse_std"] >= 0
            assert all(sample["final_objective"] <= sample["initial_objective"] for sample in samples)
    clean = result["rows"][0]["feature"]
    assert clean["final_objective"] < clean["initial_objective"]
    json.dumps(result, allow_nan=False)


def test_attack_cancellation_and_unimplemented_threat():
    config = ExperimentConfig(model="tiny_cnn", samples=60)
    assert run_attacks(config, cancelled=lambda: True)["status"] == "cancelled"
    config.privacy.threat_model = "parameter_server"
    assert run_attacks(config)["status"] == "not_applicable"
