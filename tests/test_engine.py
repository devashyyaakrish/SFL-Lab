import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sfl_lab.config import CandidateConfig, ExperimentConfig
from sfl_lab.datasets import partition_labels, prepare_data
from sfl_lab.engine import TrainingCancelled, _clip_noise, _state_hash, federated_average, run_training, split_backward, state_bytes
from sfl_lab.models import SUPPORTED_SPLITS, build_model


torch.set_num_threads(2)


def config(**kwargs):
    return ExperimentConfig(model="tiny_cnn", samples=60, clients=3, rounds=1, batch_size=8, seeds=[7], **kwargs)


@pytest.mark.parametrize("model_name", ["tiny_cnn", "resnet18"])
def test_split_backward_matches_end_to_end_autograd(model_name):
    torch.manual_seed(11)
    split = build_model(model_name, 10, [2, 4, 2]).eval()
    unsplit = copy.deepcopy(split)
    x, labels = torch.rand(2, 3, 32, 32), torch.tensor([1, 7])
    expected_loss = F.cross_entropy(unsplit(x), labels)
    expected_loss.backward()
    result = split_backward(split, x, labels)
    assert result["loss"] == pytest.approx(float(expected_loss.detach()), rel=1e-6)
    for actual, expected in zip(split.parameters(), unsplit.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
    assert result["cuts_detached"] and result["finite_gradients"]
    assert result["communication"]["activation"] == result["communication"]["gradient_front"]
    assert result["communication"]["intermediate"] == result["communication"]["gradient_back"]


@pytest.mark.parametrize("split", SUPPORTED_SPLITS)
@pytest.mark.parametrize("model_name", ["tiny_cnn", "resnet18"])
def test_supported_splits(model_name, split):
    model = build_model(model_name, 10, list(split)).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 32, 32)).shape == (2, 10)


def test_all_positive_eight_block_splits():
    for front in range(1, 7):
        for middle in range(1, 8 - front):
            split = [front, middle, 8 - front - middle]
            model = build_model("tiny_cnn", 10, split).eval()
            assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


@pytest.mark.parametrize("partition", ["iid", "quantity", "label_skew", "dirichlet"])
def test_partitions_cover_disjoint_nonempty_training_data(partition):
    labels = np.repeat(np.arange(10), 8)
    actual = partition_labels(labels, 20, partition, .05, 12)
    assert min(map(len, actual)) >= 1
    assert sorted(np.concatenate(actual).tolist()) == list(range(80))
    repeated = partition_labels(labels, 20, partition, .05, 12)
    assert all(np.array_equal(a, b) for a, b in zip(actual, repeated))


def test_validation_and_test_are_separate_reproducible_samples():
    data = prepare_data(config(), 7)
    repeated = prepare_data(config(), 7)
    assert data.metadata["split_hashes"] == repeated.metadata["split_hashes"]
    train, val, test = (set(data.metadata["indices"][k]) for k in ["train", "validation", "test"])
    assert not train & val and not train & test and not val & test
    assert (len(train), len(val), len(test)) == (42, 6, 12)
    assert data.train_x.shape == (42, 3, 32, 32)
    assert data.train_x.min() >= 0 and data.train_x.max() <= 1


def test_fedavg_known_values_and_integer_counters():
    states = [{"w": torch.tensor([1., 4.]), "counter": torch.tensor(2)},
              {"w": torch.tensor([5., 0.]), "counter": torch.tensor(3)}]
    result = federated_average(states, [1, 3])
    torch.testing.assert_close(result["w"], torch.tensor([4., 1.]))
    assert result["counter"].item() == 3
    with pytest.raises(ValueError):
        federated_average(states, [-1, 2])


def test_reference_is_immutable_and_paired_identity_hook_is_exact():
    original = config(candidate=CandidateConfig(hook="H6", expression="G", parameters={}))
    reference = run_training(original, "baseline", 7)
    identity = run_training(original, "candidate", 7)
    assert reference["initial_state_hash"] == identity["initial_state_hash"]
    assert reference["final_state_hash"] == identity["final_state_hash"]
    assert reference["client_selection"] == identity["client_selection"]
    assert reference["dataset"]["split_hashes"] == identity["dataset"]["split_hashes"]
    assert reference["metrics"]["test_accuracy"] == identity["metrics"]["test_accuracy"]
    assert sum(reference["communication"].values()) == reference["metrics"]["communication_bytes"]
    assert reference["test_evaluations"] == 1
    assert all(c["status"] == "PASS" for c in reference["compliance"])
    changed_candidate = original.model_copy(update={"candidate": CandidateConfig(activation_clip=.1, activation_noise=.2, gradient_clip=.2, gradient_noise=.8, fedprox_mu=.1, aggregation="equal")})
    reference_again = run_training(changed_candidate, "baseline", 7)
    assert reference["final_state_hash"] == reference_again["final_state_hash"]


def test_protected_training_accounts_both_released_channels():
    result = run_training(config(candidate=CandidateConfig(hook="H6", expression="G", parameters={}, activation_clip=1, activation_noise=2, gradient_clip=1, gradient_noise=2)), "candidate", 7)
    assert result["metrics"]["epsilon"] is not None
    assert result["metrics"]["epsilon"] > 0
    assert result["completed_steps"] > 0
    assert result["metrics"]["communication_bytes"] > 0
    assert "not certified DP" in result["privacy"]["scope_note"]


def test_cancellation_is_observed_before_data_preparation():
    with pytest.raises(TrainingCancelled):
        run_training(config(), "baseline", 7, cancelled=lambda: True)


def test_clipping_bounds_whole_transmitted_batch_and_calibrates_noise():
    message = torch.tensor([[3., 4.], [3., 4.]])
    clipped = _clip_noise(message, 2., 0., torch.Generator().manual_seed(8))
    assert clipped.norm().item() == pytest.approx(2.)
    assert clipped[0].norm().item() < 2.  # Not a separate radius for every row.
    noised = _clip_noise(message, 2., .7, torch.Generator().manual_seed(8))
    noise = torch.randn(message.shape, generator=torch.Generator().manual_seed(8)) * 1.4
    torch.testing.assert_close(noised, clipped + noise)


def test_fedprox_split_gradients_match_unsplit_objective():
    torch.manual_seed(15)
    model = build_model("tiny_cnn", 10, [2, 4, 2]).eval()
    unsplit = copy.deepcopy(model)
    reference = [p.detach().clone() + .01 for p in model.parameters()]
    x, labels = torch.rand(3, 3, 32, 32), torch.tensor([1, 2, 3])
    loss = F.cross_entropy(unsplit(x), labels)
    loss = loss + .5 * .2 * sum((p - ref).square().sum() for p, ref in zip(unsplit.parameters(), reference))
    loss.backward()
    result = split_backward(model, x, labels, CandidateConfig(expression="G", parameters={}, fedprox_mu=.2), reference=reference)
    assert result["loss"] == pytest.approx(float(loss.detach()), rel=1e-6)
    for actual, expected in zip(model.parameters(), unsplit.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)


def test_every_transmission_and_privacy_release_includes_evaluation():
    cfg = config(servers=3, candidate=CandidateConfig(expression="G", parameters={}, activation_clip=1., activation_noise=2., gradient_clip=1., gradient_noise=2.))
    result = run_training(cfg, "candidate", 7)
    evaluation_batches = sum(result["evaluation_batches"].values())
    steps = result["completed_steps"]
    assert result["privacy"]["release_count"] == 2 * steps + evaluation_batches
    assert result["privacy"]["channels"]["activation"]["releases"] == steps
    assert result["privacy"]["channels"]["gradient"]["releases"] == steps
    assert result["privacy"]["channels"]["evaluation_activation"]["releases"] == evaluation_batches
    assert result["message_counts"]["activation"] == steps + evaluation_batches
    assert result["message_counts"]["intermediate"] == steps + evaluation_batches
    assert result["message_counts"]["gradient_back"] == steps
    assert result["message_counts"]["gradient_front"] == steps
    model = build_model(cfg.model, 10, cfg.split)
    with torch.no_grad():
        a = model.front(torch.zeros(1, 3, 32, 32))
        z = model.middle(a)
    train_samples = result["dataset"]["train"]
    eval_samples = train_samples + result["dataset"]["validation"] + result["dataset"]["test"]
    assert result["communication"]["activation"] == (train_samples + eval_samples) * a.numel() * a.element_size()
    assert result["communication"]["intermediate"] == (train_samples + eval_samples) * z.numel() * z.element_size()
    assert result["communication"]["synchronization"] == 2 * cfg.clients * state_bytes(model.state_dict()) + 2 * (cfg.servers - 1) * state_bytes(model.middle.state_dict())
    assert result["message_counts"]["synchronization"] == 2 * cfg.clients + 2 * (cfg.servers - 1)
    assert result["privacy"]["history"][-1]["epsilon"] == result["metrics"]["epsilon"]
    assert result["metrics"]["epsilon"] > result["history"][-1]["epsilon"]


def test_best_checkpoint_callback_is_not_serialized_or_counted_as_training():
    observed = []
    result = run_training(config(), "baseline", 7, on_model=lambda model: observed.append((_state_hash(model), model.training)))
    assert observed == [(result["final_state_hash"], False)]
    import json
    json.dumps(result, allow_nan=False)


def test_imagefolder_adapter_materializes_real_class_images(tmp_path):
    from PIL import Image
    for label in range(2):
        folder = tmp_path / str(label)
        folder.mkdir()
        for index in range(22):
            pixels = np.zeros((12 + index % 3, 17, 3), dtype=np.uint8)
            pixels[:, :, label] = 100 + index
            Image.fromarray(pixels).save(folder / f"sample-{index}.png")
    cfg = config(dataset="imagefolder", dataset_path=str(tmp_path))
    result = prepare_data(cfg, 7)
    assert result.num_classes == 2
    assert result.metadata["available"] == 44
    assert result.train_x.shape == (30, 3, 32, 32)
    assert result.train_x.dtype == torch.float32
    assert result.train_y.dtype == torch.int64
    assert set(result.train_y.tolist()) == {0, 1}
    run = run_training(cfg, "baseline", 7)
    assert run["test_evaluations"] == 1
    assert run["metrics"]["test_loss"] >= 0


@pytest.mark.parametrize(("hook", "expression", "parameters"), [
    ("H1", "A * .9", {}), ("H2", "clip(A, C_A)", {"C_A": 2.}),
    ("H3", "Z * .9", {}), ("H4", "loss * .9", {}),
    ("H5", "G * .9", {}), ("H6", "G * .9", {}),
    ("H7", "eta * .5", {}), ("H8", "n / N", {}),
    ("H9", "rho + log(1 / delta) / (alpha - 1)", {}),
    ("H10", "score", {}), ("H11", "S / R + d", {}),
    ("H12", "E_compute + E_tx + E_rx", {}), ("H13", "scores + n", {}),
    ("H14", "mu / 2 * norm(w - w_global)**2", {"mu": .1}),
])
def test_each_advertised_hook_executes_inside_real_training(hook, expression, parameters, monkeypatch):
    from sfl_lab import formulas
    original = formulas.apply_hook
    executed = []
    def observe(candidate, current_hook, context):
        value = original(candidate, current_hook, context)
        if current_hook == hook and value is not None:
            executed.append(value.detach().clone() if isinstance(value, torch.Tensor) else value)
        return value
    monkeypatch.setattr(formulas, "apply_hook", observe)
    candidate = CandidateConfig(hook=hook, expression=expression, parameters=parameters)
    if hook == "H9":
        candidate = candidate.model_copy(update={"activation_clip": 1., "activation_noise": 2., "gradient_clip": 1., "gradient_noise": 2.})
    # Validation exercises hooks three times; reset the observer after validation
    # so success requires execution with actual runtime tensors and metrics.
    original_validate = formulas.validate_candidate
    def validate(value):
        outcome = original_validate(value)
        executed.clear()
        return outcome
    monkeypatch.setattr(formulas, "validate_candidate", validate)
    result = run_training(config(candidate=candidate), "candidate", 7)
    assert executed, f"{hook} was validated but never executed by the engine"
    assert result["completed_steps"] > 0
    if hook == "H9":
        diagnostic = result["privacy"]["candidate_diagnostics"][0]
        assert diagnostic["status"] == "EVALUATED"
        assert min(diagnostic["value"]) == pytest.approx(result["history"][0]["epsilon"])
    if hook == "H10":
        assert result["split_selection"]["profiles"]


def test_invalid_candidate_reports_the_actual_failure():
    with pytest.raises(ValueError, match="Candidate validation failed: .+"):
        run_training(config(candidate=CandidateConfig(expression="unknown_symbol", parameters={})), "candidate", 7)


def test_energy_hook_requires_enabled_energy_model():
    from sfl_lab.config import EnergyConfig
    with pytest.raises(ValueError, match="H12 requires energy"):
        run_training(config(candidate=CandidateConfig(hook="H12", expression="energy", parameters={}), energy=EnergyConfig(enabled=False)), "candidate", 7)


def test_budget_stop_records_only_clients_that_actually_trained():
    from sfl_lab.config import PrivacyConfig
    candidate = CandidateConfig(expression="G", parameters={}, activation_clip=1., activation_noise=2., gradient_clip=1., gradient_noise=2.)
    result = run_training(config(candidate=candidate, privacy=PrivacyConfig(epsilon_max=.1, stop_on_budget=True)), "candidate", 7)
    assert result["stopped_early"]
    assert result["completed_steps"] == 1
    assert len(result["client_selection"][0]) == 1
    assert result["privacy"]["release_count"] == 2 + sum(result["evaluation_batches"].values())
    assert result["metrics"]["epsilon"] > result["history"][0]["epsilon"]
    assert next(c for c in result["compliance"] if c["name"] == "Multiple clients participated")["status"] == "WARNING"


def test_gaussian_randomness_is_never_reused_between_evaluations(monkeypatch):
    import sfl_lab.engine as engine
    original = engine._clip_noise
    states = []
    def observe(tensor, clip, noise, generator):
        if noise:
            states.append(generator.get_state().numpy().tobytes())
        return original(tensor, clip, noise, generator)
    monkeypatch.setattr(engine, "_clip_noise", observe)
    cfg = config(candidate=CandidateConfig(expression="G", parameters={}, activation_clip=1., activation_noise=2., gradient_clip=1., gradient_noise=2.)).model_copy(update={"rounds": 2})
    result = run_training(cfg, "candidate", 7)
    assert len(states) == result["privacy"]["release_count"]
    assert len(states) == len(set(states))
