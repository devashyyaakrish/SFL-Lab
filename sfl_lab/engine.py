"""Actual U-shaped split training with isolated autograd graphs at both cuts.

The simulator is a single trusted process. Data ownership is a protocol invariant,
not an operating-system security boundary; costs are explicit model estimates.
"""
from __future__ import annotations

import copy
from hashlib import sha256
import math
import platform
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig
from .datasets import prepare_data
from .models import SUPPORTED_SPLITS, UShapedModel, build_model

_WARMED_RUNTIMES: set[tuple] = set()


def _warm_runtime(config, classes, split):
    """Initialize lazy optimizer/kernel code outside the measured comparison.

    A temporary model sees only zero tensors. The random stream is restored;
    no private example, model update or measured experiment is consumed.
    """
    key = (config.model, config.optimizer, tuple(split), classes)
    if key in _WARMED_RUNTIMES:
        return
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = build_model(config.model, classes, split)
        optimizer_type = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
        optimizer = optimizer_type(model.parameters(), lr=config.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        model(torch.zeros(2, 3, 32, 32)).sum().backward()
        optimizer.step()
    _WARMED_RUNTIMES.add(key)


class TrainingCancelled(RuntimeError):
    pass


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise TrainingCancelled("Experiment cancelled by the user.")


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor_bytes(value) for value in state.values())


def _state_hash(model: nn.Module) -> str:
    digest = sha256()
    # Parameter order is invariant to regrouping the eight blocks into three parts.
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def federated_average(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    if not states or len(states) != len(weights):
        raise ValueError("FedAvg requires one weight per model.")
    weights_t = np.asarray(weights, dtype=float)
    if not np.isfinite(weights_t).all() or (weights_t < 0).any() or weights_t.sum() <= 0:
        raise ValueError("Aggregation weights must be finite, nonnegative, and have positive sum.")
    weights_t /= weights_t.sum()
    result = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            out = torch.zeros_like(value)
            for state, weight in zip(states, weights_t):
                out.add_(state[key], alpha=float(weight))
            result[key] = out
        else:
            # BatchNorm counters are integers; largest observed count is meaningful.
            result[key] = torch.stack([state[key] for state in states]).amax(dim=0)
    return result


def _hook(candidate, hook: str, context: dict):
    if candidate is None:
        return None
    from .formulas import apply_hook
    return apply_hook(candidate, hook, context)


def _transform(candidate, hook: str, variable: str, tensor: torch.Tensor, generator) -> torch.Tensor:
    output = _hook(candidate, hook, {variable: tensor, "generator": generator})
    if output is None:
        return tensor
    if not isinstance(output, torch.Tensor) or output.shape != tensor.shape:
        raise ValueError(f"{hook} must preserve the message tensor dimensions.")
    if not torch.isfinite(output).all():
        raise ValueError(f"{hook} produced a nonfinite tensor.")
    return output


def _clip_noise(tensor: torch.Tensor, clip: float | None, noise: float, generator) -> torch.Tensor:
    if clip is not None:
        tensor = tensor * torch.clamp(torch.as_tensor(clip, dtype=tensor.dtype, device=tensor.device) / tensor.norm().clamp_min(1e-12), max=1)
    if noise:
        # Without clipping noise is usable experimentally, but has no finite DP bound.
        std = noise * (clip if clip is not None else 1.0)
        tensor = tensor + torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator) * std
    return tensor


def _activation(model, x, candidate, generator):
    a = _transform(candidate, "H1", "A", model.front(x), generator)
    if candidate is not None:
        a = _clip_noise(a, candidate.activation_clip, candidate.activation_noise, generator)
    return _transform(candidate, "H2", "A", a, generator)


def _train_mode(model, batch_size):
    model.train()
    if batch_size == 1:
        # Standard ResNet18 reaches a 1x1 spatial map for 32px images.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def split_backward(model: UShapedModel, x: torch.Tensor, y: torch.Tensor,
                   candidate=None, generator=None, reference=None) -> dict:
    """Calculate real split gradients; the caller performs the optimizer step.

    This public primitive makes numerical equivalence to unsplit backprop testable.
    The server receives only detached A and detached dL/dZ, never x or y.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    a = _activation(model, x, candidate, generator)
    server_a = a.detach().requires_grad_(True)
    z = _transform(candidate, "H3", "Z", model.middle(server_a), generator)
    client_z = z.detach().requires_grad_(True)
    predictions = model.back(client_z)
    loss = F.cross_entropy(predictions, y)
    changed_loss = _hook(candidate, "H4", {"loss": loss, "generator": generator})
    if changed_loss is not None:
        if not isinstance(changed_loss, torch.Tensor) or changed_loss.numel() != 1:
            raise ValueError("H4 must return a scalar differentiable loss tensor.")
        loss = changed_loss
    if candidate is not None and reference is not None:
        parameters = list(model.parameters())
        if candidate.fedprox_mu > 0 or (candidate.mode == "formula" and candidate.hook == "H14"):
            w = torch.cat([p.flatten() for p in parameters])
            w_global = torch.cat([p.flatten() for p in reference])
            regularizer = .5 * candidate.fedprox_mu * (w - w_global).square().sum()
            modified = _hook(candidate, "H14", {"regularizer": regularizer, "w": w, "w_global": w_global, "generator": generator})
            if modified is not None:
                regularizer = modified
            loss = loss + regularizer
    if not torch.isfinite(loss):
        raise ValueError("Candidate loss is not finite.")
    loss.backward()
    if client_z.grad is None:
        raise ValueError("The candidate disconnected loss from the client-back input.")
    gradient = _transform(candidate, "H5", "G", client_z.grad.detach(), generator)
    if candidate is not None:
        gradient = _clip_noise(gradient, candidate.gradient_clip, candidate.gradient_noise, generator)
    gradient = _transform(candidate, "H6", "G", gradient, generator).detach()
    z.backward(gradient)
    if server_a.grad is None:
        raise ValueError("The candidate disconnected the server graph from its input.")
    front_gradient = server_a.grad.detach()
    a.backward(front_gradient)
    counts = {"activation": tensor_bytes(server_a), "intermediate": tensor_bytes(client_z),
              "gradient_back": tensor_bytes(gradient), "gradient_front": tensor_bytes(front_gradient)}
    return {"loss": float(loss.detach()), "correct": int((predictions.argmax(1) == y).sum()),
            "samples": len(y), "communication": counts,
            "shapes": {"activation": list(a.shape), "intermediate": list(z.shape),
                       "gradient_back": list(gradient.shape), "gradient_front": list(front_gradient.shape)},
            "cuts_detached": server_a.is_leaf and client_z.is_leaf,
            "finite_gradients": all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())}


def _evaluate(model, x, y, batch_size, candidate, generator, cancelled=None):
    model.eval()
    total_loss, correct = 0.0, 0
    communication = {"activation": 0, "intermediate": 0}
    batches = 0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            _check_cancel(cancelled)
            a = _activation(model, x[start:start + batch_size], candidate, generator)
            z = _transform(candidate, "H3", "Z", model.middle(a.detach()), generator)
            output = model.back(z.detach())
            communication["activation"] += tensor_bytes(a)
            communication["intermediate"] += tensor_bytes(z)
            batches += 1
            labels = y[start:start + batch_size]
            loss = F.cross_entropy(output, labels, reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("Evaluation loss is nonfinite; reduce the learning rate or candidate parameters.")
            total_loss += float(loss)
            correct += int((output.argmax(1) == labels).sum())
    return {"accuracy": 100 * correct / len(y), "loss": total_loss / len(y),
            "communication": communication, "batches": batches}


def _privacy_mechanism(candidate, hook):
    if candidate is None:
        return {"clip": 0., "sigma": 0.}
    from .formulas import protection_spec
    builtin_clip = candidate.activation_clip if hook == "H2" else candidate.gradient_clip
    builtin_sigma = candidate.activation_noise if hook == "H2" else candidate.gradient_noise
    candidates = []
    if builtin_clip is not None and builtin_sigma > 0:
        candidates.append({"clip": builtin_clip, "sigma": builtin_sigma})
    formula_spec = protection_spec(candidate, hook)
    if formula_spec and formula_spec.get("clip") and formula_spec.get("sigma", 0) > 0:
        candidates.append(formula_spec)
    # All subsequent hook operations see only the already-protected tensor. A valid
    # existing Gaussian bound remains valid under deterministic/random postprocessing.
    return max(candidates, key=lambda spec: spec["sigma"]) if candidates else {"clip": 0., "sigma": 0.}


def _select_split(config, candidate, classes, seed):
    if candidate is None or candidate.mode != "formula" or candidate.hook != "H10":
        return config.split, None
    measured = []
    for split in SUPPORTED_SPLITS:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = build_model(config.model, classes, list(split)).eval()
        with torch.no_grad():
            a = model.front(torch.zeros(1, 3, 32, 32))
            z = model.middle(a)
        measured.append(2 * (tensor_bytes(a) + tensor_bytes(z)))
    size = torch.tensor(measured, dtype=torch.float32)
    score = size / size.max()
    latency = size * 8 / (config.network.bandwidth_mbps * 1e6) + 4 * config.network.latency_ms / 1000
    output = _hook(candidate, "H10", {"score": score, "S": size, "C": size, "latency": latency,
                                      "energy": latency * (config.energy.tx_power_w + config.energy.rx_power_w)})
    output = torch.as_tensor(output, dtype=torch.float32)
    if output.shape != score.shape or not torch.isfinite(output).all():
        raise ValueError("H10 must return one finite score per profiled split.")
    selected = int(output.argmin())
    return list(SUPPORTED_SPLITS[selected]), {"policy": "one-time offline selection from measured cut tensor sizes; no accuracy prediction or online migration", "profiles": [{"split": list(s), "bytes_per_sample": b, "score": float(v)} for s, b, v in zip(SUPPORTED_SPLITS, measured, output)], "selected": selected}


def run_training(config: ExperimentConfig, variant: str, seed: int, progress: Callable | None = None,
                 cancelled: Callable | None = None, on_model: Callable | None = None) -> dict:
    """Train and evaluate a paired experiment; optionally inspect its fixed checkpoint.

    ``on_model(model)`` runs once after all measured evaluation on the validation-
    selected checkpoint. It is an in-process research hook, never serialized in
    results. Consumers that retain or modify the model should deep-copy it.
    """
    from .science import RDPAccountant, network_cost, energy_cost
    if variant not in ("baseline", "candidate"):
        raise ValueError("variant must be baseline or candidate")
    candidate = config.candidate if variant == "candidate" else None
    if candidate is not None:
        from .formulas import validate_candidate
        validation = validate_candidate(candidate)
        if not validation["valid"]:
            raise ValueError("Candidate validation failed: " + "; ".join(c["detail"] for c in validation["checks"] if c["status"].lower() == "fail"))
        if candidate.mode == "formula" and candidate.hook == "H12" and not config.energy.enabled:
            raise ValueError("H12 requires energy estimation to be enabled.")
    torch.set_num_threads(2)
    _check_cancel(cancelled)
    preparation_start = time.perf_counter()
    data = prepare_data(config, seed)
    preparation_seconds = time.perf_counter() - preparation_start
    warmup_start = time.perf_counter()
    split, split_selection = _select_split(config, candidate, data.num_classes, seed)
    _warm_runtime(config, data.num_classes, split)
    warmup_seconds = time.perf_counter() - warmup_start
    _check_cancel(cancelled)
    started = time.perf_counter()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        global_model = build_model(config.model, data.num_classes, split)
    initial_hash = _state_hash(global_model)
    noise_generator = torch.Generator().manual_seed(seed + 30001)
    # Keep one advancing generator: reseeding every validation round would reuse
    # Gaussian noise and invalidate the independent-release composition assumption.
    evaluation_generator = torch.Generator().manual_seed(seed + 50001)
    selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 20001]))
    communication = dict.fromkeys(("activation", "intermediate", "gradient_back", "gradient_front", "synchronization"), 0)
    message_counts = dict.fromkeys(communication, 0)
    accountant = RDPAccountant(config.privacy.delta, config.privacy.epsilon_max, config.privacy.warning_fraction)
    activation_mechanism = _privacy_mechanism(candidate, "H2")
    gradient_mechanism = _privacy_mechanism(candidate, "H6")
    history, selection_history = [], []
    client_drifts = [None] * config.clients
    best_loss, best_state, best_round = math.inf, None, 0
    completed_steps, stopped_early = 0, False
    shapes, gradient_checks, detached_checks = {}, True, True
    round_client_states = []
    aggregation_times = []
    privacy_diagnostics = []
    privacy_history = []
    evaluation_counts = {"validation": 0, "test": 0, "client_training_partitions": 0}
    evaluation_communication = {"activation": 0, "intermediate": 0}
    total_examples = 0
    model_size = state_bytes(global_model.state_dict())
    middle_size = state_bytes(global_model.middle.state_dict())

    def record_evaluation(model, x, y, phase):
        values = _evaluate(model, x, y, config.batch_size, candidate, evaluation_generator, cancelled)
        for stage, size in values["communication"].items():
            communication[stage] += size
            evaluation_communication[stage] += size
            message_counts[stage] += values["batches"]
        evaluation_counts[phase] += values["batches"]
        accountant.record(**activation_mechanism, count=values["batches"], mechanism="evaluation_activation")
        privacy_history.append({"phase": phase, "training_steps": completed_steps,
                                "release_count": accountant.summary()["release_count"], "epsilon": accountant.epsilon})
        return values

    for round_index in range(config.rounds):
        _check_cancel(cancelled)
        round_start = time.perf_counter()
        selected_count = max(1, min(config.clients, int(math.ceil(config.clients * config.participation))))
        random_scores = torch.tensor(selection_rng.random(config.clients), dtype=torch.float32)
        proposed = _hook(candidate, "H13", {"scores": random_scores, "n": torch.tensor(data.metadata["client_sizes"], dtype=torch.float32)})
        scores = random_scores if proposed is None else torch.as_tensor(proposed)
        if scores.shape != random_scores.shape or not torch.isfinite(scores).all():
            raise ValueError("H13 must return one finite score per client.")
        selected = torch.argsort(scores, descending=True)[:selected_count].tolist()
        local_states, aggregation_sizes, local_ids = [], [], []
        round_correct, round_examples, round_loss = 0, 0, 0.
        round_bytes_before = sum(communication.values())
        reference = [p.detach().clone() for p in global_model.parameters()]
        # Each simulated replica receives and returns a middle-state snapshot. All
        # replicas run sequentially; no parallel speed-up is claimed.
        replica_extra = 2 * max(0, config.servers - 1) * middle_size
        communication["synchronization"] += replica_extra
        message_counts["synchronization"] += 2 * max(0, config.servers - 1)
        for client_id in selected:
            _check_cancel(cancelled)
            model = copy.deepcopy(global_model)
            local_state_size = state_bytes(model.state_dict())
            communication["synchronization"] += local_state_size
            message_counts["synchronization"] += 1
            eta = _hook(candidate, "H7", {"eta": config.learning_rate, "generator": noise_generator})
            rate = config.learning_rate if eta is None else float(eta)
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("H7 must produce a finite, positive learning rate.")
            optimizer_cls = torch.optim.SGD if config.optimizer == "sgd" else torch.optim.Adam
            optimizer = optimizer_cls(model.parameters(), lr=rate)
            client_indices = data.client_indices[client_id]
            for epoch in range(config.local_epochs):
                # Independent of candidate noise consumption and selected client order.
                batch_rng = np.random.default_rng(np.random.SeedSequence([seed, round_index, client_id, epoch, 40001]))
                shuffled = batch_rng.permutation(client_indices)
                for offset in range(0, len(shuffled), config.batch_size):
                    _check_cancel(cancelled)
                    indices = shuffled[offset:offset + config.batch_size]
                    x, y = data.train_x[indices], data.train_y[indices]
                    _train_mode(model, len(indices))
                    optimizer.zero_grad(set_to_none=True)
                    batch = split_backward(model, x, y, candidate, noise_generator, reference)
                    if not batch["finite_gradients"]:
                        raise ValueError("A candidate produced nonfinite gradients.")
                    optimizer.step()
                    for stage, count in batch["communication"].items():
                        communication[stage] += count
                        message_counts[stage] += 1
                    accountant.record(**activation_mechanism, mechanism="activation")
                    accountant.record(**gradient_mechanism, mechanism="gradient")
                    completed_steps += 1
                    privacy_history.append({"phase": "training", "training_steps": completed_steps,
                                            "release_count": accountant.summary()["release_count"], "epsilon": accountant.epsilon})
                    shapes = batch["shapes"]
                    detached_checks &= batch["cuts_detached"]
                    gradient_checks &= batch["finite_gradients"]
                    round_correct += batch["correct"]
                    round_examples += batch["samples"]
                    round_loss += batch["loss"] * batch["samples"]
                    total_examples += batch["samples"]
                    epsilon = accountant.epsilon
                    if config.privacy.stop_on_budget and epsilon is not None and epsilon >= config.privacy.epsilon_max:
                        stopped_early = True
                        break
                if stopped_early:
                    break
            local_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            local_ids.append(client_id)
            aggregation_sizes.append(len(client_indices))
            communication["synchronization"] += local_state_size
            message_counts["synchronization"] += 1
            if stopped_early:
                break
        weights = np.asarray(aggregation_sizes, dtype=float)
        weights /= weights.sum()
        if candidate is not None and candidate.aggregation == "equal":
            weights = np.full(len(local_states), 1 / len(local_states))
        custom_weights = _hook(candidate, "H8", {"weights": torch.tensor(weights), "n": torch.tensor(aggregation_sizes, dtype=torch.float64), "N": sum(aggregation_sizes)})
        if custom_weights is not None:
            weights = torch.as_tensor(custom_weights).detach().cpu().numpy()
            if weights.shape != (len(local_states),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-5):
                raise ValueError("H8 weights must be nonnegative, one per client, and sum to one.")
        aggregation_start = time.perf_counter()
        averaged = federated_average(local_states, weights.tolist())
        global_model.load_state_dict(averaged)
        aggregation_times.append(time.perf_counter() - aggregation_start)
        parameter_keys = set(dict(global_model.named_parameters()))
        for client_id, state in zip(local_ids, local_states):
            squared = sum((state[k] - averaged[k]).double().square().sum().item() for k in parameter_keys)
            client_drifts[client_id] = math.sqrt(squared)
        round_client_states.append({"round": round_index + 1, "clients": local_ids, "weights": weights.tolist(),
                                    "replica_assignments": {str(cid): cid % config.servers for cid in local_ids}})
        selection_history.append(local_ids)
        val = record_evaluation(global_model, data.val_x, data.val_y, "validation")
        if config.privacy.stop_on_budget and accountant.epsilon is not None and accountant.epsilon >= config.privacy.epsilon_max:
            stopped_early = True
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_round = round_index + 1
            best_state = copy.deepcopy(global_model.state_dict())
        elapsed = time.perf_counter() - started
        summary = {"round": round_index + 1, "train_accuracy": 100 * round_correct / round_examples,
                   "train_loss": round_loss / round_examples, "val_accuracy": val["accuracy"], "val_loss": val["loss"],
                   "epsilon": accountant.epsilon, "communication_bytes": sum(communication.values()),
                   "round_communication_bytes": sum(communication.values()) - round_bytes_before,
                   "runtime_seconds": elapsed, "round_runtime_seconds": time.perf_counter() - round_start,
                   "selected_clients": local_ids, "steps": completed_steps}
        history.append(summary)
        if progress is not None:
            progress({"phase": variant, "round": round_index + 1, "rounds": config.rounds,
                      "seed": seed, "metrics": summary, "message": f"{variant}: round {round_index + 1}/{config.rounds}, validation {val['accuracy']:.1f}%"})
        # Candidate accounting is a diagnostic, never a replacement privacy bound.
        if candidate is not None and candidate.mode == "formula" and candidate.hook == "H9":
            if accountant.epsilon is None:
                privacy_diagnostics.append({"round": round_index + 1, "value": None, "certified": False,
                                            "status": "UNAVAILABLE", "note": "No finite RDP transcript bound: protect every released activation and gradient before applying H9."})
            else:
                rdp = accountant.summary()["rdp"]
                rho_result = _hook(candidate, "H9", {"rho": torch.tensor([v["rho"] for v in rdp], dtype=torch.float64),
                                                     "alpha": torch.tensor([v["order"] for v in rdp], dtype=torch.float64), "delta": config.privacy.delta})
                privacy_diagnostics.append({"round": round_index + 1, "value": torch.as_tensor(rho_result).tolist(),
                                            "orders": [v["order"] for v in rdp], "certified": False, "status": "EVALUATED",
                                            "note": "Experimental hook diagnostic from actual composed RDP, excluded from enforced privacy accounting."})
        if stopped_early:
            break

    _check_cancel(cancelled)
    global_model.load_state_dict(best_state)
    # Test is accessed once, only after the validation-selected checkpoint is fixed.
    test = record_evaluation(global_model, data.test_x, data.test_y, "test")
    clients = []
    for client_id, indices in enumerate(data.client_indices):
        values = record_evaluation(global_model, data.train_x[indices], data.train_y[indices], "client_training_partitions")
        clients.append({"id": client_id, "samples": len(indices), "accuracy": values["accuracy"],
                        "loss": values["loss"], "drift": client_drifts[client_id],
                        "metric_scope": "global validation-selected checkpoint on client training partition"})
    runtime = time.perf_counter() - started
    network = network_cost(communication, config.network, seed=seed, message_counts=message_counts)
    network_value = _hook(candidate, "H11", {"S": sum(communication.values()) * 8., "R": config.network.bandwidth_mbps * 1e6,
                                           "p": config.network.packet_loss, "d": config.network.latency_ms / 1000})
    if network_value is not None:
        network["candidate_total_seconds"] = float(network_value)
        network["candidate_note"] = "Candidate equation evaluated over total bits; reference model retained separately."
    energy = energy_cost(runtime, network, config.energy)
    energy_value = _hook(candidate, "H12", {"energy": energy["total_joules"], "E": energy["total_joules"],
                                          "E_tx": energy.get("transmit_joules", 0), "E_rx": energy.get("receive_joules", 0),
                                          "E_compute": energy.get("compute_joules", 0)})
    if energy_value is not None:
        energy["candidate_total_joules"] = float(energy_value)
    privacy = accountant.summary()
    privacy["candidate_diagnostics"] = privacy_diagnostics
    privacy["history"] = privacy_history
    privacy["scope_note"] = "Bounds cover training and evaluation activation/gradient message transcripts under bounded adaptive Gaussian assumptions. Model synchronization, evaluation metric releases, and the full trained model are not certified DP."
    privacy["simulation_note"] = "Seeded noise provides reproducible experiments, not deployable secret randomness. Privacy calculations assume independent Gaussian draws with hidden random seeds."
    privacy["budget_stop_note"] = "Budget stopping halts further local training after a crossing; validation and final evaluation still execute and their releases are included. It is not a strict privacy filter preventing budget overrun."
    drift_values = [v for v in client_drifts if v is not None]
    metrics = {"test_accuracy": test["accuracy"], "test_loss": test["loss"],
               "val_accuracy": history[best_round - 1]["val_accuracy"], "val_loss": best_loss,
               "train_accuracy": sum(c["accuracy"] * c["samples"] for c in clients) / len(data.train_y),
               "train_loss": sum(c["loss"] * c["samples"] for c in clients) / len(data.train_y),
               "runtime_seconds": runtime, "communication_bytes": sum(communication.values()),
               "energy_joules": float(energy_value) if energy_value is not None else energy["total_joules"],
               "epsilon": accountant.epsilon, "drift_mean": float(np.mean(drift_values)) if drift_values else None,
               "fairness_std": float(np.std([c["accuracy"] for c in clients])),
               "network_seconds": float(network_value) if network_value is not None else network["total_seconds"],
               "aggregation_seconds": sum(aggregation_times), "target_loss_round": next((h["round"] for h in history if h["train_loss"] <= config.target_loss), None),
               "convergence_rate": (history[0]["train_loss"] - history[-1]["train_loss"]) / max(1, len(history) - 1)}
    unique_clients = set(cid for selected in selection_history for cid in selected)
    compliance = [
        {"name": "Raw data remained client-side", "status": "PASS", "detail": "Server forward receives detached activations only; logical single-process ownership, not OS isolation."},
        {"name": "Labels remained client-side", "status": "PASS", "detail": "Cross-entropy and client-back backward execute in client scope; no label argument reaches server forward/backward."},
        {"name": "Model split preserved", "status": "PASS", "detail": f"Eight blocks split {split}; stem on client front and prediction head on client back."},
        {"name": "Activation exchange valid", "status": "PASS" if detached_checks else "FAIL", "detail": f"Both receive boundaries are detached leaf tensors; observed shapes {shapes}."},
        {"name": "Backward propagation valid", "status": "PASS" if gradient_checks else "FAIL", "detail": f"Explicit back→server→front backward across {completed_steps} batches with finite parameter gradients."},
        {"name": "Federated aggregation valid", "status": "PASS", "detail": f"{len(history)} rounds of validated normalized aggregation; integer BatchNorm counters use maximum."},
        {"name": "Global redistribution valid", "status": "PASS", "detail": "Each selected client receives a deep copy of the global state before local training; actual tensor payload bytes recorded."},
        {"name": "Multiple clients participated", "status": "PASS" if len(unique_clients) > 1 else "WARNING", "detail": f"{len(unique_clients)} unique participants of {config.clients}; participation can legitimately select one per round."},
        {"name": "Held-out test isolation", "status": "PASS", "detail": f"Checkpoint from round {best_round} selected by validation loss; final test evaluated once, never used for selection."},
    ]
    final_hash = _state_hash(global_model)
    if on_model is not None:
        _check_cancel(cancelled)
        on_model(global_model)
    return {"variant": variant, "seed": seed, "history": history, "metrics": metrics, "clients": clients,
            "communication": communication, "message_counts": message_counts, "compliance": compliance,
            "privacy": privacy, "dataset": data.metadata, "stopped_early": stopped_early, "split": split,
            "split_selection": split_selection, "network": network, "energy": energy,
            "best_round": best_round, "test_evaluations": 1, "completed_steps": completed_steps,
            "data_preparation_seconds": preparation_seconds, "runtime_warmup_seconds": warmup_seconds,
            "initial_state_hash": initial_hash, "final_state_hash": final_hash,
            "evaluation_batches": evaluation_counts, "evaluation_communication": evaluation_communication,
            "client_selection": selection_history, "aggregation": round_client_states,
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                            "torch_threads": torch.get_num_threads(), "platform": platform.platform(),
                            "simulation": "single-process sequential clients and server replicas", "servers": config.servers,
                            "model_parameters": sum(p.numel() for p in global_model.parameters()),
                            "model_state_bytes": model_size, "input_resolution": [32, 32],
                            "memory_note": "Peak hardware memory not measured; model state size reported exactly."},
            "measurement_notes": {"accuracy": "measured", "runtime": "measured wall-clock for initialization, training, aggregation, and evaluation; data preparation and one-time runtime/split profiling warm-up are measured separately to avoid download/import cache bias",
                                  "communication": "exact logical tensor payload bytes for training, validation, final test, client-partition evaluation, and model synchronization; excludes wire headers and serialization framing; no physical network traffic is generated",
                                  "energy": "modeled estimate", "network": "modeled estimate; network loss changes delivery estimates, not training tensors",
                                  "drift": "last participation: norm of local parameters relative to that round's aggregated global parameters",
                                  "fairness": "population standard deviation of client training-partition accuracy", "training_examples_processed": total_examples}}
