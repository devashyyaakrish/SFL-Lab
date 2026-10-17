"""Measured attack probes and explicit, conservative scientific cost models.

The privacy bound is for the recorded, clipped/noised split transcript only.
It is not a claim that trained models, model synchronization, or this simulator
are an end-to-end differentially private system.
"""
from __future__ import annotations

import base64
from hashlib import sha256
import io
import math
import random
import time
from typing import Any


class RDPAccountant:
    """Compose Gaussian releases under replace-one adjacency, without sampling.

    ``clip`` is an L2 radius C and ``sigma`` is a noise multiplier: the actual
    isotropic Gaussian standard deviation must be sigma*C. Two possible clipped
    records differ by at most 2*C. All recorded releases compose sequentially;
    no disjoint-client or subsampling discount is silently assumed.
    """

    orders = (1.25, 1.5, 1.75, 2., 3., 4., 5., 8., 10., 16., 20., 32., 64., 128., 256., 512., 1024.)

    def __init__(self, delta: float, epsilon_max: float = 10, warning_fraction: float = .8):
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must be finite and strictly between zero and one")
        if not math.isfinite(epsilon_max) or epsilon_max <= 0:
            raise ValueError("epsilon_max must be finite and positive")
        if not math.isfinite(warning_fraction) or not 0 < warning_fraction < 1:
            raise ValueError("warning_fraction must be strictly between zero and one")
        self.delta, self.epsilon_max, self.warning_fraction = delta, epsilon_max, warning_fraction
        self._channels: dict[str, dict[str, Any]] = {}

    def record(self, clip: float | None, sigma: float, count: int = 1, mechanism: str = "gradient") -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        if clip is not None and (not math.isfinite(clip) or clip < 0):
            raise ValueError("clip must be nonnegative and finite, or None for unbounded")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be nonnegative and finite")
        if not isinstance(mechanism, str) or not mechanism:
            raise ValueError("mechanism must be a nonempty channel name")
        if count == 0:
            return
        channel = self._channels.setdefault(mechanism, {"count": 0, "unprotected": 0, "coefficient": 0.})
        channel["count"] += count
        if clip is None or clip == 0 or sigma == 0:
            channel["unprotected"] += count
        else:
            # D_alpha(N(mu,s^2 I) || N(mu',s^2 I)) <= alpha*(2C)^2/(2*(sigma*C)^2).
            # Sequential division avoids both underflow of sigma**2 and
            # overflow when a very large finite multiplier is supplied.
            channel["coefficient"] += count * max(2. / sigma / sigma, math.ulp(0.))

    def _epsilon_for(self, channels: list[dict[str, Any]]) -> float | None:
        if any(c["unprotected"] for c in channels):
            return None
        coefficient = sum(c["coefficient"] for c in channels)
        if coefficient == 0:
            return 0.
        value = min(a * coefficient + math.log(1 / self.delta) / (a - 1) for a in self.orders)
        return value if math.isfinite(value) else None

    @property
    def epsilon(self) -> float | None:
        return self._epsilon_for(list(self._channels.values()))

    def summary(self) -> dict:
        epsilon = self.epsilon
        count = sum(c["count"] for c in self._channels.values())
        unprotected = sum(c["unprotected"] for c in self._channels.values())
        status = ("UNBOUNDED" if epsilon is None else "FAIL" if epsilon >= self.epsilon_max
                  else "WARNING" if epsilon >= self.warning_fraction * self.epsilon_max else "SAFE")
        coefficient = sum(c["coefficient"] for c in self._channels.values())
        reason = ("At least one recorded release lacks a verified finite clipping bound plus positive Gaussian noise; no finite transcript epsilon is certified."
                  if unprotected else "The computed transcript bound exceeds floating-point representability."
                  if epsilon is None else "Conservative sequential Gaussian RDP bound on recorded split releases only.")
        return {
            "epsilon": epsilon, "delta": self.delta, "epsilon_max": self.epsilon_max,
            "warning_fraction": self.warning_fraction, "status": status, "budget_status": status,
            "release_count": count, "protected_releases": count - unprotected,
            "unprotected_releases": unprotected, "reason": reason,
            "adjacency": "replace_one", "sensitivity": "2 * clip", "sampling_amplification": False,
            "noise_definition": "Gaussian standard deviation = sigma * clip",
            "scope": "recorded_split_transcript_only", "end_to_end_dp": False,
            "excluded": ["model synchronization", "final model release", "unrecorded side channels", "client-level privacy"],
            "assumptions": ["L2 clipping occurs before transmission", "independent calibrated Gaussian noise per release",
                            "all externally observed split channels are recorded", "data-independent accountant parameters"],
            "rdp": [{"order": a, "rho": a * coefficient if math.isfinite(a * coefficient) else None} for a in self.orders],
            "channels": {name: {"releases": c["count"], "unprotected_releases": c["unprotected"],
                                "epsilon": self._epsilon_for([c])} for name, c in self._channels.items()},
        }


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def network_cost(byte_counts: dict, config: Any, seed: int = 0, message_counts: dict | None = None) -> dict:
    """Estimate aggregate transfer cost from actual tensor payload sizes.

    Loss is an expected-value model, not a socket or packet simulation. With
    retransmission, expected attempts are 1/(1-p). Without it, payload is sent
    once and expected loss is reported; the caller must explicitly simulate
    client/round removal if it wishes to model effects on training accuracy.
    """
    bandwidth = _finite_nonnegative(config.bandwidth_mbps, "bandwidth_mbps")
    if bandwidth == 0:
        raise ValueError("bandwidth_mbps must be positive")
    p = _finite_nonnegative(config.packet_loss, "packet_loss")
    if p >= 1:
        raise ValueError("packet_loss must be less than one")
    latency = _finite_nonnegative(config.latency_ms, "latency_ms") / 1000
    jitter = _finite_nonnegative(config.jitter_ms, "jitter_ms") / 1000
    rng = random.Random(seed)
    multiplier = 1 / (1 - p) if config.retransmission else 1.
    stages, tx_seconds, rx_seconds = {}, 0., 0.
    upload_stages = {"activation", "gradient_back", "client_to_server", "upload"}
    download_stages = {"intermediate", "gradient_front", "server_to_client", "download"}
    for name, raw in byte_counts.items():
        size = _finite_nonnegative(raw, f"byte_counts[{name}]")
        if size != int(size):
            raise ValueError("tensor byte counts must be integer-valued")
        messages = (message_counts or {}).get(name, 1 if size else 0)
        if isinstance(messages, bool) or not isinstance(messages, int) or messages < 0:
            raise ValueError("message counts must be nonnegative integers")
        if size and messages == 0:
            raise ValueError("a nonempty payload must have at least one message")
        transfer = size * 8 / (bandwidth * 1e6) * multiplier
        # A single seeded jitter scenario per flow. Mean jitter is zero before
        # clamping at zero; base latency is charged per measured message.
        one_delay = max(0., latency + (rng.uniform(-jitter, jitter) if jitter else 0.))
        delay = messages * one_delay
        stage_seconds = transfer + delay
        stages[name] = {"bytes": int(size), "messages": messages, "transfer_seconds": transfer,
                        "delay_seconds": delay, "total_seconds": stage_seconds}
        if name in upload_stages:
            tx_seconds += transfer
        elif name in download_stages:
            rx_seconds += transfer
        else:
            # Model synchronization and unknown bidirectional stages are shared.
            tx_seconds += transfer / 2
            rx_seconds += transfer / 2
    total_bytes = sum(s["bytes"] for s in stages.values())
    messages = sum(s["messages"] for s in stages.values())
    total = sum(s["total_seconds"] for s in stages.values())
    return {
        "kind": "SIMULATED_ESTIMATE", "model": "aggregate expected transfer with seeded bounded jitter",
        "total_seconds": total, "transfer_seconds": tx_seconds + rx_seconds,
        "transmit_seconds": tx_seconds, "receive_seconds": rx_seconds,
        "total_bytes": total_bytes, "expected_wire_bytes": total_bytes * multiplier,
        "expected_retransmitted_bytes": total_bytes * (multiplier - 1),
        "expected_lost_bytes": 0. if config.retransmission else total_bytes * p,
        "expected_dropped_messages": 0. if config.retransmission else messages * p,
        "expected_attempts_per_message": multiplier, "message_count": messages,
        "stages": stages, "latency_counting": "measured messages" if message_counts is not None else "one aggregate transfer per nonempty stage",
        "accuracy_impact_simulated": False, "dropped_rounds": None,
        "formula": "bytes*8/(bandwidth_mbps*1e6)/(1-p) + message_count*max(0,latency+jitter)" if config.retransmission else "bytes*8/(bandwidth_mbps*1e6) + message_count*max(0,latency+jitter)",
        "limitations": ["Not real network traffic; payload bytes exclude wire headers, encryption and serialization overhead.",
                        "Loss denotes an expected message/payload fraction, not a measured packet-loss trace.",
                        "Latency and loss alone do not change training accuracy in this cost model; dropped-round simulations are separate.",
                        "Synchronization transmit/receive time is split equally when directional counts are unavailable.",
                        "Base latency is charged once per logical message; retransmission timeouts, acknowledgments and repeated propagation delay are excluded.",
                        "Idle power during link latency is excluded from communication energy."],
    }


def energy_cost(compute_seconds: float, network: dict, config: Any) -> dict:
    seconds = _finite_nonnegative(compute_seconds, "compute_seconds")
    if not config.enabled:
        return {"kind": "DISABLED", "total_joules": None, "reason": "Energy estimation disabled."}
    frequency = _finite_nonnegative(config.frequency_ghz, "frequency_ghz") * 1e9
    cycles_rate = _finite_nonnegative(config.cycles_per_second, "cycles_per_second")
    kappa = _finite_nonnegative(config.kappa, "kappa")
    tx_power = _finite_nonnegative(config.tx_power_w, "tx_power_w")
    rx_power = _finite_nonnegative(config.rx_power_w, "rx_power_w")
    tx_seconds = _finite_nonnegative(network.get("transmit_seconds", 0), "transmit_seconds")
    rx_seconds = _finite_nonnegative(network.get("receive_seconds", 0), "receive_seconds")
    cycles = seconds * cycles_rate
    compute = kappa * frequency**2 * cycles
    tx, rx = tx_power * tx_seconds, rx_power * rx_seconds
    total = compute + tx + rx
    if not math.isfinite(total):
        raise ValueError("energy estimate overflow; reduce input magnitudes")
    return {"kind": "SIMULATED_ESTIMATE", "total_joules": total, "compute_joules": compute,
            "transmit_joules": tx, "receive_joules": rx, "measured_compute_seconds": seconds,
            "assumed_cycles": cycles, "frequency_hz": frequency,
            "formula": "kappa*frequency_hz^2*(compute_seconds*assumed_cycles_per_second) + tx_power_w*transmit_seconds + rx_power_w*receive_seconds",
            "limitations": ["Compute duration is measured; cycles, frequency, capacitance coefficient, and radio power are assumed.",
                            "This is aggregate modeled energy, not hardware telemetry or a per-client battery measurement.",
                            "Server/client compute is not disaggregated and idle network waiting power is excluded."]}


def select_profiled_split(profiles: list[dict], current_split: list[int], weights: dict | None = None,
                          limits: dict | None = None, hysteresis: float = .05,
                          bandwidth_changed: bool = False, battery_fraction: float | None = None) -> dict:
    """Select among real offline measurements; never manufacture accuracy.

    An online change is considered only when the caller reports a material
    bandwidth change or battery is below 30%. A relative objective improvement
    of ``hysteresis`` prevents small fluctuations moving the split repeatedly.
    ``limits`` keys: min_accuracy, max_energy, max_memory, max_epsilon.
    """
    hysteresis = _finite_nonnegative(hysteresis, "hysteresis")
    if hysteresis >= 1:
        raise ValueError("hysteresis must be in [0,1)")
    if battery_fraction is not None:
        battery_fraction = _finite_nonnegative(battery_fraction, "battery_fraction")
        if battery_fraction > 1:
            raise ValueError("battery_fraction must be in [0,1]")
    def valid_split(split):
        return (isinstance(split, list) and len(split) == 3 and
                all(isinstance(s, int) and not isinstance(s, bool) and s >= 1 for s in split) and sum(split) == 8)
    if not valid_split(current_split):
        raise ValueError("current split must contain three positive block counts summing to eight")
    w = {"accuracy": 1., "latency": 1., "energy": 1., "communication": 1.}
    if weights:
        if set(weights) - set(w):
            raise ValueError("unknown split objective weight")
        w.update(weights)
    for key, value in w.items():
        _finite_nonnegative(value, f"weight {key}")
    if sum(w.values()) == 0:
        raise ValueError("at least one objective weight must be positive")
    limits = limits or {}
    if set(limits) - {"min_accuracy", "max_energy", "max_memory", "max_epsilon"}:
        raise ValueError("unknown split constraint")
    for key, bound in limits.items():
        _finite_nonnegative(bound, key)
    def valid_measurement(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    candidates, rejected = [], []
    for profile in profiles:
        split, metrics = profile.get("split"), profile.get("metrics", profile)
        if not valid_split(split):
            raise ValueError("profile split must contain three positive block counts summing to eight")
        values = {"accuracy": metrics.get("val_accuracy"),
                  "latency": metrics.get("latency_seconds", metrics.get("runtime_seconds")),
                  "energy": metrics.get("energy_joules"), "communication": metrics.get("communication_bytes")}
        reasons = []
        for key, val in values.items():
            if w[key] and not valid_measurement(val):
                reasons.append(f"missing measured/modelled {key} objective")
        if valid_measurement(values["accuracy"]) and values["accuracy"] > 100:
            reasons.append("validation accuracy must be a percentage")
        for key, field, lower in [("min_accuracy", "val_accuracy", True), ("max_energy", "energy_joules", False),
                                  ("max_memory", "memory_bytes", False), ("max_epsilon", "epsilon", False)]:
            if key in limits:
                bound = _finite_nonnegative(limits[key], key)
                observed = metrics.get(field)
                if not valid_measurement(observed) or (observed < bound if lower else observed > bound):
                    reasons.append(f"constraint {key} not satisfied or not measured")
        if reasons:
            rejected.append({"split": split, "reasons": reasons})
        else:
            candidates.append({"split": split, "values": values})
    if not candidates:
        return {"status": "inconclusive", "selected_split": current_split, "changed": False,
                "reason": "No profile satisfies the declared measured objectives and constraints.", "rejected": rejected, "scores": []}
    maxima = {key: max((row["values"][key] for row in candidates), default=0) or 1
              for key in w if key != "accuracy" and w[key]}
    for row in candidates:
        row["objective"] = sum(w[key] * ((1 - value / 100) if key == "accuracy" else value / maxima[key])
                               for key, value in row["values"].items() if w[key])
    best = min(candidates, key=lambda row: row["objective"])
    current = next((row for row in candidates if row["split"] == current_split), None)
    triggered = bandwidth_changed or (battery_fraction is not None and battery_fraction < .3)
    changed, selected, reason = False, current_split, "No material condition-change trigger; preserve the current split."
    if current is None:
        changed, selected, reason = best["split"] != current_split, best["split"], "Current split is infeasible or unprofiled; selected best feasible measured profile."
    elif triggered:
        improvement = (current["objective"] - best["objective"]) / max(abs(current["objective"]), 1e-12)
        if best["split"] != current_split and improvement > hysteresis:
            changed, selected, reason = True, best["split"], "Material condition change and objective improvement exceed hysteresis."
        else:
            reason = "Improvement does not exceed hysteresis; preserve the current split."
    return {"status": "selected", "selected_split": selected, "changed": changed, "reason": reason,
            "offline_best_split": best["split"], "scores": candidates, "rejected": rejected,
            "weights": w, "hysteresis": hysteresis, "condition_triggered": triggered,
            "scope": "selection over offline profiles only; no live layer migration or every-minibatch repartitioning"}


def _clip_tensor(tensor, radius: float):
    """Match training: one L2 clipping ball for the entire released tensor."""
    return tensor * (radius / tensor.norm().clamp_min(1e-12)).clamp(max=1)


def _png(tensor) -> str:
    import numpy as np
    from PIL import Image
    array = tensor.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    image = Image.fromarray(np.uint8(np.rint(array * 255)))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _image_metrics(reference, recovered) -> dict:
    import torch
    import torch.nn.functional as F
    difference = (reference - recovered).square()
    mse = float(difference.mean().detach())
    # Standard local Gaussian-window SSIM on [0,1] pixels, channel averaged.
    window = min(11, reference.shape[-1], reference.shape[-2])
    if window % 2 == 0:
        window -= 1
    axis = torch.arange(window, device=reference.device, dtype=reference.dtype) - window // 2
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).expand(reference.shape[1], 1, window, window)
    conv = lambda x: F.conv2d(x, kernel, groups=reference.shape[1])
    ux, uy = conv(reference), conv(recovered)
    vx = (conv(reference.square()) - ux.square()).clamp_min(0)
    vy = (conv(recovered.square()) - uy.square()).clamp_min(0)
    covariance = conv(reference * recovered) - ux * uy
    ssim = (((2 * ux * uy + .01**2) * (2 * covariance + .03**2)) /
            ((ux.square() + uy.square() + .01**2) * (vx + vy + .03**2))).mean()
    return {"mse": mse, "psnr_db": -10 * math.log10(mse) if mse > 0 else None,
            "psnr_status": "infinite_exact_reconstruction" if mse == 0 else "finite",
            "ssim": float(ssim.detach()), "ssim_window": window, "data_range": 1.}


def run_attacks(config, progress=None, cancelled=None) -> dict:
    """Optimize real reconstructions against one trained, fixed reference model.

    One baseline is trained with the first configured seed and frozen at its
    validation-selected checkpoint. Each condition observes the same three
    seeded training examples with the same weights and initialization. These
    bounded, non-adaptive probes isolate outgoing protection, not the effects
    of training a separate candidate model or a population success rate.
    """
    import torch
    import torch.nn.functional as F
    from .datasets import prepare_data
    from .engine import TrainingCancelled, run_training
    from .formulas import apply_hook

    threat = config.privacy.threat_model
    if threat in {"malicious_client", "parameter_server"}:
        return {"status": "not_applicable", "rows": [], "threat_model": threat,
                "reason": "This probe requires access to split activations/gradients; the selected adversary's access is not implemented.",
                "limitations": ["No attack-resistance conclusion is available for this threat model."]}
    if cancelled and cancelled():
        return {"status": "cancelled", "rows": []}
    seed = config.seeds[0]
    started = time.perf_counter()
    data = prepare_data(config, seed)
    requested_steps = config.evaluation.attack_steps
    steps = min(requested_steps, 24 if config.model == "resnet18" else 64)
    c_a = config.candidate.activation_clip or config.candidate.parameters.get("C_A", 1.)
    c_g = config.candidate.gradient_clip or config.candidate.parameters.get("C_G", 1.)
    s_a = config.candidate.activation_noise or config.candidate.parameters.get("sigma_A", .5) or .5
    s_g = config.candidate.gradient_noise or config.candidate.parameters.get("sigma_G", .5) or .5
    c_a, c_g = max(float(c_a), 1e-8), max(float(c_g), 1e-8)
    s_a, s_g = max(float(s_a), 0.), max(float(s_g), 0.)

    def protect(tensor, condition, channel, sample_index):
        clip, noise = (c_a, s_a) if channel == "activation" else (c_g, s_g)
        generator = torch.Generator().manual_seed(seed + 719 + sample_index * 177 + (channel == "gradient"))
        output = tensor
        if condition in {"Clipping Only", "Clipping + Noise"}:
            output = _clip_tensor(output, clip)
        if condition in {"Noise Only", "Clipping + Noise"}:
            output = output + torch.randn(output.shape, generator=generator, dtype=output.dtype) * (noise * clip)
        if condition == "Candidate":
            candidate = config.candidate
            radius = candidate.activation_clip if channel == "activation" else candidate.gradient_clip
            multiplier = candidate.activation_noise if channel == "activation" else candidate.gradient_noise
            if radius is not None:
                output = _clip_tensor(output, radius)
            if multiplier:
                output = output + torch.randn(output.shape, generator=generator, dtype=output.dtype) * (multiplier * (radius or 1.))
            key, hook = ("A", "H2") if channel == "activation" else ("G", "H6")
            hooked = apply_hook(candidate, hook, {key: output, "generator": generator})
            if hooked is not None:
                output = hooked
        if output.shape != tensor.shape or not torch.isfinite(output).all():
            raise ValueError("Attack observation protection must preserve shape and finite values")
        return output

    retained = []
    def capture(model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        retained.append(model)

    def reference_progress(event):
        if progress:
            progress({**event, "phase": "attacks", "message": "Train shared attack reference: " + event.get("message", "")})

    if progress:
        progress({"phase": "attacks", "message": "Train one baseline reference checkpoint shared by all reconstruction controls"})
    try:
        reference = run_training(config, "baseline", seed, progress=reference_progress,
                                 cancelled=cancelled, on_model=capture)
    except TrainingCancelled:
        return {"status": "cancelled", "rows": [], "runtime_seconds": time.perf_counter() - started}
    if len(retained) != 1:
        raise RuntimeError("Training did not supply exactly one selected reference checkpoint")
    model = retained[0]

    def state_hash():
        digest = sha256()
        for value in model.state_dict().values():
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    checkpoint_hash = state_hash()
    if checkpoint_hash != reference["final_state_hash"]:
        raise RuntimeError("Attack reference does not match the trained checkpoint fingerprint")
    sample_indices = torch.randperm(len(data.train_y), generator=torch.Generator().manual_seed(seed + 901))[:3].tolist()
    examples = [{"sample_index": i, "training_index": index, "source_png": _png(data.train_x[index:index + 1]),
                 "source_label": int(data.train_y[index])} for i, index in enumerate(sample_indices)]
    rows = []
    result = {"status": "completed", "rows": rows, "source_png": examples[0]["source_png"],
              "source_label": examples[0]["source_label"], "examples": examples, "samples": len(examples),
              "sample_origin": "three seeded examples without replacement from the training partition",
              "model": config.model, "checkpoint": "shared trained baseline, selected by validation loss",
              "reference_training": {key: reference[key] for key in
                  ("variant", "seed", "best_round", "completed_steps", "initial_state_hash", "final_state_hash")},
              "threat_model": threat, "requested_steps": requested_steps, "effective_steps": steps,
              "controls": {"activation_clip": c_a, "gradient_clip": c_g, "activation_noise_multiplier": s_a,
                           "gradient_noise_multiplier": s_g, "fallback_noise_multiplier": .5},
              "success_rate": None,
              "metric_aggregation": "means and population standard deviations across examples; images show sample 0",
              "assumptions": ["White-box auxiliary access to front, middle and back model weights and architecture.",
                              "Oracle access to the true label for gradient matching; labels are not transmitted by the SFL engine.",
                              "Each observation contains one example with known shape and pixel range [0,1].",
                              "Identical trained model, examples, reconstruction initialization and optimizer budget across controls.",
                              "Independent observation noise between examples; matched noise draws between controls for paired comparisons.",
                              "Observed random perturbations are fixed; attacker does not know their random draws.",
                              "Attacker fits the clean representation to the released protected value (a basic, non-adaptive objective)."],
              "limitations": ["Three bounded probes on one trained baseline seed are exploratory, not population attack-success estimates or privacy guarantees.",
                              "No binary success rate is reported because a reconstruction success threshold has not been calibrated.",
                              "This suite performs one additional baseline training run to obtain the shared checkpoint; its cost is reported separately from paired experiment training.",
                              "SSIM uses an 11-pixel Gaussian window (or the largest valid odd window); PSNR uses data range 1.",
                              "Candidate probe applies only outgoing H2/H6 protections and their built-in clip/noise fields to the shared baseline checkpoint; other candidate hooks and candidate-trained weights are not tested.",
                              "Feature and gradient probes isolate one release at a time; they do not jointly reconstruct from the full protocol transcript.",
                              "Samples share the same trained checkpoint and are not independent training replications; no confidence interval is inferred.",
                              "These passive probes do not implement malicious updates, poisoning, cryptography, or active adversaries."]}
    result["reference_training"]["runtime_seconds"] = reference["metrics"]["runtime_seconds"]
    result["reference_training"]["validation_accuracy"] = reference["metrics"]["val_accuracy"]
    if threat == "external_eavesdropper":
        result["limitations"].append("External-eavesdropper probe assumes readable unencrypted messages plus auxiliary weights and oracle label; encrypted transport is not modeled.")
    if threat == "malicious_server":
        result["status"] = "partial"
        result["limitations"].append("Only the passive observation capability is probed; selected malicious-server behavior is not implemented.")

    for condition in ("Clean", "Noise Only", "Clipping Only", "Clipping + Noise", "Candidate"):
        row = {"condition": condition, "sample_results": []}
        for sample_index, training_index in enumerate(sample_indices):
            source = data.train_x[training_index:training_index + 1].clone()
            label = data.train_y[training_index:training_index + 1].clone()

            def observe(image, attack):
                features = model.front(image)
                if attack == "feature":
                    return features
                representation = model.middle(features)
                if not representation.requires_grad:
                    representation = representation.detach().requires_grad_(True)
                logits = model.back(representation)
                loss = F.cross_entropy(logits, label)
                return torch.autograd.grad(loss, representation, create_graph=True)[0]

            initial = torch.rand(source.shape, generator=torch.Generator().manual_seed(seed + 902 + sample_index * 1009))
            measured = {"sample_index": sample_index, "training_index": training_index}
            for kind in ("feature", "gradient"):
                if cancelled and cancelled():
                    result.update(status="cancelled", runtime_seconds=time.perf_counter() - started)
                    return result
                clean = observe(source.detach().requires_grad_(True), kind).detach()
                target = protect(clean, condition, "activation" if kind == "feature" else "gradient", sample_index).detach()
                recovered = initial.clone().requires_grad_(True)
                optimizer = torch.optim.Adam([recovered], lr=.08)
                scale = target.square().mean().clamp_min(1e-8)
                initial_objective, best_objective, best_image = None, float("inf"), recovered.detach().clone()
                completed = 0
                for step in range(steps + 1):
                    if cancelled and cancelled():
                        result.update(status="cancelled", runtime_seconds=time.perf_counter() - started)
                        return result
                    optimizer.zero_grad(set_to_none=True)
                    predicted = observe(recovered, kind)
                    objective = (predicted - target).square().mean() / scale
                    value = float(objective.detach())
                    if not math.isfinite(value):
                        raise RuntimeError("Nonfinite attack objective; protection or model output requires diagnosis")
                    if initial_objective is None:
                        initial_objective = value
                    if value < best_objective:
                        best_objective, best_image = value, recovered.detach().clone()
                    if step == steps:
                        break
                    objective.backward()
                    optimizer.step()
                    with torch.no_grad():
                        recovered.clamp_(0, 1)
                    completed += 1
                with torch.no_grad():
                    prediction = int(model.back(model.middle(model.front(best_image))).argmax(1)[0])
                measured[kind] = {**_image_metrics(source, best_image), "initial_objective": initial_objective,
                                  "final_objective": best_objective, "steps": completed,
                                  "reconstruction_png": _png(best_image), "predicted_class": prediction,
                                  "label_match": prediction == int(label[0]), "measurement_kind": "MEASURED"}
                if progress:
                    progress({"phase": "attacks", "message": f"{condition}: sample {sample_index + 1}/{len(examples)} {kind} inversion completed ({steps} steps)",
                              "condition": condition, "attack": kind, "sample_index": sample_index})
            row["sample_results"].append(measured)
        for kind in ("feature", "gradient"):
            measurements = [sample[kind] for sample in row["sample_results"]]
            aggregate = {"samples": len(measurements), "steps": steps, "measurement_kind": "MEASURED",
                         "reconstruction_png": measurements[0]["reconstruction_png"], "representative_image_sample_index": 0,
                         "ssim_window": measurements[0]["ssim_window"], "data_range": 1.}
            for metric in ("mse", "psnr_db", "ssim", "initial_objective", "final_objective"):
                values = [measurement[metric] for measurement in measurements]
                if any(value is None for value in values):
                    aggregate[metric], aggregate[metric + "_std"] = None, None
                else:
                    mean = sum(values) / len(values)
                    aggregate[metric] = mean
                    aggregate[metric + "_std"] = math.sqrt(sum((value - mean)**2 for value in values) / len(values))
            aggregate["psnr_status"] = "finite" if aggregate["psnr_db"] is not None else "contains_infinite_exact_reconstruction"
            row[kind] = aggregate
        rows.append(row)
    result["checkpoint_unchanged"] = state_hash() == checkpoint_hash
    if not result["checkpoint_unchanged"]:
        raise RuntimeError("Attack optimization unexpectedly changed its reference model")
    result["runtime_seconds"] = time.perf_counter() - started
    return result
