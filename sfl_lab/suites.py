"""Controlled supplementary experiments and explicitly modeled cost replays."""
from __future__ import annotations

import ast
import math
import random
from typing import Any

from .config import ExperimentConfig, NetworkConfig
from .analysis import pareto_front


EXPLORATORY = "Supplementary runs use the first configured seed and are exploratory. Selection uses validation only; test measurements are not selection inputs. Confirm selected settings with fresh data and multiple seeds."


def _row(label: str, run: dict, **extra: Any) -> dict:
    return {"label": label, "seed": run.get("seed"), "variant": run.get("variant"),
            "metrics": run.get("metrics", {}), "clients": run.get("clients", []),
            "history": run.get("history", []), "communication": run.get("communication", {}),
            "privacy": run.get("privacy", {}), "dataset": run.get("dataset", {}),
            "message_counts": run.get("message_counts", {}), "environment": run.get("environment", {}),
            "initial_state_hash": run.get("initial_state_hash"), "compliance": run.get("compliance", []),
            "split": run.get("split"), "source": "measured_training", **extra}


def _change(config: ExperimentConfig, **updates: Any) -> ExperimentConfig:
    data = config.model_dump()
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return ExperimentConfig.model_validate(data)


def _strip_calls(expression: str, names: set[str]) -> str:
    """Remove chosen tensor transforms while preserving surrounding mathematics."""
    class RemoveClip(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in names and node.args:
                return node.args[0]
            return node
    try:
        return ast.unparse(RemoveClip().visit(ast.parse(expression, mode="eval")))
    except SyntaxError:
        return expression


def _parameter_config(config: ExperimentConfig, parameter: str, value: float) -> ExperimentConfig:
    candidate = config.candidate.model_dump()
    candidate["parameters"][parameter] = value
    operations = candidate["algorithm"]
    try:
        formula_names = {node.id for node in ast.walk(ast.parse(candidate["expression"], mode="eval")) if isinstance(node, ast.Name)} if candidate["mode"] == "formula" else set()
    except SyntaxError:
        formula_names = set()
    if parameter == "C_G":
        if value <= 0:
            raise ValueError("C_G must be positive")
        active = "C_G" in formula_names
        if candidate["gradient_clip"] is not None:
            candidate["gradient_clip"] = value
            active = True
        for operation in operations:
            if operation["op"] == "GRADIENT_CLIP":
                operation["params"]["C_G"] = value
                active = True
        if not active:
            raise ValueError("C_G is not active in this candidate; choose a referenced parameter or enable gradient clipping.")
    elif parameter == "sigma_G":
        if value < 0:
            raise ValueError("sigma_G must be nonnegative")
        algorithm_noise = any(operation["op"] == "GRADIENT_GAUSSIAN" for operation in operations)
        # Updating a formula/algorithm must not also inject a second built-in noise mechanism.
        if candidate["gradient_noise"] > 0 or ("sigma_G" not in formula_names and not algorithm_noise):
            candidate["gradient_noise"] = value
        if value == 0 and "sigma_G" in formula_names:
            candidate["expression"] = _strip_calls(candidate["expression"], {"gaussian"})
        if algorithm_noise:
            for operation in operations:
                if operation["op"] == "GRADIENT_GAUSSIAN":
                    operation["params"]["sigma_G"] = value
            if value == 0:
                candidate["algorithm"] = [operation for operation in operations if operation["op"] != "GRADIENT_GAUSSIAN"]
    elif parameter == "mu":
        if value < 0:
            raise ValueError("mu must be nonnegative")
        candidate["fedprox_mu"] = value
    else:
        raise ValueError(f"Unsupported sweep parameter: {parameter}")
    return _change(config, candidate=candidate)


def run_suites(config: ExperimentConfig, base_runs: list[dict], candidate_runs: list[dict],
               progress=None, cancelled=None) -> dict:
    from .engine import run_training
    from .science import network_cost, energy_cost, run_attacks, select_profiled_split

    seed = config.seeds[0]
    results: dict[str, dict] = {}
    cache: dict[tuple[str, str, int], dict] = {}

    def check_cancel():
        if cancelled and cancelled():
            raise InterruptedError("Experiment cancelled during evaluation suites")

    def emit(suite: str, detail: str):
        check_cancel()
        if progress:
            progress({"phase": suite, "message": detail})

    def cache_key(cfg, variant, run_seed):
        data = cfg.model_dump(exclude={"name", "hypothesis", "evaluation", "seeds", "acceptance"})
        if variant == "baseline":
            data.pop("candidate", None)
        import json
        return variant, json.dumps(data, sort_keys=True), run_seed

    for variant, runs in (("baseline", base_runs), ("candidate", candidate_runs)):
        for run in runs:
            if run.get("seed") is not None:
                cache[cache_key(config, variant, run["seed"])] = run

    def train(cfg, variant, label, suite):
        key = cache_key(cfg, variant, seed)
        emit(suite, label)
        if key not in cache:
            def training_progress(event):
                if progress:
                    progress({**event, "phase": suite, "message": f"{label}: {event.get('message', '')}"})
            cache[key] = run_training(cfg, variant, seed, progress=training_progress, cancelled=cancelled)
        return cache[key]

    if config.evaluation.network:
        emit("network", "Replay measured tensor traffic across four modeled network conditions")
        scenarios = [
            ("Good", NetworkConfig(bandwidth_mbps=100, latency_ms=10, packet_loss=.005, jitter_ms=1)),
            ("Moderate", NetworkConfig(bandwidth_mbps=20, latency_ms=30, packet_loss=.05, jitter_ms=5)),
            ("Poor", NetworkConfig(bandwidth_mbps=5, latency_ms=80, packet_loss=.125, jitter_ms=15)),
            ("Severe", NetworkConfig(bandwidth_mbps=2, latency_ms=150, packet_loss=.25, jitter_ms=30)),
        ]
        rows, heatmap = [], []
        for variant, runs in (("baseline", base_runs), ("candidate", candidate_runs)):
            if not runs:
                continue
            run = next((r for r in runs if r.get("seed") == seed), runs[0])
            counts = run.get("communication", {})
            compute = run.get("metrics", {}).get("runtime_seconds", 0)
            for label, network in scenarios:
                check_cancel()
                cost = network_cost(counts, network, seed=seed, message_counts=run.get("message_counts"))
                energy = energy_cost(compute, cost, config.energy)
                rows.append({"label": f"{variant} / {label}", "scenario": label, "variant": variant,
                             "seed": run.get("seed"), "bandwidth_mbps": network.bandwidth_mbps,
                             "packet_loss": network.packet_loss, "source": "modeled_cost_replay",
                             "metrics": {"latency_seconds": cost["total_seconds"],
                                         "communication_bytes": run.get("metrics", {}).get("communication_bytes"),
                                         "energy_joules": energy.get("total_joules")},
                             "network": cost, "energy": energy})
            for bandwidth in (100, 50, 20, 10, 2):
                for loss in (0, .05, .1, .2, .3):
                    check_cancel()
                    network = config.network.model_copy(update={"bandwidth_mbps": bandwidth, "packet_loss": loss})
                    cost = network_cost(counts, network, seed=seed, message_counts=run.get("message_counts"))
                    energy = energy_cost(compute, cost, config.energy)
                    heatmap.append({"variant": variant, "bandwidth_mbps": bandwidth, "packet_loss": loss,
                                    "latency_seconds": cost["total_seconds"], "energy_joules": energy.get("total_joules"),
                                    "source": "modeled_cost_replay"})
        results["network"] = {"status": "completed", "rows": rows, "heatmap": heatmap,
                              "limitations": ["Network settings replay measured bytes through a latency/energy model; they do not retrain models, delay real packets, or alter accuracy/convergence.",
                                               "Dropped rounds and accuracy under packet loss are not measured. Expected retransmission costs are modeled only."]}
    else:
        results["network"] = {"status": "disabled", "rows": []}

    if config.evaluation.non_iid:
        rows = []
        for partition, variant, label in (("iid", "baseline", "Baseline IID"), ("dirichlet", "baseline", "Baseline Non-IID"),
                                          ("dirichlet", "candidate", "Candidate Non-IID")):
            cfg = _change(config, partition=partition)
            rows.append(_row(label, train(cfg, variant, label, "non_iid"), config_changes={"partition": partition}))
        mu = config.candidate.fedprox_mu or config.candidate.parameters.get("mu", 0) or .01
        for value in (0., mu):
            cfg = _parameter_config(_change(config, partition="dirichlet"), "mu", value)
            label = f"Candidate Non-IID / FedProx mu={value:g}"
            rows.append(_row(label, train(cfg, "candidate", label, "non_iid"),
                             config_changes={"partition": "dirichlet", "fedprox_mu": value}))
        results["non_iid"] = {"status": "completed", "rows": rows, "limitations": [EXPLORATORY,
            "Client accuracy distributions reflect each client's local evaluation observations; do not confuse local training accuracy with held-out test accuracy."]}
    else:
        results["non_iid"] = {"status": "disabled", "rows": []}

    if config.evaluation.attacks:
        emit("attacks", "Run reconstruction attacks under explicit adversary assumptions")
        results["attacks"] = run_attacks(config, progress=progress, cancelled=cancelled)
    else:
        results["attacks"] = {"status": "disabled", "rows": []}

    if config.evaluation.ablation:
        emit("ablation", "Compare active components with matched removals")
        candidate = config.candidate
        rows = [_row("Full System", train(config, "candidate", "Reuse full candidate", "ablation"))]
        operations = [operation.model_dump() for operation in candidate.algorithm] if candidate.mode == "algorithm" else []
        formula_mode = candidate.mode == "formula"
        act_active = candidate.activation_clip is not None or any(o["op"] == "ACTIVATION_CLIP" for o in operations) or (formula_mode and candidate.hook in {"H1", "H2"} and "clip(" in candidate.expression)
        gradient_active = candidate.gradient_noise > 0 or any(o["op"] == "GRADIENT_GAUSSIAN" for o in operations) or (formula_mode and candidate.hook in {"H5", "H6"} and any(word in candidate.expression for word in ("gaussian", "noise", "laplace")))
        prox_active = candidate.fedprox_mu > 0 or (formula_mode and candidate.hook == "H14" and candidate.expression.strip() not in {"0", "0.0"})
        ablations = [("minus Activation Clipping", act_active), ("minus Gradient DP", gradient_active), ("minus FedProx", prox_active)]
        for label, active in ablations:
            if not active:
                rows.append({"label": label, "status": "not_applicable", "reason": "This component is inactive in the full candidate.", "metrics": {}})
                continue
            modified = candidate.model_dump()
            if label == "minus Activation Clipping":
                # Preserve the absolute Gaussian noise scale when removing its
                # clipping bound; the resulting unbounded mechanism gets no DP claim.
                absolute_noise = candidate.activation_noise * (candidate.activation_clip or 1.)
                algorithm_clip = next((o["params"]["C_A"] for o in operations if o["op"] == "ACTIVATION_CLIP"), 1.)
                algorithm_noise = next((o["params"]["sigma_A"] for o in operations if o["op"] == "ACTIVATION_GAUSSIAN"), 0.)
                modified["activation_clip"] = None
                modified["activation_noise"] = math.hypot(absolute_noise, algorithm_noise * algorithm_clip)
                modified["algorithm"] = [o for o in operations if o["op"] not in {"ACTIVATION_CLIP", "ACTIVATION_GAUSSIAN"}]
                if formula_mode and candidate.hook in {"H1", "H2"}:
                    modified["expression"] = _strip_calls(candidate.expression, {"clip"})
            elif label == "minus Gradient DP":
                modified["gradient_clip"], modified["gradient_noise"] = None, 0
                modified["algorithm"] = [o for o in operations if o["op"] not in {"GRADIENT_CLIP", "GRADIENT_GAUSSIAN"}]
                if formula_mode and candidate.hook in {"H5", "H6"}:
                    modified["expression"] = _strip_calls(candidate.expression, {"clip", "gaussian", "noise", "laplace"})
            else:
                modified["fedprox_mu"] = 0
                modified["parameters"]["mu"] = 0
                if formula_mode and candidate.hook == "H14":
                    modified["expression"] = "0 * norm(w - w_global)**2"
            cfg = _change(config, candidate=modified)
            rows.append(_row(label, train(cfg, "candidate", label, "ablation"), config_changes={"candidate": modified}))
        split_active = formula_mode and candidate.hook == "H10"
        if split_active:
            modified = candidate.model_dump()
            modified.update(hook="H6", expression="G")
            cfg = _change(config, candidate=modified)
            rows.append(_row("minus Split Selection", train(cfg, "candidate", "Use the configured fixed split", "ablation"),
                             config_changes={"candidate": modified}))
        rows.extend([{"label": "minus Dynamic Split", "status": "not_applicable", "metrics": {}, "reason": "The main run uses a fixed configured split; optional offline profiling is separate."},
                     {"label": "minus Adaptive Noise", "status": "not_applicable", "metrics": {}, "reason": "Adaptive noise scheduling is not enabled in the implemented training engine."}])
        rows.append(_row("Baseline", train(config, "baseline", "Reuse reference baseline", "ablation")))
        results["ablation"] = {"status": "completed", "rows": rows, "limitations": [EXPLORATORY]}
    else:
        results["ablation"] = {"status": "disabled", "rows": []}

    if config.evaluation.sweep:
        from itertools import product
        parameter, values = config.evaluation.sweep_parameter, config.evaluation.sweep_values
        values = list(dict.fromkeys(values))
        secondary = config.evaluation.sweep_secondary_parameter
        secondary_values = list(dict.fromkeys(config.evaluation.sweep_secondary_values)) if secondary else [None]
        combinations = list(product(values, secondary_values))
        if config.evaluation.search == "random":
            combinations = random.Random(seed).sample(combinations, min(5, len(combinations)))
        rows = []
        for value, secondary_value in combinations:
            check_cancel()
            coordinates = {"parameter": parameter, "value": value, "secondary_parameter": secondary, "secondary_value": secondary_value}
            if not math.isfinite(value) or (secondary_value is not None and not math.isfinite(secondary_value)):
                rows.append({**coordinates, "status": "skipped", "reason": "Parameters must be finite.", "metrics": {}})
                continue
            try:
                cfg = _parameter_config(config, parameter, value)
                if secondary is not None:
                    cfg = _parameter_config(cfg, secondary, secondary_value)
                from .formulas import validate_candidate
                validation = validate_candidate(cfg.candidate)
                if not validation["valid"]:
                    raise ValueError(validation.get("error") or "The parameter value fails candidate validation.")
            except ValueError as error:
                rows.append({**coordinates, "status": "skipped", "reason": str(error), "metrics": {}})
                continue
            label = f"{parameter} = {value:g}"
            if secondary is not None:
                label += f", {secondary} = {secondary_value:g}"
            rows.append(_row(label, train(cfg, "candidate", label, "sweep"), **coordinates,
                             config_changes={"candidate": cfg.candidate.model_dump()}))
        usable = [row for row in rows if isinstance(row.get("metrics", {}).get("val_accuracy"), (float, int)) and math.isfinite(row["metrics"]["val_accuracy"])]
        selected = max(usable, key=lambda r: (r["metrics"]["val_accuracy"], -r["metrics"].get("val_loss", float("inf")))) if usable else None
        points = [{"label": row["label"], "value": row["value"], **row["metrics"]} for row in usable]
        results["sweep"] = {"status": "completed" if usable else "not_applicable", "parameter": parameter, "search": config.evaluation.search,
                            "secondary_parameter": secondary,
                            "rows": rows, "selection_metric": "val_accuracy", "tie_breaker": "val_loss",
                            "selected_value": selected["value"] if selected else None,
                            "selected_secondary_value": selected["secondary_value"] if selected else None,
                            "pareto": pareto_front(points), "privacy_pareto": pareto_front(points, {"val_accuracy": "higher", "epsilon": "lower"}),
                            "limitations": [EXPLORATORY, "Random search reproducibly samples up to five supplied parameter combinations; grid search evaluates every supplied distinct combination.",
                                            "The selected point maximizes validation accuracy; it is not automatically accepted under privacy/overhead constraints.",
                                            "Only executed parameter cells have measurements. Unrun or invalid combinations remain missing; no interpolation is used."]}
    else:
        results["sweep"] = {"status": "disabled", "rows": []}

    if config.evaluation.scalability:
        rows = []
        for axis, counts in (("clients", config.evaluation.client_counts), ("servers", config.evaluation.server_counts)):
            for count in dict.fromkeys(counts):
                check_cancel()
                if axis == "clients" and count > int(config.samples * .7):
                    rows.append({"label": f"{count} clients", "axis": axis, "count": count, "status": "skipped",
                                 "reason": "Insufficient training samples to assign at least one sample to each client.", "metrics": {}})
                    continue
                cfg = _change(config, **{axis: count})
                for variant in ("baseline", "candidate"):
                    label = f"{variant}: {count} {axis}"
                    rows.append(_row(label, train(cfg, variant, label, "scalability"), axis=axis, count=count, config_changes={axis: count}))
        results["scalability"] = {"status": "completed", "rows": rows, "limitations": [EXPLORATORY,
            "All clients and server replicas execute in one local process. Server count measures logical assignment and synchronization costs, not physical distributed speedup.",
            "Scaling changes samples per client while holding the total dataset fixed."]}
    else:
        results["scalability"] = {"status": "disabled", "rows": []}

    if config.evaluation.split_profile:
        rows = []
        for split in ([1, 6, 1], [2, 5, 1], [2, 4, 2], [3, 4, 1], [3, 3, 2]):
            overrides = {"split": split}
            if config.candidate.mode == "formula" and config.candidate.hook == "H10":
                overrides["candidate"] = {"hook": "H6", "expression": "G"}
            cfg = _change(config, **overrides)
            run = train(cfg, "candidate", f"Profile split {split}", "split_profile")
            rows.append(_row("-".join(map(str, split)), run, split=split))
        choice = select_profiled_split(rows, current_split=config.split)
        results["split_profile"] = {"status": "completed", "rows": rows, "selection": choice,
                                    "limitations": [EXPLORATORY, "Offline profile choice only; no online layer migration or minibatch split changes occur.",
                                                    "H10 split selection is disabled during the fixed-split profile to ensure each requested split is measured; its chosen policy is reported by the main run.",
                                                    "Memory constraints are evaluated only when measured memory is available; absence is not a passing memory check."]}
    else:
        results["split_profile"] = {"status": "disabled", "rows": []}
    check_cancel()
    return results
