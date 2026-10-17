"""Evidence-only aggregation and paired experimental comparisons.

Accuracy is expressed in percent, so an accuracy subtraction is in percentage
points. Missing/unbounded privacy guarantees never become numeric zeroes.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


METRICS = {
    "test_accuracy": ("percentage points", "higher"),
    "val_accuracy": ("percentage points", "higher"),
    "train_accuracy": ("percentage points", "higher"),
    "test_loss": ("loss", "lower"),
    "val_loss": ("loss", "lower"),
    "train_loss": ("loss", "lower"),
    "runtime_seconds": ("seconds", "lower"),
    "communication_bytes": ("bytes", "lower"),
    "energy_joules": ("joules (modeled)", "lower"),
    "network_seconds": ("seconds (modeled)", "lower"),
    "aggregation_seconds": ("seconds", "lower"),
    "epsilon": ("epsilon (transcript guarantee only)", "lower"),
    "drift_mean": ("L2 distance", "lower"),
    "fairness_std": ("percentage points", "lower"),
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def _aggregate(values: list[Any], *, privacy: bool = False) -> dict:
    valid = [float(value) for value in values if _finite(value)]
    result = {"mean": None, "std": None, "median": None, "ci95": None,
              "n": len(valid), "total": len(values), "status": "unavailable"}
    if privacy and len(valid) != len(values):
        result["status"] = "unbounded_or_unavailable"
        return result
    if not valid:
        return result
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else None
    result.update(mean=mean, std=std, median=float(np.median(valid)), status="measured")
    if len(valid) >= 3 and std is not None and std > 0:
        margin = float(stats.t.ppf(.975, len(valid) - 1)) * std / math.sqrt(len(valid))
        result["ci95"] = [mean - margin, mean + margin]
    return result


def _paired(baseline: list[dict], candidate: list[dict], metric: str) -> dict:
    result = {"n_pairs": 0, "mean_difference": None, "std_difference": None,
              "ci95": None, "p_value": None, "adjusted_p_value": None,
              "effect_size": None, "effect_size_name": "paired Cohen dz",
              "significant": False, "status": "insufficient_repeats",
              "test": "two-sided paired t-test", "paired_seeds": []}
    left = {r.get("seed"): r for r in baseline if r.get("seed") is not None}
    right = {r.get("seed"): r for r in candidate if r.get("seed") is not None}
    if len(left) != len(baseline) or len(right) != len(candidate):
        result["status"] = "missing_or_duplicate_seed_identifiers"
        return result
    pairs = []
    for seed in sorted(left.keys() & right.keys()):
        a, b = left[seed].get("metrics", {}).get(metric), right[seed].get("metrics", {}).get(metric)
        if _finite(a) and _finite(b):
            pairs.append(float(b) - float(a))
            result["paired_seeds"].append(seed)
    n = len(pairs)
    result["n_pairs"] = n
    if metric == "epsilon" and (n != len(baseline) or n != len(candidate)):
        result["status"] = "incomplete_privacy_observations"
        return result
    if not n:
        result["status"] = "no_finite_paired_observations"
        return result
    mean = float(np.mean(pairs))
    result["mean_difference"] = mean
    if n < 3:
        return result
    std = float(np.std(pairs, ddof=1))
    result["std_difference"] = std
    if std <= np.finfo(float).eps * max(1, abs(mean)):
        result["status"] = "constant_paired_differences_no_inferential_test"
        return result
    sem = std / math.sqrt(n)
    margin = float(stats.t.ppf(.975, n - 1)) * sem
    p = float(2 * stats.t.sf(abs(mean / sem), df=n - 1))
    result.update(ci95=[mean - margin, mean + margin], p_value=p,
                  effect_size=mean / std, status="tested")
    return result


def pareto_front(points: list[dict], objectives: dict[str, str] | None = None) -> list[dict]:
    """Mark complete finite points only; missing guarantees cannot dominate."""
    objectives = objectives or {"val_accuracy": "higher", "communication_bytes": "lower"}
    output = [dict(point) for point in points]
    usable = [i for i, point in enumerate(output) if all(_finite(point.get(k)) for k in objectives)]
    for i, point in enumerate(output):
        point["objectives"] = objectives
        point["non_dominated"] = None if i not in usable else True
        if i not in usable:
            point["pareto_status"] = "missing_objective"
            continue
        for j in usable:
            if i == j:
                continue
            signed = [(output[j][k] - point[k]) * (1 if direction == "higher" else -1)
                      for k, direction in objectives.items()]
            if all(delta >= 0 for delta in signed) and any(delta > 0 for delta in signed):
                point["non_dominated"] = False
                break
        point["pareto_status"] = "non_dominated" if point["non_dominated"] else "dominated"
    return output


def _history(runs: list[dict]) -> list[dict]:
    rounds = sorted({row["round"] for run in runs for row in run.get("history", []) if "round" in row})
    return [{"round": round_number, **{
        metric: _aggregate([row.get(metric) for run in runs for row in run.get("history", [])
                            if row.get("round") == round_number], privacy=metric == "epsilon")
        for metric in ("train_accuracy", "val_accuracy", "train_loss", "val_loss", "epsilon")
    }} for round_number in rounds]


def _convergence(runs: list[dict], target_loss: float) -> dict:
    rows = []
    for run in runs:
        history = run.get("history", [])
        reached = next((row["round"] for row in history if _finite(row.get("val_loss")) and row["val_loss"] <= target_loss), None)
        first, last = (history[0], history[-1]) if history else ({}, {})
        span = last.get("round", 0) - first.get("round", 0)
        slope = ((last["val_loss"] - first["val_loss"]) / span
                 if span > 0 and _finite(last.get("val_loss")) and _finite(first.get("val_loss")) else None)
        rows.append({"seed": run.get("seed"), "round_to_target_loss": reached,
                     "target_reached": reached is not None, "validation_loss_slope_per_round": slope})
    return {"target_loss": target_loss, "selection_metric": "val_loss", "runs": rows,
            "note": "Unreached target rounds are censored, not assigned a successful convergence time."}


def compare_runs(baseline: list[dict], candidate: list[dict], config: Any) -> dict:
    """Compare paired repeats, retaining uncertainty and the validation/test split."""
    aggregates = {name: {metric: _aggregate([r.get("metrics", {}).get(metric) for r in runs], privacy=metric == "epsilon")
                         for metric in METRICS}
                  for name, runs in (("baseline", baseline), ("candidate", candidate))}
    deltas, statistics = {}, {}
    for metric, (unit, direction) in METRICS.items():
        a, b = aggregates["baseline"][metric]["mean"], aggregates["candidate"][metric]["mean"]
        absolute = b - a if a is not None and b is not None else None
        relative = 100 * absolute / abs(a) if absolute is not None and a != 0 else None
        deltas[metric] = {"absolute": absolute, "relative_pct": relative,
                          "improvement_pct": relative * (1 if direction == "higher" else -1) if relative is not None else None,
                          "unit": unit, "direction": direction,
                          "status": "measured" if absolute is not None else "unavailable_or_unbounded"}
        statistics[metric] = _paired(baseline, candidate, metric)
    # Validation accuracy is the prespecified primary test. Holm correction for
    # all reported metric tests prevents opportunistic significance claims.
    tested = sorted([(result["p_value"], metric) for metric, result in statistics.items()
                     if result["p_value"] is not None])
    running = 0.0
    for rank, (p, metric) in enumerate(tested):
        running = max(running, min(1.0, p * (len(tested) - rank)))
        statistics[metric]["adjusted_p_value"] = running
        statistics[metric]["significant"] = running < .05
    limits = config.acceptance
    criteria = []
    accuracy = deltas["val_accuracy"]["absolute"]
    criteria.append({"name": "Validation accuracy", "status": "inconclusive" if accuracy is None else
                     "pass" if accuracy >= -limits.max_accuracy_drop_pp else "fail",
                     "observed": accuracy, "threshold": -limits.max_accuracy_drop_pp, "unit": "percentage points",
                     "detail": "Primary acceptance metric; candidate minus baseline validation accuracy."})
    for metric, name, threshold in (("communication_bytes", "Communication overhead", limits.max_communication_increase_pct),
                                    ("runtime_seconds", "Runtime overhead", limits.max_runtime_increase_pct)):
        observed = deltas[metric]["relative_pct"]
        criteria.append({"name": name, "status": "inconclusive" if observed is None else "pass" if observed <= threshold else "fail",
                         "observed": observed, "threshold": threshold, "unit": "% relative increase",
                         "detail": "Observed mean overhead; runtime is sensitive to execution load."})
    privacy_improvement = deltas["epsilon"]["improvement_pct"]
    if limits.required_privacy_improvement_pct > 0:
        criteria.append({"name": "Privacy improvement", "status": "inconclusive" if privacy_improvement is None else
                         "pass" if privacy_improvement >= limits.required_privacy_improvement_pct else "fail",
                         "observed": privacy_improvement, "threshold": limits.required_privacy_improvement_pct,
                         "unit": "% relative epsilon reduction", "detail": "A percentage reduction from unbounded/unknown epsilon is undefined. Accounting applies to protected transcript only."})
    else:
        criteria.append({"name": "Privacy improvement", "status": "not_required", "observed": privacy_improvement,
                         "threshold": 0, "detail": "No privacy improvement requirement was configured; this is not evidence of privacy."})
    protected_releases = any(run.get("privacy", {}).get("protected_releases", 0) > 0 for run in candidate)
    if protected_releases or limits.required_privacy_improvement_pct > 0:
        epsilons = [run.get("metrics", {}).get("epsilon") for run in candidate]
        finite_budget = bool(epsilons) and all(_finite(value) for value in epsilons)
        exceeds_budget = any(_finite(value) and value >= config.privacy.epsilon_max for value in epsilons)
        criteria.append({"name": "Privacy budget", "status": "fail" if exceeds_budget else "pass" if finite_budget else "inconclusive",
                         "observed": max(epsilons) if finite_budget else None, "threshold": config.privacy.epsilon_max,
                         "unit": "maximum transcript epsilon across candidate seeds",
                         "detail": "Every candidate seed must stay below the configured transcript budget; unprotected channels do not have a finite bound. This does not certify end-to-end privacy."})
    compliance = [check for run in baseline + candidate for check in run.get("compliance", [])]
    compliance_failed = any(str(c.get("status", "")).lower() in {"fail", "failed", "false"} for c in compliance)
    compliance_passed = bool(baseline and candidate) and all(run.get("compliance") for run in baseline + candidate) and all(str(c.get("status", "")).lower() in {"pass", "passed", "true"} for c in compliance)
    criteria.append({"name": "Core SFL architecture", "status": "fail" if compliance_failed else "pass" if compliance_passed else "inconclusive",
                     "detail": "Checks recorded by actual training runs; logical locality does not imply process or cryptographic isolation."})
    expected_seeds = set(config.seeds)
    actual_seeds = [[run.get("seed") for run in runs] for runs in (baseline, candidate)]
    complete_pairs = all(len(seeds) == len(expected_seeds) and set(seeds) == expected_seeds for seeds in actual_seeds)
    pairing_issues = []
    if not complete_pairs:
        pairing_issues.append("Both variants must contain exactly one run for every configured seed.")
    else:
        right = {run["seed"]: run for run in candidate}
        for left in baseline:
            match = right[left["seed"]]
            for field, first, second in (
                ("dataset split hashes", left.get("dataset", {}).get("split_hashes"), match.get("dataset", {}).get("split_hashes")),
                ("model initialization", left.get("initial_state_hash"), match.get("initial_state_hash")),
            ):
                if first is None or second is None:
                    pairing_issues.append(f"Seed {left['seed']}: {field} provenance unavailable.")
                elif first != second:
                    pairing_issues.append(f"Seed {left['seed']}: {field} differs between variants.")
    required_metrics = ["val_accuracy", "communication_bytes", "runtime_seconds"]
    if limits.required_privacy_improvement_pct > 0:
        required_metrics.append("epsilon")
    incomplete = [f"{variant}.{metric}" for variant in ("baseline", "candidate") for metric in required_metrics
                  if aggregates[variant][metric]["n"] != len(expected_seeds)]
    if incomplete:
        pairing_issues.append("Incomplete finite acceptance observations: " + ", ".join(incomplete) + ".")
    criteria.append({"name": "Paired experimental conditions", "status": "inconclusive" if pairing_issues else "pass",
                     "detail": " ".join(pairing_issues) if pairing_issues else "Every configured seed has matched dataset splits and model initialization, with finite required observations."})
    n_pairs = statistics["val_accuracy"]["n_pairs"]
    criteria.append({"name": "Repeated paired seeds", "status": "pass" if n_pairs >= 3 else "inconclusive",
                     "observed": n_pairs, "threshold": 3, "detail": "At least three paired repeats are required for an overall conclusion; more are recommended."})
    relevant = [c for c in criteria if c["status"] != "not_required"]
    failed = [c["name"] for c in relevant if c["status"] == "fail"]
    missing = [c["name"] for c in relevant if c["status"] == "inconclusive"]
    if pairing_issues and not compliance_failed:
        hypothesis_status, decision = "inconclusive", "INCONCLUSIVE"
    elif compliance_failed or "Validation accuracy" in failed or "Privacy budget" in failed:
        hypothesis_status, decision = "not_supported", "REJECT"
    elif missing:
        hypothesis_status, decision = "inconclusive", "INCONCLUSIVE"
    elif failed:
        hypothesis_status, decision = "partially_supported", "CONDITIONALLY ACCEPT"
    else:
        hypothesis_status, decision = "supported", "ACCEPT"
    reasons = ([f"Failed acceptance criterion: {name}." for name in failed]
               + [f"Insufficient evidence: {name}." for name in missing])
    if not reasons:
        reasons = ["All configured acceptance criteria pass on measured repeated-run means."]
    limitations = [
        "Recommendations assess the configured numeric acceptance criteria, not unrestricted natural-language claims in the hypothesis.",
        "Mean-based acceptance does not establish statistical significance, equivalence, causality beyond controlled changes, or population generalization.",
        "Confidence intervals and paired t-tests assume independent seeds and approximately normal paired differences; small samples limit power.",
        "P-values use Holm correction over available metric tests. Constant differences and fewer than three pairs have no inferential test.",
        "Validation selects checkpoints and parameters. Final test results are descriptive held-out measurements; repeated exploratory suite tests require a new untouched test set for confirmation.",
        "Runtime and tensor bytes are measured locally. Network latency and energy are simulation estimates, not hardware observations.",
        "Privacy epsilon describes only accountant-covered messages under its stated assumptions; it does not certify the entire trained system or model.",
        "Results apply only to the tested dataset, architecture, sample size, partitions, seeds, and resource conditions.",
    ]
    if n_pairs < 5:
        limitations.append(f"Only {n_pairs} paired seeds were available; use at least five and larger data/training budgets for research conclusions.")
    points = [{"label": name, **{metric: aggregates[name][metric]["mean"] for metric in METRICS}}
              for name in ("baseline", "candidate")]
    return {**aggregates, "deltas": deltas, "statistics": statistics,
            "primary_metric": "val_accuracy", "hypothesis": config.hypothesis,
            "hypothesis_status": hypothesis_status, "recommendation": {"decision": decision, "reasons": reasons},
            "criteria": criteria, "limitations": limitations,
            "pareto": pareto_front(points),
            "privacy_pareto": pareto_front(points, {"val_accuracy": "higher", "epsilon": "lower"}),
            "learning_curves": {"baseline": _history(baseline), "candidate": _history(candidate)},
            "convergence": {"baseline": _convergence(baseline, config.target_loss), "candidate": _convergence(candidate, config.target_loss)},
            "evidence": {"baseline_seeds": [r.get("seed") for r in baseline], "candidate_seeds": [r.get("seed") for r in candidate],
                         "primary_test": "Paired validation-accuracy t-test", "multiple_testing": "Holm over reported metrics",
                         "pairing_issues": pairing_issues,
                         "acceptance_basis": "Observed repeated-run means; significance is reported separately."}}
