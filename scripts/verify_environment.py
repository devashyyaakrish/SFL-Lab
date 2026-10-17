#!/usr/bin/env python3
"""Run bounded, real API experiments and save a machine-readable readiness report.

Run after dependency installation: .venv/bin/python scripts/verify_environment.py
Use --base-url http://127.0.0.1:8000 to check an already running server. Without
that option the same FastAPI application and real worker run via TestClient.
The small budgets establish functionality, not a published research result.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from sfl_lab.api import create_app
from sfl_lab.config import CandidateConfig, EvaluationConfig, ExperimentConfig
from sfl_lab.controller import environment_info


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


@contextmanager
def api_client(base_url, directory):
    if base_url:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=30) as client:
            yield client
    else:
        with TestClient(create_app(directory)) as client:
            yield client


def submit_and_wait(client, config, action, timeout):
    response = client.post("/api/experiments", json={"config": config.model_dump(), "action": action})
    require(response.status_code == 202, f"Submission failed: {response.status_code} {response.text}")
    ident = response.json()["id"]
    deadline, last_phase = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        response = client.get(f"/api/experiments/{ident}")
        require(response.status_code == 200, f"Could not read experiment {ident}")
        record = response.json()
        phase = record["phase"]
        if phase != last_phase:
            print(f"{ident}: {record['status']} / {phase}", flush=True)
            last_phase = phase
        if record["status"] in {"completed", "failed", "cancelled"}:
            require(record["status"] == "completed", f"{ident}: {record['status']}: {record['error']}")
            require(record["progress"] == 100, f"{ident} completed without full progress")
            return record
        time.sleep(.2)
    client.post(f"/api/experiments/{ident}/cancel")
    raise VerificationError(f"{ident} exceeded {timeout}s; cancellation was requested")


def verify_measurements(run, expected_clients=None):
    metrics = run["metrics"]
    for key in ("train_loss", "val_loss", "test_loss", "runtime_seconds", "communication_bytes"):
        require(isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key]) and metrics[key] > 0,
                f"Invalid measured {key}: {metrics.get(key)}")
    for key in ("train_accuracy", "val_accuracy", "test_accuracy"):
        require(isinstance(metrics.get(key), (int, float)) and 0 <= metrics[key] <= 100, f"Invalid {key}")
    require(run["communication"] and all(value > 0 for value in run["communication"].values()), "Missing message traffic")
    require(sum(run["communication"].values()) == metrics["communication_bytes"], "Message byte totals disagree")
    if expected_clients is not None:
        require(len(run["clients"]) == expected_clients, "Client count mismatch")
    dataset = run["dataset"]
    split_sets = [set(dataset["indices"][name]) for name in ("train", "validation", "test")]
    require(not any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i)), "Dataset split overlap")
    require(sum(map(len, split_sets)) == dataset["samples"], "Dataset samples were lost between splits")
    require(sum(dataset["client_sizes"]) == dataset["train"], "Client partitions lost or duplicated samples")
    require(all(count > 0 for count in dataset["client_sizes"]), "Empty client partition")
    require(len(run["clients"]) == len(dataset["client_sizes"]), "Client observations omit a partition")
    for observation in run["clients"]:
        require(math.isfinite(observation["loss"]) and observation["loss"] > 0, "Invalid client loss")
        require(0 <= observation["accuracy"] <= 100, "Invalid client accuracy")


def verify_run(run, expected_clients):
    verify_measurements(run, expected_clients)
    require(run["completed_steps"] > 0, "No optimization step completed")
    require(run["initial_state_hash"] != run["final_state_hash"], "Training did not update model state")
    require(run["test_evaluations"] == 1, "Held-out test was evaluated more than once")
    require(run["compliance"] and all(c["status"].lower() == "pass" for c in run["compliance"]), "SFL compliance check failed")


def verify_pair(record, config):
    result = record["result"]
    json.dumps(result, allow_nan=False)
    require(result["validation"]["valid"], "Saved candidate validation failed")
    require(len(result["baseline"]) == len(result["candidate"]) == len(config.seeds), "Missing paired seed")
    for baseline, candidate in zip(result["baseline"], result["candidate"]):
        verify_run(baseline, config.clients)
        verify_run(candidate, config.clients)
        require(baseline["seed"] == candidate["seed"], "Seed pairing mismatch")
        require(baseline["initial_state_hash"] == candidate["initial_state_hash"], "Paired models start from different parameters")
        require(baseline["dataset"]["split_hashes"] == candidate["dataset"]["split_hashes"], "Paired dataset split mismatch")
        require(baseline["dataset"]["client_sizes"] == candidate["dataset"]["client_sizes"], "Paired partitions mismatch")
        require(baseline["client_selection"] == candidate["client_selection"], "Paired client selection mismatch")
    require(result["comparison"]["statistics"]["val_accuracy"]["n_pairs"] == len(config.seeds), "Statistics omit paired seeds")
    return {
        "id": record["id"], "status": record["status"], "model": config.model,
        "samples": config.samples, "seeds": config.seeds, "clients": config.clients,
        "rounds": config.rounds, "wall_seconds": result["wall_seconds"],
        "baseline": result["comparison"]["baseline"], "candidate": result["comparison"]["candidate"],
        "hypothesis_status": result["comparison"]["hypothesis_status"],
        "recommendation": result["comparison"]["recommendation"],
        "invariants": ["finite measured losses", "disjoint train/validation/test splits", "identical paired initialization",
                       "identical paired partitions and client selection", "actual tensor updates", "measured message bytes",
                       "both cut boundaries detached", "client-side labels and raw data", "validation checkpoint selection",
                       "one final held-out test evaluation", "durable experiment record"],
    }


def verify_suites(record, config):
    suites = record["result"]["suites"]
    expected = ("network", "non_iid", "attacks", "ablation", "sweep", "scalability", "split_profile")
    for name in expected:
        require(name in suites, f"Missing evaluation suite: {name}")
        require(suites[name].get("status") == "completed", f"Evaluation suite {name} did not complete: {suites[name].get('status')}")
    require(len(suites["network"]["rows"]) == 8, "Network scenario replay is incomplete")
    require(len(suites["non_iid"]["rows"]) >= 4, "Missing IID/non-IID/FedProx comparisons")
    require(suites["sweep"]["selection_metric"] == "val_accuracy", "Sweep selection used a non-validation metric")
    require(len(suites["sweep"]["rows"]) == 6 and suites["sweep"]["secondary_parameter"] == "sigma_G", "Two-dimensional sweep did not execute all six parameter cells")
    require(len(suites["scalability"]["rows"]) == 8, "Missing client/server count experiments")
    require(len(suites["split_profile"]["rows"]) == 5, "Missing split profiles")
    for name in ("non_iid", "ablation", "sweep", "scalability", "split_profile"):
        for row in suites[name]["rows"]:
            if row.get("status") == "not_applicable":
                require(bool(row.get("reason")), f"Unexplained omitted {name} condition")
                continue
            require(row.get("source") == "measured_training", f"{name} contains an unmeasured result")
            verify_measurements(row)
    ablations = {row["label"]: row for row in suites["ablation"]["rows"]}
    for label in ("minus Activation Clipping", "minus Gradient DP", "minus FedProx"):
        require(ablations[label].get("source") == "measured_training", f"Active component {label} was not ablated")
    attacks = suites["attacks"]
    require(len(attacks["rows"]) == 5, "Missing attack controls")
    for row in attacks["rows"]:
        for kind in ("feature", "gradient"):
            probe = row[kind]
            require(probe["steps"] == config.evaluation.attack_steps, "Attack optimization budget was not executed")
            require(0 <= probe["mse"] <= 1 and -1 <= probe["ssim"] <= 1, "Invalid attack image metrics")
            require(probe["final_objective"] <= probe["initial_objective"], "Attack lost its best reconstruction")
            image = base64.b64decode(probe["reconstruction_png"].split(",", 1)[1], validate=True)
            require(image.startswith(b"\x89PNG\r\n\x1a\n"), "Attack reconstruction is not a PNG image")
    for run in record["result"]["candidate"]:
        epsilon = run["metrics"]["epsilon"]
        require(isinstance(epsilon, (float, int)) and 0 < epsilon <= config.privacy.epsilon_max,
                f"Protected transcript exceeded the verification budget: {epsilon}")
        require(run["privacy"]["end_to_end_dp"] is False, "Transcript accounting was misreported as full-system DP")
    require(all(run["metrics"]["epsilon"] is None for run in record["result"]["baseline"]),
            "Unprotected baseline must not claim a finite privacy bound")
    return {name: {"status": suites[name]["status"], "rows": len(suites[name].get("rows", [])),
                   "limitations": suites[name].get("limitations", [])} for name in expected}


def verify_exports(client, record, output_directory):
    exports = {}
    for format in ("json", "csv", "pdf", "markdown"):
        response = client.get(f"/api/experiments/{record['id']}/export", params={"format": format})
        require(response.status_code == 200, f"{format} export failed: {response.text[:200]}")
        data = response.content
        if format == "json":
            require(response.json()["id"] == record["id"], "JSON export contains the wrong experiment")
        elif format == "csv":
            require(len(list(csv.reader(io.StringIO(response.text)))) > 3, "CSV report is empty")
        elif format == "pdf":
            require(data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF"), "PDF report is malformed")
        else:
            require(len(data) > 100, "Markdown report is empty")
        suffix = "md" if format == "markdown" else format
        path = output_directory / f"{record['id']}.{suffix}"
        path.write_bytes(data)
        exports[format] = {"path": str(path), "bytes": len(data)}
    return exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Use a running server instead of an in-process API")
    parser.add_argument("--output", type=Path, default=Path("artifacts/verification.json"))
    parser.add_argument("--timeout", type=float, default=600, help="Maximum seconds per experiment")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports = args.output.parent / "verification-reports"
    reports.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
        "transport": args.base_url or "FastAPI TestClient with real serial training worker",
        "experiments": [], "capabilities": {},
        "scope": "Bounded CPU functional verification. These small data/training budgets do not establish convergence, publication-quality accuracy, full-system DP, or distributed production readiness.",
    }
    try:
        with api_client(args.base_url, args.output.parent / "verification-store") as client:
            health = client.get("/api/health")
            require(health.status_code == 200 and health.json()["status"] == "ok", "API is unhealthy")
            report["capabilities"]["api"] = "passed"
            candidate = CandidateConfig(
                name="Verification: clipping, Gaussian protection and FedProx",
                expression="clip(G, C_G)", parameters={"C_G": 1.0},
                activation_clip=1.0, activation_noise=10.0, gradient_clip=1.0, gradient_noise=10.0, fedprox_mu=.01,
            )
            evaluation = EvaluationConfig(attack_steps=3, scalability=True, split_profile=True,
                                          client_counts=[2, 3], server_counts=[1, 2], sweep_values=[.5, 1, 2],
                                          sweep_secondary_parameter="sigma_G", sweep_secondary_values=[0, 10])
            tiny = ExperimentConfig(name="Environment verification: complete tiny CNN evaluation", model="tiny_cnn",
                                    samples=100, clients=2, rounds=1, seeds=[1, 2, 3],
                                    candidate=candidate, evaluation=evaluation)
            record = submit_and_wait(client, tiny, "full", args.timeout)
            evidence = verify_pair(record, tiny)
            evidence["suites"] = verify_suites(record, tiny)
            evidence["exports"] = verify_exports(client, record, reports)
            report["experiments"].append(evidence)
            report["capabilities"].update({"paired_training": "passed", "full_evaluation": "passed",
                                           "json_csv_pdf_markdown_exports": "passed"})
            resnet = ExperimentConfig(name="Environment verification: ResNet18", model="resnet18",
                                      samples=40, clients=2, rounds=1, seeds=[1])
            record = submit_and_wait(client, resnet, "compare", args.timeout)
            evidence = verify_pair(record, resnet)
            evidence["exports"] = verify_exports(client, record, reports)
            report["experiments"].append(evidence)
            report["capabilities"]["resnet18_training"] = "passed"
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Verification {report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
