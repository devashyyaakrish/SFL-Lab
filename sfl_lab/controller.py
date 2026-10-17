"""Run paired experiments with a fixed reference and persist evidence."""
from __future__ import annotations

import importlib.metadata
import hashlib
import platform
import subprocess
import time
from pathlib import Path

from . import __version__
from .config import ExperimentConfig


class CancelledError(RuntimeError):
    pass


def environment_info() -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    source_files = sorted(list((root / "sfl_lab").glob("*.py")) + list((root / "frontend" / "src").glob("*")) + [root / "requirements.lock", root / "pyproject.toml", root / "frontend" / "package-lock.json"])
    for path in source_files:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    versions = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "fastapi"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1], text=True))
    except (subprocess.SubprocessError, FileNotFoundError):
        commit, dirty = None, None
    return {"software_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.machine(), "packages": versions, "git_commit": commit, "git_dirty": dirty, "source_sha256": digest.hexdigest(),
            "execution": "Single-process CPU simulation with logical client/server boundaries"}


def execute_experiment(config: ExperimentConfig, action="compare", progress=None, cancelled=None) -> dict:
    from .analysis import compare_runs
    from .engine import run_training
    from .formulas import validate_candidate

    emit = progress or (lambda event: None)
    is_cancelled = cancelled or (lambda: False)
    started = time.perf_counter()
    validation = validate_candidate(config.candidate)
    if action != "baseline" and not validation["valid"]:
        raise ValueError(f"Candidate formula is invalid: {validation}")
    if action not in {"baseline", "candidate", "compare", "full"}:
        raise ValueError("Unknown experiment action")
    result = {"baseline": [], "candidate": [], "comparison": {}, "suites": {}, "validation": validation,
              "environment": environment_info(), "limitations": [
                  "Client locality is enforced by the simulated message interfaces, not by process or cryptographic isolation.",
                  "Accuracy, loss, tensor bytes and CPU duration are measured; network delays and energy are simulated estimates.",
                  "The reference is OS-USFL-inspired sample-weighted FedAvg; numerical reproduction of the original paper is not established.",
                  "Privacy accounting covers only explicitly bounded Gaussian split releases under stated assumptions; it is not a full-system DP guarantee.",
                  "Results apply only to the selected data, architecture, partitions, seeds and training budget. No cross-dataset generalization is established.",
              ]}
    variants = ["baseline", "candidate"] if action in {"compare", "full"} else [action]
    total = len(variants) * len(config.seeds)
    completed = 0
    emit({"phase": "Formula validation", "progress": 1, "message": "Candidate syntax, hook, dimensions and mathematical constraints validated."})
    # Interleave each seed's pair to reduce long-term environmental timing drift.
    for seed in config.seeds:
        for variant in variants:
            if is_cancelled():
                raise CancelledError("Experiment cancelled by researcher")
            emit({"phase": f"{variant.title()} training", "progress": 5 + 60 * completed / total,
                  "message": f"{variant.title()} seed {seed}: same initialization, partition and training configuration."})
            def training_progress(event, current=completed, v=variant, s=seed):
                emit({**event, "phase": f"{v.title()} training", "seed": s, "variant": v,
                      "progress": 5 + 60 * (current + min(1, float(event.get("round", 0)) / config.rounds)) / total})
            run = run_training(config, variant, seed, progress=training_progress, cancelled=is_cancelled)
            run["seed"] = seed
            run["variant"] = variant
            result[variant].append(run)
            completed += 1
            if is_cancelled():
                raise CancelledError("Experiment cancelled by researcher")
    if len(variants) == 2:
        emit({"phase": "Statistical validation", "progress": 66})
        result["comparison"] = compare_runs(result["baseline"], result["candidate"], config)
    if action == "full":
        from .suites import run_suites
        def suite_progress(event):
            emit({**event, "progress": max(67, min(97, float(event.get("progress", 70))))})
        result["suites"] = run_suites(config, result["baseline"], result["candidate"], progress=suite_progress, cancelled=is_cancelled)
    if is_cancelled():
        raise CancelledError("Experiment cancelled by researcher")
    result["wall_seconds"] = time.perf_counter() - started
    emit({"phase": "Final report", "progress": 99, "message": "Measured evidence and scoped conclusions saved. Export available."})
    return result
