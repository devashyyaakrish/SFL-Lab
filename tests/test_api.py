"""Exercise the real queue, tensor training, exports and durable records."""
from __future__ import annotations

import csv
import io
import json
import math
import time

import pytest
from fastapi.testclient import TestClient

from sfl_lab.api import create_app
from sfl_lab.config import ExperimentConfig
from sfl_lab.storage import Store


def small_config(**changes):
    return ExperimentConfig(
        model="tiny_cnn", samples=80, clients=2, rounds=1,
        seeds=[1, 2, 3], **changes,
    ).model_dump()


def wait_finished(client, ident, timeout=60):
    deadline = time.monotonic() + timeout
    record = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/experiments/{ident}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(.05)
    pytest.fail(f"Experiment did not finish within {timeout}s: {record}")


def test_reference_can_run_independently_of_invalid_candidate_formula(tmp_path):
    config = small_config()
    config["candidate"]["expression"] = "unsupported_variable"
    config["seeds"] = [1]
    with TestClient(create_app(tmp_path)) as client:
        rejected = client.post("/api/experiments", json={"config": config, "action": "candidate"})
        assert rejected.status_code == 422
        submitted = client.post("/api/experiments", json={"config": config, "action": "baseline"})
        assert submitted.status_code == 202
        record = wait_finished(client, submitted.json()["id"])
        assert record["status"] == "completed", record["error"]
        assert record["result"]["baseline"] and not record["result"]["candidate"]
        assert record["result"]["validation"]["valid"] is False


def test_real_paired_run_and_all_exports(tmp_path):
    directory = tmp_path / "runs"
    with TestClient(create_app(directory)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        catalog = client.get("/api/catalog")
        assert catalog.status_code == 200
        assert catalog.json()["defaults"]["model"] == "resnet18"
        response = client.post("/api/experiments", json={"config": small_config(), "action": "compare"})
        assert response.status_code == 202, response.text
        ident = response.json()["id"]
        record = wait_finished(client, ident)
        assert record["status"] == "completed", record["error"]
        assert record["progress"] == 100
        assert record["logs"] and record["events"]
        result = record["result"]
        assert result["validation"]["valid"]
        assert len(result["baseline"]) == len(result["candidate"]) == 3
        for baseline, candidate in zip(result["baseline"], result["candidate"]):
            assert baseline["seed"] == candidate["seed"]
            assert baseline["initial_state_hash"] == candidate["initial_state_hash"]
            assert baseline["client_selection"] == candidate["client_selection"]
            assert baseline["dataset"]["split_hashes"] == candidate["dataset"]["split_hashes"]
            assert baseline["dataset"]["client_sizes"] == candidate["dataset"]["client_sizes"]
            for run in (baseline, candidate):
                metrics = run["metrics"]
                for key in ("train_loss", "val_loss", "test_loss", "runtime_seconds", "communication_bytes"):
                    assert math.isfinite(metrics[key]) and metrics[key] > 0
                for key in ("train_accuracy", "val_accuracy", "test_accuracy"):
                    assert 0 <= metrics[key] <= 100
                assert metrics["communication_bytes"] == sum(run["communication"].values())
                assert len(run["history"]) == 1
                assert len(run["clients"]) == 2
                assert run["completed_steps"] > 0
                assert run["initial_state_hash"] != run["final_state_hash"]
                assert run["test_evaluations"] == 1
                assert run["compliance"]
                assert all(check["status"].lower() == "pass" for check in run["compliance"])
                indices = run["dataset"]["indices"]
                train, validation, test = (set(indices[key]) for key in ("train", "validation", "test"))
                assert not train & validation and not train & test and not validation & test
                assert len(train | validation | test) == 80
        assert result["comparison"]["statistics"]["val_accuracy"]["n_pairs"] == 3
        assert result["limitations"]
        # JSON serialization rejects any accidental NaN or infinity in nested results.
        json.dumps(result, allow_nan=False)
        exported_json = client.get(f"/api/experiments/{ident}/export?format=json")
        assert exported_json.status_code == 200
        assert exported_json.json()["id"] == ident
        exported_csv = client.get(f"/api/experiments/{ident}/export?format=csv")
        assert exported_csv.status_code == 200
        rows = list(csv.reader(io.StringIO(exported_csv.text)))
        assert len(rows) > 6 and any("baseline" in row for row in rows)
        exported_pdf = client.get(f"/api/experiments/{ident}/export?format=pdf")
        assert exported_pdf.status_code == 200
        assert exported_pdf.content.startswith(b"%PDF-")
        assert exported_pdf.content.rstrip().endswith(b"%%EOF")
        exported_markdown = client.get(f"/api/experiments/{ident}/export?format=markdown")
        assert exported_markdown.status_code == 200
        assert len(exported_markdown.content) > 100
        assert ident in exported_pdf.headers["content-disposition"]
        listed = client.get("/api/experiments").json()
        assert listed[0]["id"] == ident and listed[0]["status"] == "completed"
        assert "result" not in listed[0]
        assert client.post(f"/api/experiments/{ident}/cancel").json()["status"] == "completed"
    # Completed evidence survives application and Store recreation.
    with TestClient(create_app(directory)) as restarted:
        restored = restarted.get(f"/api/experiments/{ident}").json()
        assert restored["result"] == result
        assert restored["status"] == "completed"


def test_validation_errors_and_missing_resources(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/experiments/DOES-NOT-EXIST").status_code == 404
        assert client.post("/api/experiments/DOES-NOT-EXIST/cancel").status_code == 404
        config = small_config()
        config["split"] = [1, 1, 1]
        assert client.post("/api/experiments", json={"config": config}).status_code == 422
        config = small_config()
        config["candidate"]["expression"] = "__import__('os').system('echo forbidden')"
        validation = client.post("/api/validate", json=config["candidate"])
        assert validation.status_code == 200
        assert not validation.json()["valid"]
        rejected = client.post("/api/experiments", json={"config": config})
        assert rejected.status_code == 422
        assert client.get("/api/experiments").json() == []
        missing_data = small_config(dataset="imagefolder", dataset_path=str(tmp_path / "missing-dataset"))
        response = client.post("/api/experiments", json={"config": missing_data, "action": "baseline"})
        assert response.status_code == 202
        ident = response.json()["id"]
        record = wait_finished(client, ident)
        assert record["status"] == "failed"
        assert "dataset_path" in record["error"]
        assert record["result"] is None
        assert client.get(f"/api/experiments/{ident}/export").status_code == 409
        assert client.get(f"/api/experiments/{ident}/export?format=unsupported").status_code == 422


def test_queued_cancellation_runs_no_training(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        long_config = small_config()
        long_config.update(rounds=1000, seeds=[7])
        first = client.post("/api/experiments", json={"config": long_config, "action": "baseline"})
        assert first.status_code == 202
        first_id = first.json()["id"]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            running = client.get(f"/api/experiments/{first_id}").json()
            if any(event.get("round", 0) >= 1 for event in running["events"]):
                break
            assert running["status"] not in {"failed", "cancelled", "completed"}, running["error"]
            time.sleep(.05)
        else:
            pytest.fail("The cancellation test never reached an actual completed training round")
        # The first real job occupies the sole worker; the second remains queued.
        second = client.post("/api/experiments", json={"config": small_config(), "action": "compare"})
        assert second.status_code == 202
        second_id = second.json()["id"]
        assert client.get(f"/api/experiments/{second_id}").json()["status"] == "queued"
        assert client.get(f"/api/experiments/{second_id}/export").status_code == 409
        assert client.post(f"/api/experiments/{second_id}/cancel").json()["status"] == "cancellation_requested"
        assert client.post(f"/api/experiments/{first_id}/cancel").json()["status"] == "cancellation_requested"
        assert wait_finished(client, first_id)["status"] == "cancelled"
        cancelled = wait_finished(client, second_id)
        assert cancelled["status"] == "cancelled"
        assert cancelled["result"] is None
        assert not any("training" in event.get("phase", "").lower() for event in cancelled["events"])


def test_restart_marks_only_unfinished_records_interrupted(tmp_path):
    store = Store(tmp_path)
    done = store.create(small_config(), "compare")
    store.update(done["id"], status="completed", result={"proof": "durable"}, progress=100)
    queued = store.create(small_config(), "compare")
    running = store.create(small_config(), "baseline")
    store.update(running["id"], status="running")
    store.event(running["id"], {"phase": "Baseline training", "message": "Durable event", "progress": 20})
    store.event(running["id"], {"progress": 10})
    assert Store(tmp_path).get(running["id"])["progress"] == 20
    with TestClient(create_app(tmp_path)) as client:
        for ident in (queued["id"], running["id"]):
            record = client.get(f"/api/experiments/{ident}").json()
            assert record["status"] == "failed"
            assert record["phase"] == "Interrupted"
            assert "restarted" in record["error"]
        restored = client.get(f"/api/experiments/{running['id']}").json()
        assert any("Durable event" in line for line in restored["logs"])
        completed = client.get(f"/api/experiments/{done['id']}").json()
        assert completed["status"] == "completed"
        assert completed["result"] == {"proof": "durable"}


def test_store_rejects_nonfinite_results_atomically(tmp_path):
    store = Store(tmp_path)
    record = store.create(small_config(), "baseline")
    with pytest.raises(ValueError):
        store.update(record["id"], result={"invalid": float("nan")})
    assert store.get(record["id"])["result"] is None
    with pytest.raises(KeyError):
        store.update("missing", status="failed")
