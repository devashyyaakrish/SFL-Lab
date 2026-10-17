"""FastAPI interface for validated experiments and a durable single-worker queue."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from . import __version__
from .config import CandidateConfig, ExperimentConfig
from .controller import CancelledError, execute_experiment
from .storage import Store


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: ExperimentConfig
    action: Literal["baseline", "candidate", "compare", "full"] = "compare"


def create_app(directory=None):
    store = Store(directory)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sfl-experiment")
    cancellations: dict[str, threading.Event] = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        store.interrupt_stale()
        yield
        for flag in list(cancellations.values()):
            flag.set()
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="SFL-Lab", version=__version__, lifespan=lifespan)
    app.state.store = store

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "worker": "serial CPU", "storage": "sqlite"}

    @app.get("/api/catalog")
    def get_catalog():
        from .formulas import catalog
        return {**catalog(), "defaults": ExperimentConfig().model_dump(), "version": __version__}

    @app.post("/api/validate")
    def validate(candidate: CandidateConfig):
        from .formulas import validate_candidate
        return validate_candidate(candidate)

    @app.get("/api/experiments")
    def experiments():
        return store.list()

    @app.get("/api/experiments/{ident}")
    def get_experiment(ident: str):
        record = store.get(ident)
        if record is None:
            raise HTTPException(404, "Experiment does not exist")
        return record

    def run(ident, request, flag):
        try:
            if flag.is_set():
                raise CancelledError("Cancelled before training")
            store.update(ident, status="running", phase="Dataset preparation")
            result = execute_experiment(request.config, request.action,
                                        progress=lambda event: store.event(ident, event), cancelled=flag.is_set)
            if flag.is_set():
                raise CancelledError("Experiment cancelled")
            store.update(ident, status="completed", phase="Complete", progress=100, result=result)
        except CancelledError as exc:
            store.update(ident, status="cancelled", phase="Cancelled", error=str(exc))
        except Exception as exc:
            if flag.is_set():
                store.update(ident, status="cancelled", phase="Cancelled", error="Experiment cancelled by researcher")
            else:
                store.update(ident, status="failed", phase="Failed", error=f"{type(exc).__name__}: {exc}")
                store.event(ident, {"message": f"Run failed: {type(exc).__name__}: {exc}"})
        finally:
            with lock:
                cancellations.pop(ident, None)

    @app.post("/api/experiments", status_code=202)
    def create_experiment(request: ExperimentRequest):
        from .formulas import validate_candidate
        validation = validate_candidate(request.config.candidate)
        if request.action != "baseline" and not validation["valid"]:
            raise HTTPException(422, detail=validation)
        with lock:
            if len(cancellations) >= 8:
                raise HTTPException(429, "The local experiment queue is full; wait for a job or cancel one.")
            record = store.create(request.config.model_dump(), request.action)
            flag = threading.Event()
            cancellations[record["id"]] = flag
            executor.submit(run, record["id"], request, flag)
        return record

    @app.post("/api/experiments/{ident}/cancel")
    def cancel_experiment(ident: str):
        record = get_experiment(ident)
        with lock:
            flag = cancellations.get(ident)
            if flag is not None:
                flag.set()
        if record["status"] not in {"queued", "running"}:
            return {"id": ident, "status": record["status"], "message": "Already finished"}
        store.event(ident, {"message": "Cancellation requested. The current tensor operation will finish safely."})
        return {"id": ident, "status": "cancellation_requested"}

    @app.get("/api/experiments/{ident}/export")
    def export(ident: str, format: Literal["json", "csv", "pdf", "markdown"] = Query("json")):
        from .reports import make_report
        record = get_experiment(ident)
        if record["status"] != "completed":
            raise HTTPException(409, "Exports are available after an experiment completes.")
        data = make_report(record, format)
        types = {"json": "application/json", "csv": "text/csv", "pdf": "application/pdf", "markdown": "text/markdown"}
        extension = "md" if format == "markdown" else format
        return Response(data, media_type=types[format], headers={"Content-Disposition": f'attachment; filename="{ident}.{extension}"'})

    build = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if (build / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=build / "assets"), name="assets")

    @app.get("/")
    def index():
        if (build / "index.html").is_file():
            return FileResponse(build / "index.html")
        return {"message": "SFL-Lab API is running. Build the dashboard with: cd frontend && npm ci && npm run build", "api_docs": "/docs"}

    return app
