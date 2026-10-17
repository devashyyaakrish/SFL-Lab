"""Small durable experiment store; each update is committed atomically."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False)


class Store:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or os.environ.get("SFL_LAB_DATA_DIR", ROOT / "artifacts"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "experiments.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS experiments (sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, payload TEXT NOT NULL)")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def create(self, config: dict, action: str) -> dict:
        with self.connect() as db:
            cursor = db.execute("INSERT INTO experiments(payload) VALUES ('{}')")
            ident = f"EXP-{datetime.now(timezone.utc).year}-{cursor.lastrowid:05d}"
            record = {"id": ident, "name": config["name"], "status": "queued", "phase": "Queued", "progress": 0,
                      "config": config, "action": action, "created_at": now(), "updated_at": now(),
                      "error": None, "result": None, "logs": [], "events": []}
            db.execute("UPDATE experiments SET id=?, payload=? WHERE sequence=?", (ident, encode(record), cursor.lastrowid))
        return record

    def get(self, ident: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM experiments WHERE id=?", (ident,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self, limit=100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM experiments ORDER BY sequence DESC LIMIT ?", (limit,)).fetchall()
        records = [json.loads(r["payload"]) for r in rows]
        for record in records:
            result = record.pop("result", None)
            record["summary"] = result.get("comparison", {}) if result else None
            record.pop("logs", None)
            record.pop("events", None)
        return records

    def update(self, ident: str, **changes) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM experiments WHERE id=?", (ident,)).fetchone()
            if row is None:
                raise KeyError(ident)
            record = json.loads(row["payload"])
            record.update(changes, updated_at=now())
            db.execute("UPDATE experiments SET payload=? WHERE id=?", (encode(record), ident))
        return record

    def event(self, ident: str, event: dict):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM experiments WHERE id=?", (ident,)).fetchone()
            if row is None:
                return
            record = json.loads(row["payload"])
            stamped = {"time": now(), **event}
            record["events"] = (record.get("events", []) + [stamped])[-2000:]
            message = event.get("message") or event.get("phase")
            if message:
                record["logs"] = (record.get("logs", []) + [f"{stamped['time']} {message}"])[-1000:]
            if event.get("phase"):
                record["phase"] = event["phase"]
            if isinstance(event.get("progress"), (float, int)):
                record["progress"] = max(record["progress"], min(99, event["progress"]))
            record["updated_at"] = now()
            db.execute("UPDATE experiments SET payload=? WHERE id=?", (encode(record), ident))

    def interrupt_stale(self):
        for record in self.list(limit=10000):
            if record["status"] in ("queued", "running"):
                self.update(record["id"], status="failed", phase="Interrupted", error="The application restarted before this job completed. Submit a new run; partial records are retained.")
