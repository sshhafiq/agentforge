"""SQLite-backed durable store: runs, event log, checkpoints, approvals.

The engine writes through to this store on every event, so a workflow can
survive a process restart. ``RunStore()`` with no path uses an in-memory
database (handy for tests); pass a file path for real durability.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Optional


class RunStore:
    """Thread-safe SQLite store for workflow runs."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_run
                    ON events (run_id, seq);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY,
                    step_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    context TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    decision TEXT,
                    created_at REAL NOT NULL,
                    resolved_at REAL,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                );
                """
            )

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------
    def create_run(self, run_id: str, name: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs (id, name, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, name, "running", now, now),
            )

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_run_status(self, run_id: str, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), run_id),
            )

    def list_runs(self, status: Optional[str] = None) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY created_at",
                    (status,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM runs ORDER BY created_at"
                ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def append_event(self, event: Any) -> None:
        """Persist an :class:`engine.Event` (duck-typed: run_id, type, payload, ts)."""
        with self._lock, self._conn:
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE run_id = ?",
                (event.run_id,),
            ).fetchone()[0]
            self._conn.execute(
                "INSERT INTO events (run_id, seq, type, payload, ts)"
                " VALUES (?, ?, ?, ?, ?)",
                (event.run_id, seq, event.type,
                 json.dumps(event.payload), event.ts),
            )

    def get_events(self, run_id: str) -> list[Any]:
        """Return stored events as lightweight namespaces (run_id/type/payload/ts)."""
        from types import SimpleNamespace

        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, type, payload, ts FROM events"
                " WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [
            SimpleNamespace(
                run_id=r["run_id"],
                type=r["type"],
                payload=json.loads(r["payload"]),
                ts=r["ts"],
            )
            for r in rows
        ]

    def count_events(self, run_id: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(self, run_id: str, step_index: int, state: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO checkpoints (run_id, step_index, state, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT (run_id) DO UPDATE SET"
                " step_index = excluded.step_index,"
                " state = excluded.state,"
                " updated_at = excluded.updated_at",
                (run_id, step_index, json.dumps(state), time.time()),
            )

    def load_checkpoint(self, run_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT step_index, state FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {"step_index": row["step_index"],
                "state": json.loads(row["state"])}

    # ------------------------------------------------------------------
    # approvals (human-in-the-loop)
    # ------------------------------------------------------------------
    def save_approval(
        self,
        run_id: str,
        step_name: str,
        question: str,
        context: Optional[dict] = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO approvals"
                " (run_id, step_name, question, context, status, created_at)"
                " VALUES (?, ?, ?, ?, 'pending', ?)",
                (run_id, step_name, question,
                 json.dumps(context or {}), time.time()),
            )
            return cur.lastrowid

    def pending_approvals(self, run_id: Optional[str] = None) -> list[dict]:
        with self._lock:
            if run_id:
                rows = self._conn.execute(
                    "SELECT * FROM approvals"
                    " WHERE run_id = ? AND status = 'pending'"
                    " ORDER BY created_at",
                    (run_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM approvals WHERE status = 'pending'"
                    " ORDER BY created_at"
                ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["context"] = json.loads(d["context"])
            d["decision"] = json.loads(d["decision"]) if d["decision"] else None
            result.append(d)
        return result

    def resolve_approval(self, approval_id: int, decision: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE approvals SET status = 'resolved', decision = ?,"
                " resolved_at = ? WHERE id = ?",
                (json.dumps(decision), time.time(), approval_id),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
