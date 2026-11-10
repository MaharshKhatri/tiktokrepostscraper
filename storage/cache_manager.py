"""SQLite-backed checkpoint store.

Every expensive step (pagination cursors, downloads, transcripts, vision
batches, digests, report sections) writes its result here keyed by a
deterministic key, so an interrupted run can simply be started again and will
skip everything that already finished.

The store is a namespaced key/value table holding JSON documents, plus a small
``stages`` table that records which coarse pipeline stages completed for a run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE TABLE IF NOT EXISTS stages (
    run_key    TEXT NOT NULL,
    stage      TEXT NOT NULL,
    detail     TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (run_key, stage)
);
"""


def _like_prefix(prefix: str) -> str:
    """Escape LIKE wildcards so handles such as ``foo_bar`` match literally."""

    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def stable_hash(*parts: Any) -> str:
    """Deterministic short hash of JSON-serializable parts (sorted keys)."""

    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def file_fingerprint(path: str | Path) -> str:
    """Cheap identity for a local file: path + size + mtime."""

    p = Path(path)
    try:
        stat = p.stat()
    except FileNotFoundError:
        return f"missing:{p}"
    return f"{p.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"


class CacheManager:
    """Thread-safe JSON key/value store on top of a single SQLite file."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------------------------------------------------------- basics
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> CacheManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- key/value
    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE namespace = ? AND key = ?", (namespace, key)
            ).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, namespace: str, key: str, value: Any) -> None:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        payload = json.dumps(value, default=str, ensure_ascii=False)
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO kv(namespace, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (namespace, key, payload, time.time()),
            )

    def has(self, namespace: str, key: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM kv WHERE namespace = ? AND key = ?", (namespace, key)).fetchone()
        return row is not None

    def delete(self, namespace: str, key: str | None = None) -> int:
        with self._tx() as conn:
            if key is None:
                cur = conn.execute("DELETE FROM kv WHERE namespace = ?", (namespace,))
            else:
                cur = conn.execute("DELETE FROM kv WHERE namespace = ? AND key = ?", (namespace, key))
            return cur.rowcount

    def keys(self, namespace: str, prefix: str = "") -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key FROM kv WHERE namespace = ? AND key LIKE ? ESCAPE '\\' ORDER BY key",
                (namespace, _like_prefix(prefix)),
            ).fetchall()
        return [r[0] for r in rows]

    def count(self, namespace: str, prefix: str = "") -> int:
        return len(self.keys(namespace, prefix))

    def get_model(self, namespace: str, key: str, model: type[ModelT]) -> ModelT | None:
        data = self.get(namespace, key)
        if data is None:
            return None
        try:
            return model.model_validate(data)
        except Exception:  # stale schema - treat as a miss so it is recomputed
            return None

    # ---------------------------------------------------------------- stages
    def mark_stage(self, run_key: str, stage: str, detail: str | None = None) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO stages(run_key, stage, detail, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_key, stage) DO UPDATE SET detail = excluded.detail, "
                "updated_at = excluded.updated_at",
                (run_key, stage, detail, time.time()),
            )

    def stage_done(self, run_key: str, stage: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM stages WHERE run_key = ? AND stage = ?", (run_key, stage)
            ).fetchone()
        return row is not None

    def clear_stage(self, run_key: str, stage: str | None = None) -> None:
        with self._tx() as conn:
            if stage is None:
                conn.execute("DELETE FROM stages WHERE run_key = ?", (run_key,))
            else:
                conn.execute("DELETE FROM stages WHERE run_key = ? AND stage = ?", (run_key, stage))

    def stages(self, run_key: str) -> dict[str, str | None]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT stage, detail FROM stages WHERE run_key = ? ORDER BY updated_at",
                (run_key,),
            ).fetchall()
        return {stage: detail for stage, detail in rows}

    def reset_run(self, run_key: str) -> None:
        """Forget every checkpoint for a run (namespaces are prefixed by run key)."""

        with self._tx() as conn:
            conn.execute("DELETE FROM stages WHERE run_key = ?", (run_key,))
            conn.execute(
                "DELETE FROM kv WHERE namespace LIKE ? ESCAPE '\\'",
                (_like_prefix(f"{run_key}:"),),
            )
