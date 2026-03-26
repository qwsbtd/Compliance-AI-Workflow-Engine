"""
Pluggable audit storage backends.

Provides append-only storage for audit records with tamper resistance.
FileBackend for JSONL files, SQLiteBackend with DB-level immutability
triggers, and CompositeBackend for defense-in-depth redundancy.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .audit import AuditRecord


@runtime_checkable
class AuditBackend(Protocol):
    """Abstract interface for audit record storage."""

    def write(self, record: AuditRecord) -> None:
        """Persist a single audit record. Must be append-only."""
        ...

    def read_all(self, workflow_id: str | None = None) -> list[dict]:
        """Read all records, optionally filtered by workflow_id."""
        ...

    def count(self) -> int:
        """Return total number of stored records."""
        ...


class FileBackend:
    """Append-only JSONL file backend.

    Each record is written as a single JSON line, flushed immediately.
    Files can be made immutable via OS-level permissions (chattr +i on Linux).
    """

    def __init__(self, file_path: str | Path) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        """Append record as JSON line, flush immediately."""
        data = record.to_dict()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, sort_keys=True) + "\n")
            f.flush()

    def read_all(self, workflow_id: str | None = None) -> list[dict]:
        """Read all records from the JSONL file."""
        if not self._path.exists():
            return []
        records = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if workflow_id is None or record.get("workflow_id") == workflow_id:
                    records.append(record)
        return records

    def count(self) -> int:
        """Count records in the file."""
        if not self._path.exists():
            return 0
        count = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class SQLiteBackend:
    """SQLite backend with append-only semantics enforced via triggers.

    DELETE and UPDATE triggers RAISE(ABORT), making the table immutable
    at the database level. This is defense-in-depth: the application
    prevents mutation, the database prevents mutation, and the hash
    chain detects mutation.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create table and immutability triggers."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    workflow_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_id
                ON audit_records(workflow_id)
            """)
            # Immutability triggers — prevent DELETE and UPDATE at DB level
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_delete
                BEFORE DELETE ON audit_records
                BEGIN
                    SELECT RAISE(ABORT, 'Audit records are immutable: DELETE not allowed');
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_update
                BEFORE UPDATE ON audit_records
                BEGIN
                    SELECT RAISE(ABORT, 'Audit records are immutable: UPDATE not allowed');
                END
            """)
            conn.commit()

    def write(self, record: AuditRecord) -> None:
        """Insert a single audit record."""
        data = record.to_dict()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO audit_records (record_id, workflow_id, record_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (data["record_id"], data["workflow_id"], json.dumps(data, sort_keys=True), data["timestamp"]),
            )
            conn.commit()

    def read_all(self, workflow_id: str | None = None) -> list[dict]:
        """Read all records, optionally filtered."""
        with sqlite3.connect(self._db_path) as conn:
            if workflow_id:
                cursor = conn.execute(
                    "SELECT record_json FROM audit_records WHERE workflow_id = ? ORDER BY id",
                    (workflow_id,),
                )
            else:
                cursor = conn.execute(
                    "SELECT record_json FROM audit_records ORDER BY id"
                )
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def count(self) -> int:
        """Count total records."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM audit_records")
            return cursor.fetchone()[0]


class CompositeBackend:
    """Write to multiple backends simultaneously for defense-in-depth.

    Nuclear scenario: write to two independent file backends for
    redundant tamper-evident storage accessible to IAEA inspectors.
    """

    def __init__(self, backends: list[AuditBackend]) -> None:
        if not backends:
            raise ValueError("CompositeBackend requires at least one backend")
        self._backends = list(backends)

    def write(self, record: AuditRecord) -> None:
        """Write to ALL backends. Collects errors but attempts all writes."""
        errors = []
        for backend in self._backends:
            try:
                backend.write(record)
            except Exception as e:
                errors.append(f"{type(backend).__name__}: {e}")
        if errors:
            raise RuntimeError(
                f"CompositeBackend write failed on {len(errors)} backend(s): "
                + "; ".join(errors)
            )

    def read_all(self, workflow_id: str | None = None) -> list[dict]:
        """Read from the first backend (primary)."""
        return self._backends[0].read_all(workflow_id)

    def count(self) -> int:
        """Count from the first backend (primary)."""
        return self._backends[0].count()
