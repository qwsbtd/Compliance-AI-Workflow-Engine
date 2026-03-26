"""
Workflow state persistence for durable, resumable workflows.

Workflows can pause (e.g., awaiting human approval) and resume hours or days
later. Persistence backends store workflow checkpoints with integrity hashes
to detect corruption.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class WorkflowState(Enum):
    """Lifecycle states of a workflow execution."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class WorkflowCheckpoint:
    """Serializable snapshot of workflow state for persistence."""

    workflow_id: str
    workflow_name: str
    state: WorkflowState
    current_step_index: int
    step_results: dict  # step_name -> serialized StepResult
    context: dict
    pending_approvals: dict  # step_name -> approval state
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    checkpoint_hash: str = ""

    def compute_hash(self) -> str:
        """Compute integrity hash over all fields except checkpoint_hash."""
        d = self.to_dict()
        d.pop("checkpoint_hash", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "state": self.state.value,
            "current_step_index": self.current_step_index,
            "step_results": self.step_results,
            "context": self.context,
            "pending_approvals": self.pending_approvals,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "checkpoint_hash": self.checkpoint_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorkflowCheckpoint:
        """Deserialize from dict."""
        return cls(
            workflow_id=d["workflow_id"],
            workflow_name=d["workflow_name"],
            state=WorkflowState(d["state"]),
            current_step_index=d["current_step_index"],
            step_results=d["step_results"],
            context=d["context"],
            pending_approvals=d["pending_approvals"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            checkpoint_hash=d.get("checkpoint_hash", ""),
        )


@runtime_checkable
class StatePersister(Protocol):
    """Abstract interface for persisting workflow checkpoints."""

    def save(self, checkpoint: WorkflowCheckpoint) -> None: ...
    def load(self, workflow_id: str) -> WorkflowCheckpoint | None: ...
    def list_workflows(self, state: WorkflowState | None = None) -> list[str]: ...
    def delete(self, workflow_id: str) -> None: ...


class SQLiteStatePersister:
    """Persist workflow state to SQLite.

    Good for single-node deployments. Verifies checkpoint integrity
    hash on load to detect corruption.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    workflow_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    checkpoint_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint with integrity hash."""
        checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
        checkpoint.checkpoint_hash = checkpoint.compute_hash()
        data = json.dumps(checkpoint.to_dict(), sort_keys=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workflow_checkpoints "
                "(workflow_id, state, checkpoint_json, checkpoint_hash, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    checkpoint.workflow_id,
                    checkpoint.state.value,
                    data,
                    checkpoint.checkpoint_hash,
                    checkpoint.updated_at,
                ),
            )
            conn.commit()

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """Load and verify checkpoint integrity."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT checkpoint_json, checkpoint_hash FROM workflow_checkpoints "
                "WHERE workflow_id = ?",
                (workflow_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

        checkpoint = WorkflowCheckpoint.from_dict(json.loads(row[0]))
        expected_hash = checkpoint.compute_hash()
        if expected_hash != row[1]:
            raise ValueError(
                f"Checkpoint integrity check failed for workflow {workflow_id}. "
                f"Expected hash {row[1]}, computed {expected_hash}. "
                "Data may be corrupted."
            )
        return checkpoint

    def list_workflows(self, state: WorkflowState | None = None) -> list[str]:
        """List workflow IDs, optionally filtered by state."""
        with sqlite3.connect(self._db_path) as conn:
            if state:
                cursor = conn.execute(
                    "SELECT workflow_id FROM workflow_checkpoints WHERE state = ?",
                    (state.value,),
                )
            else:
                cursor = conn.execute(
                    "SELECT workflow_id FROM workflow_checkpoints"
                )
            return [row[0] for row in cursor.fetchall()]

    def delete(self, workflow_id: str) -> None:
        """Delete a workflow checkpoint."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM workflow_checkpoints WHERE workflow_id = ?",
                (workflow_id,),
            )
            conn.commit()


class JSONStatePersister:
    """Persist each workflow to a JSON file in a directory.

    Uses atomic writes (write-to-tmp + os.replace) to prevent partial
    writes on crash. Suitable for air-gapped environments where SQLite
    may not be approved.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, workflow_id: str) -> Path:
        """Get file path for a workflow ID."""
        # Sanitize workflow_id for filesystem safety
        safe_id = workflow_id.replace("/", "_").replace("..", "_")
        return self._dir / f"{safe_id}.json"

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint atomically (write to tmp, then rename)."""
        checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
        checkpoint.checkpoint_hash = checkpoint.compute_hash()
        data = json.dumps(checkpoint.to_dict(), sort_keys=True, indent=2)

        target = self._path_for(checkpoint.workflow_id)
        tmp_path = target.with_suffix(".tmp")
        tmp_path.write_text(data, encoding="utf-8")
        os.replace(str(tmp_path), str(target))

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """Load and verify checkpoint integrity."""
        path = self._path_for(workflow_id)
        if not path.exists():
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = WorkflowCheckpoint.from_dict(data)
        expected_hash = checkpoint.compute_hash()
        if expected_hash != data.get("checkpoint_hash", ""):
            raise ValueError(
                f"Checkpoint integrity check failed for workflow {workflow_id}. "
                "Data may be corrupted."
            )
        return checkpoint

    def list_workflows(self, state: WorkflowState | None = None) -> list[str]:
        """List workflow IDs from directory."""
        results = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if state is None or data.get("state") == state.value:
                    results.append(data["workflow_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def delete(self, workflow_id: str) -> None:
        """Delete a workflow checkpoint file."""
        path = self._path_for(workflow_id)
        if path.exists():
            path.unlink()
