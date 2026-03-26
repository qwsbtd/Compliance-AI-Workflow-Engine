"""Tests for pluggable audit storage backends."""

import json
import os
import sqlite3
import tempfile

import pytest

from compliance_engine.audit import AuditRecord, AuditTrail, GENESIS_HASH
from compliance_engine.backends import CompositeBackend, FileBackend, SQLiteBackend


def _make_record(record_id="r1", workflow_id="wf1"):
    """Create a minimal AuditRecord for testing."""
    r = AuditRecord(
        record_id=record_id, run_id="run1", workflow_id=workflow_id,
        step_name="s1", step_type="test", action="test", actor="me",
        timestamp="2026-01-01T00:00:00Z",
        input_summary={}, output_summary={},
        decision=None, confidence=None, model_info=None,
        explainability=None, tradeoff=None, escalated=False,
        escalation_reason=None, human_decision=None,
        phi_redacted=False, previous_hash=GENESIS_HASH,
        record_hash="abc123",
    )
    return r


class TestFileBackend:
    def test_write_and_read(self, tmp_path):
        backend = FileBackend(tmp_path / "audit.jsonl")
        record = _make_record()
        backend.write(record)
        records = backend.read_all()
        assert len(records) == 1
        assert records[0]["record_id"] == "r1"

    def test_count(self, tmp_path):
        backend = FileBackend(tmp_path / "audit.jsonl")
        assert backend.count() == 0
        backend.write(_make_record("r1"))
        backend.write(_make_record("r2"))
        assert backend.count() == 2

    def test_filter_by_workflow_id(self, tmp_path):
        backend = FileBackend(tmp_path / "audit.jsonl")
        backend.write(_make_record("r1", "wf1"))
        backend.write(_make_record("r2", "wf2"))
        records = backend.read_all(workflow_id="wf1")
        assert len(records) == 1
        assert records[0]["workflow_id"] == "wf1"

    def test_creates_parent_dirs(self, tmp_path):
        backend = FileBackend(tmp_path / "deep" / "nested" / "audit.jsonl")
        backend.write(_make_record())
        assert backend.count() == 1


class TestSQLiteBackend:
    def test_write_and_read(self, tmp_path):
        backend = SQLiteBackend(tmp_path / "audit.db")
        backend.write(_make_record())
        records = backend.read_all()
        assert len(records) == 1

    def test_count(self, tmp_path):
        backend = SQLiteBackend(tmp_path / "audit.db")
        backend.write(_make_record("r1"))
        backend.write(_make_record("r2"))
        assert backend.count() == 2

    def test_immutability_delete_blocked(self, tmp_path):
        db_path = tmp_path / "audit.db"
        backend = SQLiteBackend(db_path)
        backend.write(_make_record())

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("DELETE FROM audit_records WHERE record_id = 'r1'")

    def test_immutability_update_blocked(self, tmp_path):
        db_path = tmp_path / "audit.db"
        backend = SQLiteBackend(db_path)
        backend.write(_make_record())

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE audit_records SET record_json = '{}' WHERE record_id = 'r1'"
                )

    def test_filter_by_workflow_id(self, tmp_path):
        backend = SQLiteBackend(tmp_path / "audit.db")
        backend.write(_make_record("r1", "wf1"))
        backend.write(_make_record("r2", "wf2"))
        records = backend.read_all(workflow_id="wf1")
        assert len(records) == 1


class TestCompositeBackend:
    def test_writes_to_all_backends(self, tmp_path):
        b1 = FileBackend(tmp_path / "a.jsonl")
        b2 = FileBackend(tmp_path / "b.jsonl")
        composite = CompositeBackend([b1, b2])
        composite.write(_make_record())
        assert b1.count() == 1
        assert b2.count() == 1

    def test_read_from_primary(self, tmp_path):
        b1 = FileBackend(tmp_path / "a.jsonl")
        b2 = FileBackend(tmp_path / "b.jsonl")
        composite = CompositeBackend([b1, b2])
        composite.write(_make_record())
        records = composite.read_all()
        assert len(records) == 1

    def test_requires_at_least_one_backend(self):
        with pytest.raises(ValueError):
            CompositeBackend([])

    def test_integration_with_audit_trail(self, tmp_path):
        """End-to-end: AuditTrail writes through CompositeBackend."""
        b1 = FileBackend(tmp_path / "primary.jsonl")
        b2 = FileBackend(tmp_path / "secondary.jsonl")
        composite = CompositeBackend([b1, b2])

        trail = AuditTrail(workflow_id="wf1", backend=composite)
        trail.append(step_name="s1", step_type="t", action="a", actor="me")
        trail.append(step_name="s2", step_type="t", action="b", actor="me")

        assert b1.count() == 2
        assert b2.count() == 2
