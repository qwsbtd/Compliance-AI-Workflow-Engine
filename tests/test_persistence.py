"""Tests for workflow state persistence."""

import pytest

from compliance_engine.persistence import (
    JSONStatePersister,
    SQLiteStatePersister,
    WorkflowCheckpoint,
    WorkflowState,
)


def _make_checkpoint(workflow_id="wf1", state=WorkflowState.RUNNING):
    cp = WorkflowCheckpoint(
        workflow_id=workflow_id,
        workflow_name="Test Workflow",
        state=state,
        current_step_index=2,
        step_results={"step1": {"status": "completed"}},
        context={"key": "value"},
        pending_approvals={},
    )
    cp.checkpoint_hash = cp.compute_hash()
    return cp


class TestWorkflowCheckpoint:
    def test_compute_hash_deterministic(self):
        cp = _make_checkpoint()
        h1 = cp.compute_hash()
        h2 = cp.compute_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_to_dict_round_trip(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        cp2 = WorkflowCheckpoint.from_dict(d)
        assert cp2.workflow_id == cp.workflow_id
        assert cp2.state == cp.state
        assert cp2.current_step_index == cp.current_step_index

    def test_hash_changes_with_data(self):
        cp1 = _make_checkpoint(workflow_id="a")
        cp2 = _make_checkpoint(workflow_id="b")
        assert cp1.compute_hash() != cp2.compute_hash()


class TestSQLiteStatePersister:
    def test_save_and_load(self, tmp_path):
        persister = SQLiteStatePersister(tmp_path / "state.db")
        cp = _make_checkpoint()
        persister.save(cp)

        loaded = persister.load("wf1")
        assert loaded is not None
        assert loaded.workflow_id == "wf1"
        assert loaded.state == WorkflowState.RUNNING

    def test_load_nonexistent(self, tmp_path):
        persister = SQLiteStatePersister(tmp_path / "state.db")
        assert persister.load("nope") is None

    def test_integrity_check_on_load(self, tmp_path):
        import json
        import sqlite3

        db_path = tmp_path / "state.db"
        persister = SQLiteStatePersister(db_path)
        cp = _make_checkpoint()
        persister.save(cp)

        # Tamper with the stored JSON
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE workflow_checkpoints SET checkpoint_json = ? WHERE workflow_id = ?",
                ('{"workflow_id":"wf1","workflow_name":"TAMPERED","state":"running",'
                 '"current_step_index":2,"step_results":{},"context":{},'
                 '"pending_approvals":{},"created_at":"x","updated_at":"x",'
                 '"checkpoint_hash":"wrong"}',
                 "wf1"),
            )
            conn.commit()

        with pytest.raises(ValueError, match="integrity"):
            persister.load("wf1")

    def test_list_workflows(self, tmp_path):
        persister = SQLiteStatePersister(tmp_path / "state.db")
        persister.save(_make_checkpoint("wf1", WorkflowState.RUNNING))
        persister.save(_make_checkpoint("wf2", WorkflowState.COMPLETED))

        all_wf = persister.list_workflows()
        assert sorted(all_wf) == ["wf1", "wf2"]

        running = persister.list_workflows(state=WorkflowState.RUNNING)
        assert running == ["wf1"]

    def test_delete(self, tmp_path):
        persister = SQLiteStatePersister(tmp_path / "state.db")
        persister.save(_make_checkpoint())
        persister.delete("wf1")
        assert persister.load("wf1") is None


class TestJSONStatePersister:
    def test_save_and_load(self, tmp_path):
        persister = JSONStatePersister(tmp_path / "states")
        cp = _make_checkpoint()
        persister.save(cp)

        loaded = persister.load("wf1")
        assert loaded is not None
        assert loaded.workflow_id == "wf1"

    def test_load_nonexistent(self, tmp_path):
        persister = JSONStatePersister(tmp_path / "states")
        assert persister.load("nope") is None

    def test_atomic_write(self, tmp_path):
        """Verify no .tmp files remain after save."""
        state_dir = tmp_path / "states"
        persister = JSONStatePersister(state_dir)
        persister.save(_make_checkpoint())

        files = list(state_dir.iterdir())
        assert all(not f.name.endswith(".tmp") for f in files)

    def test_list_workflows(self, tmp_path):
        persister = JSONStatePersister(tmp_path / "states")
        persister.save(_make_checkpoint("wf1", WorkflowState.RUNNING))
        persister.save(_make_checkpoint("wf2", WorkflowState.COMPLETED))

        all_wf = persister.list_workflows()
        assert sorted(all_wf) == ["wf1", "wf2"]

    def test_delete(self, tmp_path):
        persister = JSONStatePersister(tmp_path / "states")
        persister.save(_make_checkpoint())
        persister.delete("wf1")
        assert persister.load("wf1") is None

    def test_integrity_check(self, tmp_path):
        state_dir = tmp_path / "states"
        persister = JSONStatePersister(state_dir)
        persister.save(_make_checkpoint())

        # Tamper with the file
        fp = state_dir / "wf1.json"
        fp.write_text('{"workflow_id":"wf1","workflow_name":"x","state":"running",'
                       '"current_step_index":0,"step_results":{},"context":{},'
                       '"pending_approvals":{},"created_at":"x","updated_at":"x",'
                       '"checkpoint_hash":"wrong"}')

        with pytest.raises(ValueError, match="integrity"):
            persister.load("wf1")
