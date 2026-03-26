"""Tests for audit trail with SHA-256 hash chain and signatures."""

import json

import pytest

from compliance_engine.audit import GENESIS_HASH, AuditRecord, AuditTrail
from compliance_engine.crypto import HMACSigner, KeyRing


class TestAuditRecord:
    def test_canonical_bytes_deterministic(self):
        r = AuditRecord(
            record_id="r1", run_id="run1", workflow_id="wf1",
            step_name="s1", step_type="test", action="test", actor="me",
            timestamp="2026-01-01T00:00:00Z",
            input_summary={"b": 2, "a": 1}, output_summary={},
            decision=None, confidence=None, model_info=None,
            explainability=None, tradeoff=None, escalated=False,
            escalation_reason=None, human_decision=None,
            phi_redacted=False, previous_hash=GENESIS_HASH,
        )
        b1 = r.canonical_bytes()
        b2 = r.canonical_bytes()
        assert b1 == b2

    def test_compute_hash_is_sha256(self):
        r = AuditRecord(
            record_id="r1", run_id="run1", workflow_id="wf1",
            step_name="s1", step_type="test", action="test", actor="me",
            timestamp="2026-01-01T00:00:00Z",
            input_summary={}, output_summary={},
            decision=None, confidence=None, model_info=None,
            explainability=None, tradeoff=None, escalated=False,
            escalation_reason=None, human_decision=None,
            phi_redacted=False, previous_hash=GENESIS_HASH,
        )
        h = r.compute_hash()
        assert len(h) == 64  # SHA-256 hex digest length

    def test_to_dict_round_trip(self):
        r = AuditRecord(
            record_id="r1", run_id="run1", workflow_id="wf1",
            step_name="s1", step_type="test", action="test", actor="me",
            timestamp="2026-01-01T00:00:00Z",
            input_summary={"x": 1}, output_summary={"y": 2},
            decision="ok", confidence=0.9, model_info=None,
            explainability=None, tradeoff=None, escalated=False,
            escalation_reason=None, human_decision=None,
            phi_redacted=False, previous_hash=GENESIS_HASH,
            record_hash="abc", signature="sig", signed_by="key1",
        )
        d = r.to_dict()
        r2 = AuditRecord.from_dict(d)
        assert r2.record_id == "r1"
        assert r2.signature == "sig"


class TestAuditTrail:
    def test_append_creates_record(self):
        trail = AuditTrail(workflow_id="wf1")
        record = trail.append(step_name="s1", step_type="t", action="a", actor="me")
        assert record.step_name == "s1"
        assert record.record_hash != ""
        assert record.previous_hash == GENESIS_HASH
        assert len(trail) == 1

    def test_hash_chain_links(self):
        trail = AuditTrail(workflow_id="wf1")
        r1 = trail.append(step_name="s1", step_type="t", action="a", actor="me")
        r2 = trail.append(step_name="s2", step_type="t", action="a", actor="me")
        assert r2.previous_hash == r1.record_hash

    def test_verify_chain_valid(self):
        trail = AuditTrail(workflow_id="wf1")
        trail.append(step_name="s1", step_type="t", action="a", actor="me")
        trail.append(step_name="s2", step_type="t", action="a", actor="me")
        trail.append(step_name="s3", step_type="t", action="a", actor="me")
        valid, errors = trail.verify_chain()
        assert valid is True
        assert errors == []

    def test_verify_chain_detects_tampering(self):
        trail = AuditTrail(workflow_id="wf1")
        trail.append(step_name="s1", step_type="t", action="a", actor="me")
        trail.append(step_name="s2", step_type="t", action="a", actor="me")

        # Tamper with the first record by replacing it
        tampered = AuditRecord(
            record_id=trail._records[0].record_id,
            run_id=trail._records[0].run_id,
            workflow_id=trail._records[0].workflow_id,
            step_name="TAMPERED",
            step_type="t", action="a", actor="me",
            timestamp=trail._records[0].timestamp,
            input_summary={}, output_summary={},
            decision=None, confidence=None, model_info=None,
            explainability=None, tradeoff=None, escalated=False,
            escalation_reason=None, human_decision=None,
            phi_redacted=False,
            previous_hash=trail._records[0].previous_hash,
            record_hash=trail._records[0].record_hash,  # keep old hash
        )
        trail._records[0] = tampered

        valid, errors = trail.verify_chain()
        assert valid is False
        assert len(errors) > 0
        assert "hash mismatch" in errors[0].lower() or "tampered" in errors[0].lower()

    def test_phi_redaction(self):
        trail = AuditTrail(
            workflow_id="wf1",
            phi_fields={"patient_name", "ssn"},
        )
        record = trail.append(
            step_name="s1", step_type="t", action="a", actor="me",
            input_summary={"patient_name": "Jane Doe", "ssn": "123-45-6789", "score": 0.9},
        )
        assert record.input_summary["patient_name"] == "[REDACTED-PHI]"
        assert record.input_summary["ssn"] == "[REDACTED-PHI]"
        assert record.input_summary["score"] == 0.9
        assert record.phi_redacted is True

    def test_signatures(self):
        keyring = KeyRing()
        key = keyring.generate_key("signer-1")
        signer = HMACSigner()

        trail = AuditTrail(
            workflow_id="wf1",
            signer=signer,
            signing_key=key,
        )
        record = trail.append(step_name="s1", step_type="t", action="a", actor="me")
        assert record.signature is not None
        assert record.signed_by == "signer-1"

        valid, errors = trail.verify_signatures(keyring)
        assert valid is True
        assert errors == []

    def test_to_json(self):
        trail = AuditTrail(workflow_id="wf1")
        trail.append(step_name="s1", step_type="t", action="a", actor="me")
        j = trail.to_json()
        data = json.loads(j)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_get_records_filters(self):
        trail = AuditTrail(workflow_id="wf1")
        trail.append(step_name="s1", step_type="t", action="start", actor="me")
        trail.append(step_name="s1", step_type="t", action="complete", actor="me")
        trail.append(step_name="s2", step_type="t", action="start", actor="me")

        by_step = trail.get_records(step_name="s1")
        assert len(by_step) == 2

        by_action = trail.get_records(action="start")
        assert len(by_action) == 2
