"""End-to-end integration tests for nuclear and clinical scenarios."""

import tempfile
import os

import pytest

from compliance_engine import (
    AIHumanTradeoff,
    ApprovalRequirement,
    AuditTrail,
    CompositeBackend,
    ExplainabilityInfo,
    FairnessConstraint,
    FairnessGate,
    FileBackend,
    HMACSigner,
    HumanCheckpoint,
    HumanDecision,
    KeyRing,
    ModelStep,
    NRC_PROFILE,
    ProcessingStep,
    RegulatoryFramework,
    ReportGenerator,
    SQLiteBackend,
    StepResult,
    StepStatus,
    WorkflowEngine,
)
from compliance_engine.persistence import WorkflowState


# ─── Shared Test Models ──────────────────────────────────────────────────────


class AnomalyModel(ModelStep):
    def __init__(self):
        super().__init__(
            name="Anomaly Detection",
            model_id="anomaly-v1",
            model_version="1.0",
            explain=True,
        )

    def predict(self, input_data, context):
        return {"anomaly": True, "score": 0.92, "decision": "anomaly_detected"}, 0.91

    def generate_explanation(self, input_data, prediction, context):
        return ExplainabilityInfo(
            method="feature_importance",
            summary="Vibration exceeded threshold",
            feature_importances={"vibration": 0.8, "temp": 0.2},
        )


class SepsisModel(ModelStep):
    def __init__(self):
        super().__init__(
            name="Sepsis Risk",
            model_id="sepsis-v1",
            model_version="2.0",
            explain=True,
        )

    def predict(self, input_data, context):
        return {"risk": "high", "score": 0.78, "decision": "high_risk"}, 0.87


# ─── Nuclear Scenario ────────────────────────────────────────────────────────


class TestNuclearScenario:
    def test_full_pipeline_with_nrc_compliance(self, tmp_path):
        """Complete nuclear monitoring pipeline with dual storage, signing, NRC profile."""
        keyring = KeyRing()
        key = keyring.generate_key("facility-key")
        signer = HMACSigner()

        primary = FileBackend(tmp_path / "primary.jsonl")
        secondary = FileBackend(tmp_path / "secondary.jsonl")
        backend = CompositeBackend([primary, secondary])

        audit_trail = AuditTrail(
            signer=signer, signing_key=key, backend=backend,
            workflow_id="NUC-001",
        )

        call_count = 0

        def dual_reviewer(ctx, instr):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return HumanDecision(approver_id="SUP-1", role="supervisor", approved=True)
            else:
                return HumanDecision(approver_id="HPO-1", role="health_physics", approved=True)

        engine = WorkflowEngine(
            workflow_name="Nuclear Anomaly Response",
            steps=[
                AnomalyModel(),
                HumanCheckpoint(
                    name="Dual Review",
                    review_callback=dual_reviewer,
                    approval_requirement=ApprovalRequirement(
                        min_approvals=2,
                        require_different_roles=True,
                    ),
                ),
                ProcessingStep(
                    name="Action",
                    processor=lambda i, c: StepResult(
                        status=StepStatus.COMPLETED,
                        output={"action": "investigate"},
                        decision="investigate",
                    ),
                ),
            ],
            audit_trail=audit_trail,
            compliance_profile=NRC_PROFILE,
            workflow_id="NUC-001",
        )

        # Pre-flight validation
        gaps = engine.validate_workflow()
        assert len(gaps) == 0

        # Run
        result = engine.run(initial_input={"unit": "CEN-042"})
        assert result.completed is True
        assert result.state == WorkflowState.COMPLETED

        # Chain integrity
        valid, errors = result.audit_trail.verify_chain()
        assert valid is True

        # Signatures
        sig_valid, sig_errors = result.audit_trail.verify_signatures(keyring)
        assert sig_valid is True

        # Dual backends
        assert primary.count() == secondary.count()
        assert primary.count() > 0

        # Report generation
        gen = ReportGenerator(result.audit_trail, NRC_PROFILE)
        report = gen.generate_inspector_report()
        assert "PASS" in report

    def test_nrc_rejects_missing_human_checkpoint(self):
        """NRC profile should flag model without human checkpoint."""
        engine = WorkflowEngine(
            workflow_name="Bad Pipeline",
            steps=[AnomalyModel()],
            compliance_profile=NRC_PROFILE,
        )
        gaps = engine.validate_workflow()
        assert len(gaps) > 0


# ─── Clinical Scenario ───────────────────────────────────────────────────────


class TestClinicalScenario:
    def test_full_pipeline_with_phi_redaction(self, tmp_path):
        """Complete clinical pipeline with PHI redaction, fairness gate, FDA+HIPAA."""
        profile = RegulatoryFramework.compose(
            "FDA 21 CFR Part 11", "HIPAA Privacy & Security"
        )
        assert profile.require_phi_redaction is True
        assert profile.require_signatures is True

        keyring = KeyRing()
        key = keyring.generate_key("hospital-key")
        signer = HMACSigner()

        backend = SQLiteBackend(tmp_path / "audit.db")
        audit_trail = AuditTrail(
            signer=signer, signing_key=key, backend=backend,
            phi_fields={"patient_name", "mrn", "ssn"},
            workflow_id="CLIN-001",
        )

        def physician_approve(ctx, instr):
            return HumanDecision(approver_id="DR-1", role="attending", approved=True)

        engine = WorkflowEngine(
            workflow_name="Sepsis Assessment",
            steps=[
                SepsisModel(),
                FairnessGate(
                    name="Equity Check",
                    constraints=[
                        FairnessConstraint(
                            metric="demographic_parity",
                            protected_attribute="race",
                            threshold=0.8,
                        ),
                    ],
                ),
                HumanCheckpoint(
                    name="Physician Review",
                    review_callback=physician_approve,
                ),
            ],
            audit_trail=audit_trail,
            compliance_profile=profile,
            workflow_id="CLIN-001",
        )

        # Fair predictions
        fair_data = {
            "predictions": [
                {"outcome": True}, {"outcome": False},
                {"outcome": True}, {"outcome": False},
            ],
            "groups": {"race": ["A", "A", "B", "B"]},
        }

        result = engine.run(
            initial_input=fair_data,
            context={
                "patient_name": "John Doe",
                "mrn": "MRN-123",
                "ssn": "000-00-0000",
                "heart_rate": 120,
            },
        )

        assert result.completed is True

        # PHI should be redacted in audit records
        for record in result.audit_trail.records:
            if "patient_name" in record.input_summary:
                assert record.input_summary["patient_name"] == "[REDACTED-PHI]"
            if "ssn" in record.input_summary:
                assert record.input_summary["ssn"] == "[REDACTED-PHI]"

        # Non-PHI should be preserved
        # heart_rate should not be redacted
        found_hr = False
        for record in result.audit_trail.records:
            if "heart_rate" in record.input_summary:
                assert record.input_summary["heart_rate"] == 120
                found_hr = True
        assert found_hr

        # Chain + signatures valid
        assert result.audit_trail.verify_chain()[0] is True
        assert result.audit_trail.verify_signatures(keyring)[0] is True

        # Backend has records
        assert backend.count() > 0

    def test_fairness_gate_blocks_biased_pipeline(self):
        """Biased predictions should block the clinical pipeline."""
        engine = WorkflowEngine(
            workflow_name="Biased Pipeline",
            steps=[
                SepsisModel(),
                FairnessGate(
                    name="Equity Check",
                    constraints=[
                        FairnessConstraint(
                            metric="disparate_impact",
                            protected_attribute="race",
                            threshold=0.8,
                            block_on_failure=True,
                        ),
                    ],
                ),
                HumanCheckpoint(name="Should Not Reach"),
            ],
        )

        # Very biased: group A all positive, group B all negative
        result = engine.run(initial_input={
            "predictions": [
                {"outcome": True}, {"outcome": True},
                {"outcome": False}, {"outcome": False},
            ],
            "groups": {"race": ["A", "A", "B", "B"]},
        })

        assert result.completed is False
        assert result.state == WorkflowState.FAILED
