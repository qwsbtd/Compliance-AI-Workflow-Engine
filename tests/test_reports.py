"""Tests for compliance report generation."""

import json

from compliance_engine.audit import AuditTrail
from compliance_engine.regulatory import NRC_PROFILE
from compliance_engine.reports import ReportFormat, ReportGenerator


def _build_trail():
    """Create an audit trail with several records for testing."""
    trail = AuditTrail(workflow_id="wf-test")
    trail.append(
        step_name="_workflow", step_type="system",
        action="workflow_started", actor="engine",
    )
    trail.append(
        step_name="AI Step", step_type="model",
        action="step_completed", actor="engine",
        decision="approve", confidence=0.92,
        model_info={"model_id": "test", "model_version": "1.0", "input_hash": "abc123"},
        explainability={"method": "shap", "summary": "Feature X was key"},
    )
    trail.append(
        step_name="Review", step_type="human_checkpoint",
        action="approval_submitted", actor="DR-SMITH",
        human_decision={"approver_id": "DR-SMITH", "approved": True},
    )
    trail.append(
        step_name="_workflow", step_type="system",
        action="workflow_completed", actor="engine",
    )
    return trail


class TestReportGenerator:
    def test_text_report_contains_sections(self):
        trail = _build_trail()
        gen = ReportGenerator(trail, NRC_PROFILE)
        report = gen.generate_compliance_report()

        assert "COMPLIANCE REPORT" in report
        assert "Executive Summary" in report
        assert "Regulatory Framework" in report
        assert "Step Audit Log" in report
        assert "Approval Chain" in report
        assert "Model Versioning" in report
        assert "Explainability" in report
        assert "Chain Integrity" in report
        assert "END OF REPORT" in report

    def test_json_report_is_valid_json(self):
        trail = _build_trail()
        gen = ReportGenerator(trail)
        report = gen.generate_compliance_report(fmt=ReportFormat.JSON)
        data = json.loads(report)
        assert "executive_summary" in data
        assert "chain_integrity" in data

    def test_inspector_report(self):
        trail = _build_trail()
        gen = ReportGenerator(trail, NRC_PROFILE)
        report = gen.generate_inspector_report()
        assert "REGULATORY INSPECTOR REPORT" in report
        assert "Chain Integrity" in report

    def test_audit_summary(self):
        trail = _build_trail()
        gen = ReportGenerator(trail)
        summary = gen.generate_audit_summary()
        assert summary["total_records"] == 4
        assert summary["chain_valid"] is True
        assert isinstance(summary["actions"], dict)

    def test_report_with_no_profile(self):
        trail = _build_trail()
        gen = ReportGenerator(trail)
        report = gen.generate_compliance_report()
        assert "No compliance profile configured" in report

    def test_chain_integrity_section(self):
        trail = _build_trail()
        gen = ReportGenerator(trail)
        report = gen.generate_compliance_report(fmt=ReportFormat.JSON)
        data = json.loads(report)
        assert data["chain_integrity"]["chain_valid"] is True
