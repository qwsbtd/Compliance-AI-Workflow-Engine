"""
Compliance report generation from audit trail data.

Generates human-readable compliance reports (text, JSON) for regulatory
review. These are summary documents for auditors — NOT structured evidence
bundles or integrity manifests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from .audit import AuditTrail
from .regulatory import ComplianceProfile

if TYPE_CHECKING:
    from .crypto import KeyRing


class ReportFormat(Enum):
    TEXT = "text"
    JSON = "json"


class ReportGenerator:
    """Generates compliance reports from audit trail data.

    Produces human-readable summaries covering: workflow execution,
    approval chains, model versioning, explainability, fairness
    assessments, chain integrity, and compliance gap analysis.
    """

    def __init__(
        self,
        audit_trail: AuditTrail,
        profile: ComplianceProfile | None = None,
    ) -> None:
        self._audit = audit_trail
        self._profile = profile

    def generate_compliance_report(
        self,
        workflow_id: str | None = None,
        fmt: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """Generate a full compliance report."""
        records = self._audit.records
        if workflow_id:
            records = [r for r in records if r.workflow_id == workflow_id]

        sections = {
            "executive_summary": self._executive_summary(records),
            "regulatory_framework": self._regulatory_section(),
            "step_audit_log": self._step_audit_log(records),
            "approval_chain": self._approval_chain(records),
            "model_versioning": self._model_versioning(records),
            "explainability": self._explainability_section(records),
            "fairness_assessment": self._fairness_section(records),
            "chain_integrity": self._chain_integrity(),
            "compliance_gaps": self._compliance_gaps(),
        }

        if fmt == ReportFormat.JSON:
            return json.dumps(sections, indent=2, default=str)
        return self._render_text(sections)

    def generate_inspector_report(
        self,
        workflow_id: str | None = None,
        fmt: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """Condensed report for regulatory inspectors (NRC, FDA).

        Focuses on: chain integrity, signatures, approval chains,
        and compliance status.
        """
        records = self._audit.records
        if workflow_id:
            records = [r for r in records if r.workflow_id == workflow_id]

        sections = {
            "summary": self._executive_summary(records),
            "chain_integrity": self._chain_integrity(),
            "approval_chain": self._approval_chain(records),
            "compliance_status": self._compliance_gaps(),
        }

        if fmt == ReportFormat.JSON:
            return json.dumps(sections, indent=2, default=str)
        return self._render_inspector_text(sections)

    def generate_audit_summary(self, workflow_id: str | None = None) -> dict:
        """Machine-readable summary for dashboards."""
        records = self._audit.records
        if workflow_id:
            records = [r for r in records if r.workflow_id == workflow_id]

        chain_valid, chain_errors = self._audit.verify_chain()

        return {
            "workflow_id": workflow_id or self._audit.workflow_id,
            "total_records": len(records),
            "actions": self._count_actions(records),
            "chain_valid": chain_valid,
            "chain_errors": len(chain_errors),
            "has_signatures": any(r.signature for r in records),
            "phi_redacted_records": sum(1 for r in records if r.phi_redacted),
            "escalations": sum(1 for r in records if r.escalated),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ─── Section Generators ──────────────────────────────────────────────

    def _executive_summary(self, records: list) -> dict:
        """Generate executive summary."""
        if not records:
            return {"status": "no_records", "total_records": 0}

        workflow_start = None
        workflow_end = None
        workflow_status = "unknown"

        for r in records:
            if r.action == "workflow_started":
                workflow_start = r.timestamp
            if r.action == "workflow_completed":
                workflow_end = r.timestamp
                workflow_status = "completed"
            if r.action == "pipeline_blocked":
                workflow_status = "blocked"
            if r.action == "workflow_halted_rejection":
                workflow_status = "rejected"

        return {
            "workflow_id": self._audit.workflow_id,
            "run_id": self._audit.run_id,
            "status": workflow_status,
            "total_records": len(records),
            "started_at": workflow_start,
            "completed_at": workflow_end,
            "unique_steps": len(set(r.step_name for r in records)),
        }

    def _regulatory_section(self) -> dict:
        """Generate regulatory framework section."""
        if not self._profile:
            return {"profile": "none configured"}

        return {
            "profile_name": self._profile.name,
            "framework_ids": self._profile.framework_ids,
            "total_requirements": len(self._profile.requirements),
            "requirements": [
                {
                    "req_id": r.req_id,
                    "description": r.description,
                    "category": r.category,
                    "severity": r.severity,
                }
                for r in self._profile.requirements
            ],
        }

    def _step_audit_log(self, records: list) -> list[dict]:
        """Generate per-step audit log."""
        step_records = [
            r for r in records if r.action in ("step_started", "step_completed")
        ]
        entries = []
        for r in step_records:
            entry = {
                "step_name": r.step_name,
                "action": r.action,
                "timestamp": r.timestamp,
                "decision": r.decision,
                "confidence": r.confidence,
            }
            if r.escalated:
                entry["escalated"] = True
                entry["escalation_reason"] = r.escalation_reason
            entries.append(entry)
        return entries

    def _approval_chain(self, records: list) -> list[dict]:
        """Generate approval chain log."""
        approval_records = [
            r
            for r in records
            if r.action in ("approval_submitted", "rejection_submitted")
        ]
        return [
            {
                "step_name": r.step_name,
                "action": r.action,
                "actor": r.actor,
                "timestamp": r.timestamp,
                "human_decision": r.human_decision,
                "signed": r.signature is not None,
            }
            for r in approval_records
        ]

    def _model_versioning(self, records: list) -> list[dict]:
        """Generate model versioning summary."""
        model_records = [r for r in records if r.model_info is not None]
        return [
            {
                "step_name": r.step_name,
                "timestamp": r.timestamp,
                "model_info": r.model_info,
            }
            for r in model_records
        ]

    def _explainability_section(self, records: list) -> list[dict]:
        """Generate explainability summary."""
        explain_records = [r for r in records if r.explainability is not None]
        return [
            {
                "step_name": r.step_name,
                "timestamp": r.timestamp,
                "explainability": r.explainability,
            }
            for r in explain_records
        ]

    def _fairness_section(self, records: list) -> list[dict]:
        """Generate fairness assessment summary."""
        fairness_records = [
            r for r in records if r.step_type == "fairness_gate"
        ]
        return [
            {
                "step_name": r.step_name,
                "action": r.action,
                "decision": r.decision,
                "timestamp": r.timestamp,
                "output": r.output_summary,
            }
            for r in fairness_records
        ]

    def _chain_integrity(self) -> dict:
        """Verify and report on chain integrity."""
        valid, errors = self._audit.verify_chain()
        return {
            "chain_valid": valid,
            "total_records": len(self._audit),
            "errors": errors,
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _compliance_gaps(self) -> dict:
        """Analyze compliance gaps."""
        if not self._profile:
            return {"profile": "none", "gaps": []}

        # We can't access steps from here, so report profile info
        return {
            "profile_name": self._profile.name,
            "framework_ids": self._profile.framework_ids,
            "policy_flags": {
                "require_signatures": self._profile.require_signatures,
                "require_human_after_model": self._profile.require_human_checkpoint_after_model,
                "min_approvers": self._profile.min_approvers,
                "require_explainability": self._profile.require_explainability,
                "require_fairness_check": self._profile.require_fairness_check,
                "require_phi_redaction": self._profile.require_phi_redaction,
                "require_model_versioning": self._profile.require_model_versioning,
            },
        }

    def _count_actions(self, records: list) -> dict:
        """Count occurrences of each action type."""
        counts: dict[str, int] = {}
        for r in records:
            counts[r.action] = counts.get(r.action, 0) + 1
        return counts

    # ─── Renderers ───────────────────────────────────────────────────────

    def _render_text(self, sections: dict) -> str:
        """Render full compliance report as text."""
        lines = []
        lines.append("=" * 72)
        lines.append("COMPLIANCE REPORT")
        lines.append("=" * 72)

        # Executive Summary
        summary = sections["executive_summary"]
        lines.append("\n## Executive Summary")
        lines.append(f"  Workflow ID:   {summary.get('workflow_id', 'N/A')}")
        lines.append(f"  Run ID:        {summary.get('run_id', 'N/A')}")
        lines.append(f"  Status:        {summary.get('status', 'N/A')}")
        lines.append(f"  Total Records: {summary.get('total_records', 0)}")
        lines.append(f"  Started:       {summary.get('started_at', 'N/A')}")
        lines.append(f"  Completed:     {summary.get('completed_at', 'N/A')}")

        # Regulatory Framework
        reg = sections["regulatory_framework"]
        lines.append("\n## Regulatory Framework")
        if reg.get("profile") == "none configured":
            lines.append("  No compliance profile configured.")
        else:
            lines.append(f"  Profile: {reg.get('profile_name', 'N/A')}")
            lines.append(f"  Frameworks: {', '.join(reg.get('framework_ids', []))}")
            lines.append(f"  Requirements: {reg.get('total_requirements', 0)}")

        # Step Audit Log
        steps = sections["step_audit_log"]
        lines.append(f"\n## Step Audit Log ({len(steps)} entries)")
        for entry in steps:
            marker = " [ESCALATED]" if entry.get("escalated") else ""
            decision_str = f" -> {entry['decision']}" if entry.get("decision") else ""
            conf = entry.get("confidence")
            conf_str = f" (confidence: {conf:.2f})" if conf is not None else ""
            lines.append(
                f"  [{entry['timestamp']}] {entry['step_name']}: "
                f"{entry['action']}{decision_str}{conf_str}{marker}"
            )

        # Approval Chain
        approvals = sections["approval_chain"]
        lines.append(f"\n## Approval Chain ({len(approvals)} entries)")
        for a in approvals:
            signed = " [SIGNED]" if a.get("signed") else ""
            lines.append(
                f"  [{a['timestamp']}] {a['step_name']}: "
                f"{a['actor']} -> {a['action']}{signed}"
            )

        # Model Versioning
        models = sections["model_versioning"]
        lines.append(f"\n## Model Versioning ({len(models)} entries)")
        for m in models:
            info = m.get("model_info", {})
            lines.append(
                f"  {m['step_name']}: {info.get('model_id', 'N/A')} "
                f"v{info.get('model_version', 'N/A')} "
                f"(input_hash: {info.get('input_hash', 'N/A')[:16]}...)"
            )

        # Explainability
        explains = sections["explainability"]
        lines.append(f"\n## Explainability ({len(explains)} entries)")
        for e in explains:
            info = e.get("explainability", {})
            lines.append(
                f"  {e['step_name']}: [{info.get('method', 'N/A')}] "
                f"{info.get('summary', 'N/A')}"
            )

        # Fairness Assessment
        fairness = sections["fairness_assessment"]
        lines.append(f"\n## Fairness Assessment ({len(fairness)} entries)")
        for f_entry in fairness:
            lines.append(
                f"  {f_entry['step_name']}: {f_entry.get('decision', 'N/A')} "
                f"({f_entry['action']})"
            )

        # Chain Integrity
        integrity = sections["chain_integrity"]
        lines.append("\n## Chain Integrity Verification")
        status = "VALID" if integrity["chain_valid"] else "INVALID"
        lines.append(f"  Status:  {status}")
        lines.append(f"  Records: {integrity['total_records']}")
        lines.append(f"  Verified: {integrity['verification_timestamp']}")
        if integrity["errors"]:
            lines.append("  Errors:")
            for err in integrity["errors"]:
                lines.append(f"    - {err}")

        # Compliance Gaps
        gaps = sections["compliance_gaps"]
        lines.append("\n## Compliance Status")
        if gaps.get("profile") == "none":
            lines.append("  No compliance profile configured.")
        else:
            lines.append(f"  Profile: {gaps.get('profile_name', 'N/A')}")
            flags = gaps.get("policy_flags", {})
            for flag, value in flags.items():
                lines.append(f"    {flag}: {value}")

        lines.append("\n" + "=" * 72)
        lines.append("END OF REPORT")
        lines.append("=" * 72)

        return "\n".join(lines)

    def _render_inspector_text(self, sections: dict) -> str:
        """Render condensed inspector report."""
        lines = []
        lines.append("=" * 72)
        lines.append("REGULATORY INSPECTOR REPORT")
        lines.append("=" * 72)

        summary = sections["summary"]
        lines.append(f"\nWorkflow: {summary.get('workflow_id', 'N/A')}")
        lines.append(f"Status:   {summary.get('status', 'N/A')}")
        lines.append(f"Records:  {summary.get('total_records', 0)}")

        integrity = sections["chain_integrity"]
        status = "PASS" if integrity["chain_valid"] else "FAIL"
        lines.append(f"\nChain Integrity: {status}")
        if integrity["errors"]:
            for err in integrity["errors"]:
                lines.append(f"  ERROR: {err}")

        approvals = sections["approval_chain"]
        lines.append(f"\nApproval Chain: {len(approvals)} entries")
        for a in approvals:
            signed = " [SIGNED]" if a.get("signed") else " [UNSIGNED]"
            lines.append(f"  {a['actor']} -> {a['action']}{signed}")

        gaps = sections["compliance_status"]
        lines.append(f"\nCompliance Profile: {gaps.get('profile_name', 'N/A')}")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)
