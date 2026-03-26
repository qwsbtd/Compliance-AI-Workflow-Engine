#!/usr/bin/env python3
"""
Nuclear Centrifuge Anomaly Detection Pipeline

Demonstrates the Compliance AI Workflow Engine in an NRC-regulated
nuclear enrichment facility. AI monitors centrifuge vibration sensors
for anomalies. Any detected anomaly must be reviewed by TWO independent
operators (four-eyes principle) before action is taken.

Features demonstrated:
- ModelStep with full versioning and explainability
- Multi-party HumanCheckpoint (2 approvers, different roles required)
- NRC compliance profile validation
- CompositeBackend (dual audit storage for IAEA inspectors)
- Cryptographic signatures on all audit records
- Inspector report generation
- Hash chain integrity verification

Run: python examples/nuclear_monitoring.py
"""

import sys
import os
import tempfile

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compliance_engine import (
    AIHumanTradeoff,
    ApprovalRequirement,
    AuditTrail,
    CompositeBackend,
    ExplainabilityInfo,
    FileBackend,
    HMACSigner,
    HumanCheckpoint,
    HumanDecision,
    KeyRing,
    ModelStep,
    NRC_PROFILE,
    ProcessingStep,
    ReportGenerator,
    StepResult,
    StepStatus,
    WorkflowEngine,
)


# ─── Custom Steps ─────────────────────────────────────────────────────────────


class CentrifugeAnomalyModel(ModelStep):
    """AI model that analyzes centrifuge vibration sensor data."""

    def __init__(self):
        super().__init__(
            name="Centrifuge Anomaly Detection",
            model_id="centrifuge-anomaly-detector",
            model_version="3.2.1",
            model_config={
                "algorithm": "isolation_forest",
                "contamination": 0.05,
                "n_estimators": 200,
            },
            model_hash="a1b2c3d4e5f6789012345678abcdef0123456789abcdef0123456789abcdef01",
            framework="sklearn",
            explain=True,
            confidence_threshold=0.85,
            tradeoff=AIHumanTradeoff(
                step_name="Centrifuge Anomaly Detection",
                performer="ai",
                rationale=(
                    "Centrifuge vibration analysis requires continuous monitoring "
                    "of 500+ sensors at sub-second intervals. Human operators cannot "
                    "process this volume. AI provides real-time anomaly flagging; "
                    "all flagged anomalies require mandatory human review."
                ),
                risk_level="high",
                regulatory_reference="10 CFR 74.19 (Material Control & Accounting)",
                fallback_plan="Route to senior nuclear engineer for manual sensor review",
                review_frequency="Quarterly model performance audit",
            ),
        )

    def predict(self, input_data, context):
        """Simulate centrifuge anomaly detection."""
        # Simulated sensor readings
        sensor_data = input_data or {
            "unit_id": "CEN-042",
            "vibration_hz": 847.3,
            "temperature_c": 62.1,
            "pressure_kpa": 101.2,
            "rpm": 50200,
        }

        # Simulated anomaly detection
        anomaly_score = 0.92
        anomaly_type = "vibration_deviation"
        confidence = 0.91

        prediction = {
            "anomaly_detected": True,
            "anomaly_score": anomaly_score,
            "anomaly_type": anomaly_type,
            "affected_unit": sensor_data.get("unit_id", "unknown"),
            "severity": "elevated",
            "recommended_action": "investigate",
            "decision": f"Anomaly detected on {sensor_data.get('unit_id', 'unknown')} "
                        f"(score: {anomaly_score:.2f})",
        }

        return prediction, confidence

    def generate_explanation(self, input_data, prediction, context):
        """Explain which sensor readings triggered the anomaly."""
        return ExplainabilityInfo(
            method="feature_importance",
            summary=(
                f"Anomaly on unit {prediction.get('affected_unit', 'unknown')}: "
                f"Vibration frequency 847.3 Hz exceeds normal range (800-840 Hz). "
                f"Temperature nominal. Pressure nominal."
            ),
            feature_importances={
                "vibration_hz": 0.78,
                "rpm": 0.12,
                "temperature_c": 0.06,
                "pressure_kpa": 0.04,
            },
            reasoning_trace=[
                "1. Ingested 500 sensor readings from unit CEN-042",
                "2. Isolation forest flagged vibration_hz as anomalous (z-score: 3.2)",
                "3. Cross-referenced with historical maintenance data",
                "4. Pattern matches 'bearing wear' signature with 91% confidence",
                "5. Recommended action: investigate (not immediate shutdown)",
            ],
            confidence=0.91,
        )


def material_accounting_check(input_data, context):
    """Cross-reference anomaly with material accounting records."""
    affected_unit = context.get("affected_unit", "unknown")
    return StepResult(
        status=StepStatus.COMPLETED,
        output={
            "material_status": "consistent",
            "last_inventory": "2026-03-25T08:00:00Z",
            "material_balance": "within_limits",
            "diversion_indicators": "none",
        },
        decision=f"Material accounting for {affected_unit}: consistent, no diversion indicators",
    )


def action_decision(input_data, context):
    """Determine action based on approved anomaly assessment."""
    severity = context.get("severity", "unknown")
    approved = context.get("approved", False)

    if not approved:
        action = "no_action_pending_review"
    elif severity == "critical":
        action = "immediate_shutdown"
    elif severity == "elevated":
        action = "schedule_inspection"
    else:
        action = "log_and_monitor"

    return StepResult(
        status=StepStatus.COMPLETED,
        output={"final_action": action},
        decision=f"Action determined: {action}",
    )


# ─── Simulated Human Reviewers ───────────────────────────────────────────────


def shift_supervisor_review(context, instructions):
    """Simulated shift supervisor approval."""
    return HumanDecision(
        approver_id="SUP-WILLIAMS",
        role="shift_supervisor",
        approved=True,
        comment="Vibration pattern consistent with bearing wear. Approve investigation.",
    )


def health_physics_review(context, instructions):
    """Simulated health physics officer approval."""
    return HumanDecision(
        approver_id="HPO-CHEN",
        role="health_physics",
        approved=True,
        comment="No radiation anomalies detected. Material accounting clear. Approve.",
    )


# Track which reviewer to call next
_review_count = 0


def dual_reviewer_callback(context, instructions):
    """Alternates between two reviewers for four-eyes demonstration."""
    global _review_count
    _review_count += 1
    if _review_count % 2 == 1:
        return shift_supervisor_review(context, instructions)
    else:
        return health_physics_review(context, instructions)


# ─── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    print("=" * 72)
    print("NUCLEAR CENTRIFUGE ANOMALY DETECTION PIPELINE")
    print("NRC-Regulated Environment | Four-Eyes Principle")
    print("=" * 72)

    # Set up cryptographic signing
    keyring = KeyRing()
    system_key = keyring.generate_key(
        "nuclear-facility-001",
        metadata={"facility": "Enrichment Plant Alpha", "role": "system"},
    )
    signer = HMACSigner()

    # Set up dual audit storage (defense-in-depth for IAEA)
    tmpdir = tempfile.mkdtemp(prefix="nuclear_audit_")
    primary_backend = FileBackend(os.path.join(tmpdir, "primary_audit.jsonl"))
    secondary_backend = FileBackend(os.path.join(tmpdir, "secondary_audit.jsonl"))
    composite_backend = CompositeBackend([primary_backend, secondary_backend])

    # Set up audit trail with signing and dual storage
    audit_trail = AuditTrail(
        signer=signer,
        signing_key=system_key,
        backend=composite_backend,
        workflow_id="NUC-INCIDENT-2026-0042",
    )

    # Build the pipeline
    # NRC profile requires HumanCheckpoint immediately after every ModelStep
    steps = [
        CentrifugeAnomalyModel(),
        HumanCheckpoint(
            name="Dual Operator Review",
            approval_requirement=ApprovalRequirement(
                min_approvals=2,
                eligible_approvers=[
                    "SUP-WILLIAMS",
                    "HPO-CHEN",
                    "SFG-RODRIGUEZ",
                ],
                require_different_people=True,
                require_different_roles=True,
                approval_timeout_seconds=1800,  # 30 minutes
                escalation_chain=["plant_manager", "NRC_resident_inspector"],
            ),
            instructions=(
                "MANDATORY DUAL REVIEW: Two operators from different roles must "
                "independently verify the anomaly assessment before any action. "
                "Review: (1) AI anomaly detection results, (2) Material accounting "
                "status, (3) Recommended action."
            ),
            review_callback=dual_reviewer_callback,
            sla_seconds=1800,
            escalation_chain=["plant_manager", "NRC_resident_inspector"],
            tradeoff=AIHumanTradeoff(
                step_name="Dual Operator Review",
                performer="human",
                rationale="NRC requires human oversight on all safety-critical decisions",
                risk_level="critical",
                regulatory_reference="10 CFR 73.55 (Physical Protection)",
            ),
        ),
        ProcessingStep(
            name="Material Accounting Check",
            processor=material_accounting_check,
            tradeoff=AIHumanTradeoff(
                step_name="Material Accounting Check",
                performer="ai",
                rationale="Automated cross-reference with inventory database",
                risk_level="medium",
                regulatory_reference="10 CFR 74.19",
                fallback_plan="Manual inventory count by two independent operators",
            ),
        ),
        ProcessingStep(
            name="Action Decision",
            processor=action_decision,
        ),
    ]

    # Create engine with NRC compliance profile
    engine = WorkflowEngine(
        workflow_name="Centrifuge Anomaly Response",
        steps=steps,
        audit_trail=audit_trail,
        compliance_profile=NRC_PROFILE,
        workflow_id="NUC-INCIDENT-2026-0042",
    )

    # Pre-flight compliance check
    print("\n--- Pre-flight Compliance Validation ---")
    gaps = engine.validate_workflow()
    if gaps:
        print("COMPLIANCE GAPS DETECTED:")
        for gap in gaps:
            print(f"  ! {gap}")
    else:
        print("  All NRC compliance requirements satisfied.")

    # Run the pipeline
    print("\n--- Executing Pipeline ---")
    sensor_data = {
        "unit_id": "CEN-042",
        "vibration_hz": 847.3,
        "temperature_c": 62.1,
        "pressure_kpa": 101.2,
        "rpm": 50200,
        "timestamp": "2026-03-26T14:30:00Z",
    }

    result = engine.run(initial_input=sensor_data)

    print(f"\n  Workflow ID:  {result.workflow_id}")
    print(f"  Run ID:       {result.run_id}")
    print(f"  Status:       {result.state.value}")
    print(f"  Completed:    {result.completed}")
    print(f"  Audit Records: {len(result.audit_trail)}")

    # Verify chain integrity
    print("\n--- Hash Chain Integrity ---")
    valid, errors = result.audit_trail.verify_chain()
    print(f"  Chain Valid: {valid}")
    if errors:
        for err in errors:
            print(f"  ERROR: {err}")

    # Verify signatures
    print("\n--- Signature Verification ---")
    sig_valid, sig_errors = result.audit_trail.verify_signatures(keyring)
    print(f"  Signatures Valid: {sig_valid}")
    if sig_errors:
        for err in sig_errors:
            print(f"  ERROR: {err}")

    # Generate inspector report
    print("\n--- NRC Inspector Report ---")
    report_gen = ReportGenerator(result.audit_trail, NRC_PROFILE)
    inspector_report = report_gen.generate_inspector_report()
    print(inspector_report)

    # Show audit storage locations
    print(f"\n--- Audit Storage ---")
    print(f"  Primary:   {os.path.join(tmpdir, 'primary_audit.jsonl')}")
    print(f"  Secondary: {os.path.join(tmpdir, 'secondary_audit.jsonl')}")
    print(f"  Records in primary:   {primary_backend.count()}")
    print(f"  Records in secondary: {secondary_backend.count()}")

    # Show context (final state)
    print("\n--- Final Decision ---")
    print(f"  Action: {result.context.get('final_action', 'N/A')}")
    print(f"  Anomaly: {result.context.get('anomaly_type', 'N/A')}")
    print(f"  Unit: {result.context.get('affected_unit', 'N/A')}")

    print("\n" + "=" * 72)
    print("Pipeline complete. All records signed and stored in dual backends.")
    print("=" * 72)


if __name__ == "__main__":
    main()
