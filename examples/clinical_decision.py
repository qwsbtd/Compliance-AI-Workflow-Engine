#!/usr/bin/env python3
"""
ICU Sepsis Risk Scoring Pipeline

Demonstrates the Compliance AI Workflow Engine in a hospital setting
regulated by FDA 21 CFR Part 11 and HIPAA. AI assists with sepsis risk
scoring — analyzing patient vitals and lab results to produce a risk
score and recommended intervention. Clinician must review before any
treatment change.

Features demonstrated:
- ModelStep with explainability (feature importance for clinicians)
- FairnessGate blocking biased models (demographic parity, disparate impact)
- PHI redaction in audit trail (HIPAA compliance)
- Composed FDA + HIPAA compliance profile
- Electronic signatures (FDA 21 CFR Part 11)
- Full compliance report generation

Run: python examples/clinical_decision.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compliance_engine import (
    AIHumanTradeoff,
    ApprovalRequirement,
    AuditTrail,
    ExplainabilityInfo,
    FairnessConstraint,
    FairnessGate,
    HMACSigner,
    HumanCheckpoint,
    HumanDecision,
    KeyRing,
    ModelStep,
    ProcessingStep,
    RegulatoryFramework,
    ReportFormat,
    ReportGenerator,
    SQLiteBackend,
    StepResult,
    StepStatus,
    WorkflowEngine,
)


# ─── Custom Steps ─────────────────────────────────────────────────────────────


class SepsisRiskModel(ModelStep):
    """AI model that scores sepsis risk from patient vitals and labs."""

    def __init__(self):
        super().__init__(
            name="Sepsis Risk Scoring",
            model_id="sepsis-risk-scorer",
            model_version="2.1.0",
            model_config={
                "algorithm": "gradient_boosted_trees",
                "features": ["heart_rate", "temp", "wbc", "lactate", "bp_systolic", "respiratory_rate"],
                "threshold_high": 0.7,
                "threshold_medium": 0.4,
            },
            model_hash="b2c3d4e5f6a789012345678abcdef0123456789abcdef0123456789abcdef02",
            framework="xgboost",
            explain=True,
            confidence_threshold=0.80,
            tradeoff=AIHumanTradeoff(
                step_name="Sepsis Risk Scoring",
                performer="ai",
                rationale=(
                    "Early sepsis detection requires continuous monitoring of multiple "
                    "biomarkers. AI screening reduces time-to-detection from hours to "
                    "minutes. All positive screens require attending physician review."
                ),
                risk_level="high",
                regulatory_reference="FDA 21 CFR Part 11 (Electronic Records)",
                fallback_plan="Manual qSOFA scoring by bedside nurse every 4 hours",
                review_frequency="Monthly model performance review by clinical informatics",
            ),
        )

    def predict(self, input_data, context):
        """Simulate sepsis risk prediction."""
        patient_data = input_data or context.get("patient_vitals", {})

        # Simulated prediction
        risk_score = 0.78
        risk_level = "high"
        confidence = 0.87

        prediction = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_actions": [
                "Blood cultures x2 before antibiotics",
                "Serum lactate level",
                "Broad-spectrum antibiotics within 1 hour",
                "30 mL/kg crystalloid for hypotension",
            ],
            "decision": f"Sepsis risk: {risk_level} (score: {risk_score:.2f})",
        }

        return prediction, confidence

    def generate_explanation(self, input_data, prediction, context):
        """Explain which vitals drove the risk score."""
        return ExplainabilityInfo(
            method="feature_importance",
            summary=(
                "Elevated sepsis risk driven primarily by: elevated lactate (4.2 mmol/L, "
                "normal <2.0), tachycardia (HR 118), and elevated WBC (18.2 x10^9/L). "
                "Blood pressure borderline low (92/58). Temperature mildly elevated (38.4C)."
            ),
            feature_importances={
                "lactate": 0.35,
                "heart_rate": 0.22,
                "wbc": 0.18,
                "bp_systolic": 0.12,
                "temperature": 0.08,
                "respiratory_rate": 0.05,
            },
            reasoning_trace=[
                "1. Patient vitals ingested: HR 118, Temp 38.4C, BP 92/58",
                "2. Lab results: WBC 18.2, Lactate 4.2, Procalcitonin 2.8",
                "3. qSOFA score: 2/3 (altered mentation + hypotension)",
                "4. Model feature analysis: lactate and HR are primary drivers",
                "5. Risk classification: HIGH (score 0.78, threshold 0.70)",
                "6. Recommendation: Sepsis bundle activation within 1 hour",
            ],
            confidence=0.87,
            alternatives_considered=[
                {"diagnosis": "SIRS without infection", "probability": 0.15},
                {"diagnosis": "Dehydration", "probability": 0.05},
                {"diagnosis": "Medication reaction", "probability": 0.02},
            ],
        )


def attending_physician_callback(context, instructions):
    """Simulated attending physician review."""
    return HumanDecision(
        approver_id="DR-PATEL",
        role="attending_physician",
        approved=True,
        comment=(
            "Agree with sepsis screen. Lactate and clinical picture consistent. "
            "Initiate sepsis bundle. Order blood cultures and start vancomycin + "
            "piperacillin-tazobactam empirically."
        ),
    )


def treatment_plan_step(input_data, context):
    """Generate treatment plan from approved recommendation."""
    risk_level = context.get("risk_level", "unknown")
    actions = context.get("recommended_actions", [])

    return StepResult(
        status=StepStatus.COMPLETED,
        output={
            "treatment_plan": {
                "risk_level": risk_level,
                "interventions": actions,
                "monitoring": "Continuous vitals q15min, repeat lactate in 6 hours",
                "reassessment": "4 hours post-intervention",
            },
        },
        decision=f"Treatment plan generated for {risk_level}-risk sepsis",
    )


# ─── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    print("=" * 72)
    print("ICU SEPSIS RISK SCORING PIPELINE")
    print("FDA 21 CFR Part 11 + HIPAA Regulated")
    print("=" * 72)

    # Compose FDA + HIPAA profile (strictest-wins)
    profile = RegulatoryFramework.compose("FDA 21 CFR Part 11", "HIPAA Privacy & Security")
    print(f"\n  Compliance Profile: {profile.name}")
    print(f"  Frameworks: {', '.join(profile.framework_ids)}")
    print(f"  PHI Redaction Required: {profile.require_phi_redaction}")
    print(f"  Signatures Required: {profile.require_signatures}")

    # Set up crypto
    keyring = KeyRing()
    system_key = keyring.generate_key(
        "hospital-ehr-system",
        metadata={"facility": "Metro General Hospital", "department": "ICU"},
    )
    signer = HMACSigner()

    # Set up audit storage
    tmpdir = tempfile.mkdtemp(prefix="clinical_audit_")
    db_path = os.path.join(tmpdir, "clinical_audit.db")
    backend = SQLiteBackend(db_path)

    # PHI fields to redact from audit trail
    phi_fields = {"patient_name", "mrn", "dob", "ssn", "address", "phone"}

    audit_trail = AuditTrail(
        signer=signer,
        signing_key=system_key,
        backend=backend,
        phi_fields=phi_fields,
        workflow_id="CLIN-SEPSIS-2026-0318",
    )

    # Build fairness check data (recent batch for equity validation)
    # This simulates a batch of recent predictions for fairness analysis
    recent_predictions = [
        # Group A (e.g., demographic group 1) — 60% positive rate
        {"outcome": True, "actual": True},
        {"outcome": True, "actual": True},
        {"outcome": True, "actual": False},
        {"outcome": False, "actual": False},
        {"outcome": False, "actual": True},
        # Group B (e.g., demographic group 2) — 50% positive rate
        {"outcome": True, "actual": True},
        {"outcome": True, "actual": True},
        {"outcome": True, "actual": False},
        {"outcome": False, "actual": False},
        {"outcome": False, "actual": False},
    ]
    race_labels = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    # Build the pipeline
    steps = [
        SepsisRiskModel(),
        FairnessGate(
            name="Equity Check",
            constraints=[
                FairnessConstraint(
                    metric="demographic_parity",
                    protected_attribute="race",
                    threshold=0.8,
                    block_on_failure=True,
                ),
                FairnessConstraint(
                    metric="disparate_impact",
                    protected_attribute="race",
                    threshold=0.8,
                    block_on_failure=True,
                ),
            ],
        ),
        HumanCheckpoint(
            name="Attending Physician Review",
            approval_requirement=ApprovalRequirement(
                min_approvals=1,
                eligible_approvers=["DR-PATEL", "DR-JOHNSON", "DR-LEE"],
            ),
            instructions=(
                "Review AI sepsis risk assessment. Examine: (1) Risk score and "
                "confidence level, (2) Feature importance — which vitals drove "
                "the score, (3) Recommended interventions. Approve to activate "
                "sepsis bundle."
            ),
            review_callback=attending_physician_callback,
            sla_seconds=900,  # 15 minutes for high-risk
            escalation_chain=["charge_nurse", "department_chief", "CMO"],
            tradeoff=AIHumanTradeoff(
                step_name="Attending Physician Review",
                performer="human",
                rationale="Clinical judgment required for treatment decisions",
                risk_level="critical",
                regulatory_reference="FDA 21 CFR Part 11",
            ),
        ),
        ProcessingStep(
            name="Treatment Plan Generation",
            processor=treatment_plan_step,
        ),
    ]

    engine = WorkflowEngine(
        workflow_name="ICU Sepsis Risk Assessment",
        steps=steps,
        audit_trail=audit_trail,
        compliance_profile=profile,
        workflow_id="CLIN-SEPSIS-2026-0318",
    )

    # Pre-flight compliance check
    print("\n--- Pre-flight Compliance Validation ---")
    gaps = engine.validate_workflow()
    if gaps:
        print("COMPLIANCE GAPS:")
        for gap in gaps:
            print(f"  ! {gap}")
    else:
        print("  All FDA + HIPAA requirements satisfied.")

    # Patient data (contains PHI that will be redacted in audit)
    patient_data = {
        "patient_name": "Jane Doe",
        "mrn": "MRN-123456",
        "dob": "1965-08-15",
        "ssn": "123-45-6789",
        "heart_rate": 118,
        "temperature": 38.4,
        "bp_systolic": 92,
        "bp_diastolic": 58,
        "respiratory_rate": 24,
        "wbc": 18.2,
        "lactate": 4.2,
        "procalcitonin": 2.8,
    }

    # Store fairness data in context for the FairnessGate
    initial_context = {
        "patient_vitals": patient_data,
        "predictions": recent_predictions,
        "groups": {"race": race_labels},
    }

    # Prepare fairness input for the gate
    # The FairnessGate expects input_data with predictions and groups
    fairness_input = {
        "predictions": recent_predictions,
        "groups": {"race": race_labels},
    }

    print("\n--- Executing Pipeline ---")
    result = engine.run(
        initial_input=fairness_input,
        context=initial_context,
    )

    print(f"\n  Workflow ID:   {result.workflow_id}")
    print(f"  Status:        {result.state.value}")
    print(f"  Completed:     {result.completed}")
    print(f"  Audit Records: {len(result.audit_trail)}")

    # Show PHI redaction in action
    print("\n--- PHI Redaction Verification ---")
    for record in result.audit_trail.records:
        for field_name in phi_fields:
            input_val = record.input_summary.get(field_name)
            output_val = record.output_summary.get(field_name)
            if input_val == "[REDACTED-PHI]":
                print(f"  {record.step_name}: '{field_name}' -> [REDACTED-PHI]")

    # Verify chain integrity
    print("\n--- Hash Chain Integrity ---")
    valid, errors = result.audit_trail.verify_chain()
    print(f"  Chain Valid: {valid}")

    # Verify signatures
    sig_valid, sig_errors = result.audit_trail.verify_signatures(keyring)
    print(f"  Signatures Valid: {sig_valid}")

    # Generate full compliance report
    print("\n--- FDA/HIPAA Compliance Report ---")
    report_gen = ReportGenerator(result.audit_trail, profile)
    report = report_gen.generate_compliance_report()
    print(report)

    # Show treatment plan
    print("\n--- Treatment Plan ---")
    plan = result.context.get("treatment_plan", {})
    if plan:
        print(f"  Risk Level: {plan.get('risk_level', 'N/A')}")
        print(f"  Monitoring: {plan.get('monitoring', 'N/A')}")
        print(f"  Reassessment: {plan.get('reassessment', 'N/A')}")
        print("  Interventions:")
        for intervention in plan.get("interventions", []):
            print(f"    - {intervention}")

    print(f"\n  Audit DB: {db_path}")
    print(f"  Records in DB: {backend.count()}")

    print("\n" + "=" * 72)
    print("Pipeline complete. PHI redacted. All records signed per FDA 21 CFR Part 11.")
    print("=" * 72)


if __name__ == "__main__":
    main()
