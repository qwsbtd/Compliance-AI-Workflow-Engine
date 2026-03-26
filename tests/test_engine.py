"""Tests for the WorkflowEngine orchestrator."""

import pytest

from compliance_engine.audit import AuditTrail
from compliance_engine.engine import WorkflowEngine, WorkflowResult
from compliance_engine.persistence import WorkflowState
from compliance_engine.regulatory import NRC_PROFILE
from compliance_engine.steps import (
    ApprovalRequirement,
    FairnessConstraint,
    FairnessGate,
    HumanCheckpoint,
    HumanDecision,
    ModelStep,
    ProcessingStep,
    StepResult,
    StepStatus,
)


class SimpleModel(ModelStep):
    def __init__(self, confidence=0.9):
        super().__init__(name="AI Step", model_id="test", model_version="1.0")
        self._confidence = confidence

    def predict(self, input_data, context):
        return {"prediction": "ok", "decision": "approve"}, self._confidence


def ok_step(input_data, context):
    return StepResult(status=StepStatus.COMPLETED, output={"done": True}, decision="ok")


def approve_callback(ctx, instr):
    return HumanDecision(approver_id="u1", role="admin", approved=True)


def reject_callback(ctx, instr):
    return HumanDecision(approver_id="u1", role="admin", approved=False, comment="No")


class TestWorkflowEngine:
    def test_simple_pipeline(self):
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[ProcessingStep(name="Step1", processor=ok_step)],
        )
        result = engine.run()
        assert result.completed is True
        assert result.state == WorkflowState.COMPLETED
        assert len(result.audit_trail) > 0

    def test_context_flows_between_steps(self):
        def step_a(input_data, ctx):
            return StepResult(status=StepStatus.COMPLETED, output={"a": 1})

        def step_b(input_data, ctx):
            assert ctx.get("a") == 1
            return StepResult(status=StepStatus.COMPLETED, output={"b": 2})

        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[
                ProcessingStep(name="A", processor=step_a),
                ProcessingStep(name="B", processor=step_b),
            ],
        )
        result = engine.run()
        assert result.completed is True
        assert result.context.get("a") == 1
        assert result.context.get("b") == 2

    def test_human_checkpoint_approved(self):
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[
                ProcessingStep(name="Before", processor=ok_step),
                HumanCheckpoint(
                    name="Review",
                    review_callback=approve_callback,
                    approval_requirement=ApprovalRequirement(min_approvals=1),
                ),
                ProcessingStep(name="After", processor=ok_step),
            ],
        )
        result = engine.run()
        assert result.completed is True

    def test_human_checkpoint_rejected_halts(self):
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[
                HumanCheckpoint(
                    name="Review",
                    review_callback=reject_callback,
                    approval_requirement=ApprovalRequirement(min_approvals=1),
                ),
                ProcessingStep(name="Should Not Run", processor=ok_step),
            ],
        )
        result = engine.run()
        assert result.completed is False
        assert result.state == WorkflowState.FAILED

    def test_fairness_gate_blocks_pipeline(self):
        gate = FairnessGate(
            name="Equity",
            constraints=[
                FairnessConstraint(
                    metric="disparate_impact",
                    protected_attribute="group",
                    threshold=0.8,
                    block_on_failure=True,
                ),
            ],
        )
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[gate],
        )
        # Very biased predictions
        result = engine.run(
            initial_input={
                "predictions": [
                    {"outcome": True}, {"outcome": True},
                    {"outcome": False}, {"outcome": False},
                ],
                "groups": {"group": ["A", "A", "B", "B"]},
            }
        )
        assert result.completed is False
        assert result.state == WorkflowState.FAILED

    def test_model_step_in_pipeline(self):
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[SimpleModel()],
        )
        result = engine.run()
        assert result.completed is True
        # Check audit trail has model info
        records = result.audit_trail.get_records(action="step_completed")
        model_records = [r for r in records if r.model_info is not None]
        assert len(model_records) > 0

    def test_escalation(self):
        step = ProcessingStep(
            name="Slow Step",
            processor=ok_step,
            escalation_chain=["manager", "director"],
        )
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[step],
        )
        engine.run()

        event = engine.trigger_escalation("Slow Step", "sla_breach")
        assert event is not None
        assert event.target == "manager"
        assert event.tier == 0

        event2 = engine.trigger_escalation("Slow Step", "still_breached")
        assert event2.target == "director"
        assert event2.tier == 1

        # No more targets
        event3 = engine.trigger_escalation("Slow Step", "again")
        assert event3 is None

    def test_validate_workflow_with_nrc_profile(self):
        # NRC requires human checkpoint after every model step
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[
                SimpleModel(),
                HumanCheckpoint(
                    name="Review",
                    review_callback=approve_callback,
                    approval_requirement=ApprovalRequirement(
                        min_approvals=2,
                        eligible_approvers=["u1", "u2"],
                    ),
                ),
            ],
            compliance_profile=NRC_PROFILE,
        )
        gaps = engine.validate_workflow()
        assert len(gaps) == 0  # Model followed by human checkpoint

    def test_validate_workflow_detects_gaps(self):
        # NRC requires human after model — this workflow lacks it
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[SimpleModel()],
            compliance_profile=NRC_PROFILE,
        )
        gaps = engine.validate_workflow()
        assert len(gaps) > 0
        assert any("HumanCheckpoint" in g for g in gaps)

    def test_audit_trail_passed_through(self):
        """Engine should use the audit trail passed to it, not create a new one."""
        from compliance_engine.crypto import HMACSigner, KeyRing
        keyring = KeyRing()
        key = keyring.generate_key("test")
        signer = HMACSigner()
        trail = AuditTrail(signer=signer, signing_key=key, workflow_id="wf1")

        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[ProcessingStep(name="S", processor=ok_step)],
            audit_trail=trail,
        )
        result = engine.run()

        # Signatures should be present because we passed a signer
        valid, errors = result.audit_trail.verify_signatures(keyring)
        assert valid is True

    def test_step_exception_handled(self):
        def failing_step(input_data, ctx):
            raise ValueError("Something broke")

        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[ProcessingStep(name="Fail", processor=failing_step)],
        )
        result = engine.run()
        assert result.state == WorkflowState.FAILED
        assert result.completed is False

    def test_sla_check(self):
        step = ProcessingStep(name="Timed", processor=ok_step, sla_seconds=3600)
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[step],
        )
        engine.run()
        sla = engine.check_sla("Timed")
        assert sla["has_sla"] is True
        assert sla["breached"] is False

    def test_sla_check_no_sla(self):
        engine = WorkflowEngine(
            workflow_name="Test",
            steps=[ProcessingStep(name="NoSLA", processor=ok_step)],
        )
        engine.run()
        sla = engine.check_sla("NoSLA")
        assert sla["has_sla"] is False
