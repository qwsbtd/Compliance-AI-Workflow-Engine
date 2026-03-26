"""Tests for workflow step types."""

import pytest

from compliance_engine.steps import (
    AIHumanTradeoff,
    ApprovalRequirement,
    ExplainabilityInfo,
    FairnessConstraint,
    FairnessGate,
    HumanCheckpoint,
    HumanDecision,
    ModelStep,
    ProcessingStep,
    StepResult,
    StepStatus,
)


# ─── Test Fixtures ────────────────────────────────────────────────────────────


class SimpleModel(ModelStep):
    """Test model that returns a fixed prediction."""

    def __init__(self, confidence=0.9):
        super().__init__(
            name="Test Model",
            model_id="test-model",
            model_version="1.0",
        )
        self._confidence = confidence

    def predict(self, input_data, context):
        return {"result": "positive", "decision": "approve"}, self._confidence


class LowConfidenceModel(ModelStep):
    """Test model with confidence below threshold."""

    def __init__(self):
        super().__init__(
            name="Low Confidence Model",
            model_id="test-low",
            model_version="1.0",
            confidence_threshold=0.8,
        )

    def predict(self, input_data, context):
        return {"result": "uncertain"}, 0.5


# ─── ModelStep Tests ──────────────────────────────────────────────────────────


class TestModelStep:
    def test_execute_returns_completed(self):
        model = SimpleModel()
        result = model.execute({"x": 1}, {})
        assert result.status == StepStatus.COMPLETED
        assert result.confidence == 0.9

    def test_model_info_captured(self):
        model = SimpleModel()
        result = model.execute({"x": 1}, {})
        assert result.model_info is not None
        assert result.model_info.model_id == "test-model"
        assert result.model_info.model_version == "1.0"
        assert result.model_info.input_hash != ""

    def test_input_hash_deterministic(self):
        model = SimpleModel()
        r1 = model.execute({"x": 1}, {})
        r2 = model.execute({"x": 1}, {})
        assert r1.model_info.input_hash == r2.model_info.input_hash

    def test_input_hash_changes_with_input(self):
        model = SimpleModel()
        r1 = model.execute({"x": 1}, {})
        r2 = model.execute({"x": 2}, {})
        assert r1.model_info.input_hash != r2.model_info.input_hash

    def test_explainability_captured(self):
        model = SimpleModel()
        result = model.execute({"x": 1}, {})
        assert result.explainability is not None
        assert result.explainability.method == "default"

    def test_low_confidence_triggers_escalation(self):
        model = LowConfidenceModel()
        result = model.execute({"x": 1}, {})
        assert result.escalate is True
        assert "0.50" in result.escalation_reason
        assert "0.80" in result.escalation_reason

    def test_step_type_is_model(self):
        model = SimpleModel()
        assert model.step_type == "model"


# ─── HumanCheckpoint Tests ───────────────────────────────────────────────────


class TestHumanCheckpoint:
    def test_without_callback_returns_awaiting(self):
        cp = HumanCheckpoint(name="Review")
        result = cp.execute({}, {})
        assert result.status == StepStatus.AWAITING_APPROVAL

    def test_with_callback_single_approver(self):
        def approve(ctx, instr):
            return HumanDecision(approver_id="user1", role="admin", approved=True)

        cp = HumanCheckpoint(
            name="Review",
            review_callback=approve,
            approval_requirement=ApprovalRequirement(min_approvals=1),
        )
        result = cp.execute({}, {})
        assert result.status == StepStatus.COMPLETED
        assert result.decision == "approved"

    def test_with_callback_rejection(self):
        def reject(ctx, instr):
            return HumanDecision(approver_id="user1", role="admin", approved=False, comment="No")

        cp = HumanCheckpoint(
            name="Review",
            review_callback=reject,
            approval_requirement=ApprovalRequirement(min_approvals=1),
        )
        result = cp.execute({}, {})
        assert result.decision == "rejected"

    def test_multi_party_approval(self):
        calls = iter([
            HumanDecision(approver_id="u1", role="supervisor", approved=True),
            HumanDecision(approver_id="u2", role="engineer", approved=True),
        ])

        def callback(ctx, instr):
            return next(calls)

        cp = HumanCheckpoint(
            name="Dual Review",
            review_callback=callback,
            approval_requirement=ApprovalRequirement(
                min_approvals=2,
                require_different_people=True,
            ),
        )
        result = cp.execute({}, {})
        assert result.status == StepStatus.COMPLETED
        assert result.decision == "approved"

    def test_submit_approval_duplicate_person_rejected(self):
        cp = HumanCheckpoint(
            name="Review",
            approval_requirement=ApprovalRequirement(
                min_approvals=2,
                require_different_people=True,
            ),
        )
        d1 = HumanDecision(approver_id="same", role="r1", approved=True)
        d2 = HumanDecision(approver_id="same", role="r2", approved=True)
        cp.submit_approval(d1)
        met, msg = cp.submit_approval(d2)
        assert met is False
        assert "already approved" in msg

    def test_submit_approval_duplicate_role_rejected(self):
        cp = HumanCheckpoint(
            name="Review",
            approval_requirement=ApprovalRequirement(
                min_approvals=2,
                require_different_roles=True,
            ),
        )
        d1 = HumanDecision(approver_id="u1", role="same_role", approved=True)
        d2 = HumanDecision(approver_id="u2", role="same_role", approved=True)
        cp.submit_approval(d1)
        met, msg = cp.submit_approval(d2)
        assert met is False
        assert "already represented" in msg

    def test_approval_status(self):
        cp = HumanCheckpoint(
            name="Review",
            approval_requirement=ApprovalRequirement(min_approvals=2),
        )
        cp.submit_approval(HumanDecision(approver_id="u1", role="r1", approved=True))
        status = cp.approval_status
        assert status["approvals"] == 1
        assert status["required"] == 2
        assert status["quorum_met"] is False

    def test_reset_clears_approvals(self):
        cp = HumanCheckpoint(name="Review")
        cp.submit_approval(HumanDecision(approver_id="u1", role="r1", approved=True))
        cp.reset()
        assert cp.approval_status["approvals"] == 0


# ─── FairnessGate Tests ──────────────────────────────────────────────────────


class TestFairnessGate:
    def test_passes_fair_predictions(self):
        gate = FairnessGate(
            name="Equity",
            constraints=[
                FairnessConstraint(
                    metric="demographic_parity",
                    protected_attribute="group",
                    threshold=0.8,
                ),
            ],
        )
        # Equal rates across groups = fair
        predictions = [
            {"outcome": True}, {"outcome": False},  # group A: 50%
            {"outcome": True}, {"outcome": False},  # group B: 50%
        ]
        groups = {"group": ["A", "A", "B", "B"]}
        result = gate.execute({"predictions": predictions, "groups": groups}, {})
        assert result.status == StepStatus.COMPLETED
        assert result.decision == "passed"

    def test_blocks_unfair_predictions(self):
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
        # Very different rates: A=100%, B=0%
        predictions = [
            {"outcome": True}, {"outcome": True},   # group A: 100%
            {"outcome": False}, {"outcome": False},  # group B: 0%
        ]
        groups = {"group": ["A", "A", "B", "B"]}
        result = gate.execute({"predictions": predictions, "groups": groups}, {})
        assert result.status == StepStatus.BLOCKED
        assert result.decision == "blocked_bias_detected"

    def test_non_blocking_constraint_allows_continuation(self):
        gate = FairnessGate(
            name="Equity",
            constraints=[
                FairnessConstraint(
                    metric="demographic_parity",
                    protected_attribute="group",
                    threshold=0.8,
                    block_on_failure=False,
                ),
            ],
        )
        predictions = [
            {"outcome": True}, {"outcome": True},
            {"outcome": False}, {"outcome": False},
        ]
        groups = {"group": ["A", "A", "B", "B"]}
        result = gate.execute({"predictions": predictions, "groups": groups}, {})
        assert result.status == StepStatus.COMPLETED

    def test_step_type_is_fairness_gate(self):
        gate = FairnessGate(name="Equity", constraints=[])
        assert gate.step_type == "fairness_gate"


# ─── ProcessingStep Tests ────────────────────────────────────────────────────


class TestProcessingStep:
    def test_with_processor(self):
        def my_proc(input_data, context):
            return StepResult(status=StepStatus.COMPLETED, output={"x": 1})

        step = ProcessingStep(name="Proc", processor=my_proc)
        result = step.execute(None, {})
        assert result.output == {"x": 1}

    def test_without_processor_raises(self):
        step = ProcessingStep(name="Empty")
        with pytest.raises(NotImplementedError):
            step.execute(None, {})
