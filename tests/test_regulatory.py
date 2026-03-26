"""Tests for regulatory compliance profiles."""

import pytest

from compliance_engine.regulatory import (
    EU_AI_ACT_HIGH_RISK,
    FDA_PROFILE,
    HIPAA_PROFILE,
    NRC_PROFILE,
    NIST_AI_RMF_PROFILE,
    ComplianceProfile,
    RegulatoryFramework,
    RegulatoryRequirement,
)
from compliance_engine.steps import (
    ApprovalRequirement,
    HumanCheckpoint,
    ModelStep,
    ProcessingStep,
    StepResult,
    StepStatus,
)


class SimpleModel(ModelStep):
    def __init__(self):
        super().__init__(name="AI", model_id="m1", model_version="1.0")

    def predict(self, input_data, context):
        return {"x": 1}, 0.9


def ok_step(i, c):
    return StepResult(status=StepStatus.COMPLETED, output={})


class TestComplianceProfile:
    def test_nrc_requires_human_after_model(self):
        steps = [SimpleModel()]  # No human checkpoint after
        gaps = NRC_PROFILE.validate_workflow(steps)
        assert len(gaps) > 0
        assert any("HumanCheckpoint" in g for g in gaps)

    def test_nrc_passes_with_human_after_model(self):
        from compliance_engine.steps import HumanDecision

        def cb(ctx, instr):
            return HumanDecision(approver_id="u1", role="r1", approved=True)

        steps = [
            SimpleModel(),
            HumanCheckpoint(
                name="Review",
                review_callback=cb,
                approval_requirement=ApprovalRequirement(
                    min_approvals=2,
                    eligible_approvers=["u1", "u2"],
                ),
            ),
        ]
        gaps = NRC_PROFILE.validate_workflow(steps)
        assert len(gaps) == 0

    def test_nrc_min_approvers_check(self):
        from compliance_engine.steps import HumanDecision

        def cb(ctx, instr):
            return HumanDecision(approver_id="u1", role="r1", approved=True)

        steps = [
            SimpleModel(),
            HumanCheckpoint(
                name="Review",
                review_callback=cb,
                approval_requirement=ApprovalRequirement(min_approvals=1),  # NRC needs 2
            ),
        ]
        gaps = NRC_PROFILE.validate_workflow(steps)
        assert any("min_approvals" in g for g in gaps)

    def test_hipaa_requires_phi_redaction(self):
        assert HIPAA_PROFILE.require_phi_redaction is True

    def test_fda_requires_signatures(self):
        assert FDA_PROFILE.require_signatures is True

    def test_nist_requires_explainability(self):
        assert NIST_AI_RMF_PROFILE.require_explainability is True

    def test_eu_ai_act_requires_fairness(self):
        assert EU_AI_ACT_HIGH_RISK.require_fairness_check is True


class TestRegulatoryFramework:
    def test_get_profile(self):
        profile = RegulatoryFramework.get("NRC Nuclear Safety")
        assert profile.name == "NRC Nuclear Safety"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            RegulatoryFramework.get("Nonexistent")

    def test_list_profiles(self):
        profiles = RegulatoryFramework.list_profiles()
        assert "NRC Nuclear Safety" in profiles
        assert "FDA 21 CFR Part 11" in profiles

    def test_compose_strictest_wins(self):
        composed = RegulatoryFramework.compose(
            "FDA 21 CFR Part 11", "HIPAA Privacy & Security"
        )
        # HIPAA requires PHI redaction, FDA doesn't
        assert composed.require_phi_redaction is True
        # Both require signatures
        assert composed.require_signatures is True
        # Requirements are unioned
        fda_reqs = {r.req_id for r in FDA_PROFILE.requirements}
        hipaa_reqs = {r.req_id for r in HIPAA_PROFILE.requirements}
        composed_reqs = {r.req_id for r in composed.requirements}
        assert fda_reqs.issubset(composed_reqs)
        assert hipaa_reqs.issubset(composed_reqs)

    def test_compose_min_approvers_takes_max(self):
        composed = RegulatoryFramework.compose(
            "NRC Nuclear Safety", "FDA 21 CFR Part 11"
        )
        # NRC requires 2, FDA requires 1 → composed should be 2
        assert composed.min_approvers == 2

    def test_compose_requires_at_least_one(self):
        with pytest.raises(ValueError):
            RegulatoryFramework.compose()

    def test_register_custom_profile(self):
        custom = ComplianceProfile(
            name="Custom Test Profile",
            framework_ids=["CUSTOM"],
        )
        RegulatoryFramework.register(custom)
        assert RegulatoryFramework.get("Custom Test Profile") is custom
        # Cleanup
        del RegulatoryFramework._profiles["Custom Test Profile"]
