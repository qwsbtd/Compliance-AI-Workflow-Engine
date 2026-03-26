"""
AI vs. human tradeoff documentation and analysis.

Provides structured documentation of why AI was chosen over a human
(or vice versa) for each workflow step, with regulatory mapping.
This is a key deliverable for regulators — it answers "why did you
use AI here and not a human?" with a defensible rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .regulatory import ComplianceProfile
from .steps import AIHumanTradeoff, WorkflowStep

if TYPE_CHECKING:
    pass


class TradeoffAnalyzer:
    """Analyzes and documents AI/human tradeoff decisions in a workflow.

    Generates compliance-focused analysis including risk assessment,
    regulatory mapping, and recommendations.
    """

    def __init__(self, profile: ComplianceProfile | None = None) -> None:
        self._profile = profile

    def analyze(
        self,
        steps: list[WorkflowStep],
        considerations: dict | None = None,
    ) -> dict:
        """Analyze tradeoffs across all workflow steps.

        Returns a dict with:
        - step_analysis: per-step tradeoff assessment
        - compliance_gaps: unmet regulatory requirements
        - risk_summary: overall risk assessment
        - recommendations: suggested improvements
        """
        step_analysis = []
        recommendations = []
        risk_levels = []

        for step in steps:
            tradeoff = getattr(step, "tradeoff", None)
            analysis = self._analyze_step(step, tradeoff)
            step_analysis.append(analysis)

            if tradeoff:
                risk_levels.append(tradeoff.risk_level)

            # Generate recommendations
            recs = self._generate_recommendations(step, tradeoff)
            recommendations.extend(recs)

        # Compliance gaps
        compliance_gaps = []
        if self._profile:
            compliance_gaps = self._profile.validate_workflow(steps)

        # Risk summary
        risk_summary = self._compute_risk_summary(risk_levels)

        return {
            "step_analysis": step_analysis,
            "compliance_gaps": compliance_gaps,
            "risk_summary": risk_summary,
            "recommendations": recommendations,
            "regulatory_mapping": (
                self._map_steps_to_requirements(steps)
                if self._profile
                else {}
            ),
        }

    def document_tradeoff(
        self, step: WorkflowStep, tradeoff: AIHumanTradeoff
    ) -> dict:
        """Generate a structured tradeoff document for a single step."""
        return {
            "step_name": step.name,
            "step_type": step.step_type,
            "tradeoff": tradeoff.to_dict(),
            "analysis": self._analyze_step(step, tradeoff),
        }

    def _analyze_step(
        self, step: WorkflowStep, tradeoff: AIHumanTradeoff | None
    ) -> dict:
        """Analyze a single step's tradeoff."""
        analysis = {
            "step_name": step.name,
            "step_type": step.step_type,
            "has_tradeoff_doc": tradeoff is not None,
        }

        if tradeoff:
            analysis.update({
                "performer": tradeoff.performer,
                "risk_level": tradeoff.risk_level,
                "rationale": tradeoff.rationale,
                "regulatory_reference": tradeoff.regulatory_reference,
                "fallback_plan": tradeoff.fallback_plan,
                "review_frequency": tradeoff.review_frequency,
            })
        else:
            analysis["warning"] = (
                f"Step '{step.name}' has no tradeoff documentation. "
                f"Consider documenting the AI/human decision rationale."
            )

        return analysis

    def _generate_recommendations(
        self, step: WorkflowStep, tradeoff: AIHumanTradeoff | None
    ) -> list[str]:
        """Generate recommendations for a step."""
        recs = []

        if not tradeoff:
            recs.append(
                f"Document AI/human tradeoff rationale for step '{step.name}'"
            )
            return recs

        if tradeoff.risk_level in ("high", "critical") and tradeoff.performer == "ai":
            if not tradeoff.fallback_plan:
                recs.append(
                    f"Step '{step.name}' is high-risk AI — add a fallback plan"
                )
            if not tradeoff.review_frequency:
                recs.append(
                    f"Step '{step.name}' is high-risk AI — specify review frequency"
                )

        if tradeoff.performer == "ai" and not tradeoff.regulatory_reference:
            if self._profile:
                recs.append(
                    f"Step '{step.name}' uses AI but has no regulatory reference. "
                    f"Consider mapping to {', '.join(self._profile.framework_ids)}"
                )

        return recs

    def _compute_risk_summary(self, risk_levels: list[str]) -> dict:
        """Compute overall risk summary."""
        if not risk_levels:
            return {"overall": "unknown", "distribution": {}}

        distribution = {}
        for level in risk_levels:
            distribution[level] = distribution.get(level, 0) + 1

        # Overall = highest risk present
        priority = ["critical", "high", "medium", "low"]
        overall = "low"
        for level in priority:
            if level in distribution:
                overall = level
                break

        return {"overall": overall, "distribution": distribution}

    def _map_steps_to_requirements(self, steps: list[WorkflowStep]) -> dict:
        """Map steps to regulatory requirements they help satisfy."""
        if not self._profile:
            return {}

        mapping = {}
        for step in steps:
            satisfied = []
            for req in self._profile.requirements:
                if self._step_satisfies_requirement(step, req):
                    satisfied.append(req.req_id)
            if satisfied:
                mapping[step.name] = satisfied

        return mapping

    def _step_satisfies_requirement(
        self, step: WorkflowStep, req: "RegulatoryRequirement"
    ) -> bool:
        """Heuristic check if a step helps satisfy a requirement."""
        category = req.category

        if category == "human_oversight" and step.step_type == "human_checkpoint":
            return True
        if category == "fairness" and step.step_type == "fairness_gate":
            return True
        if category == "audit":
            return True  # All steps generate audit records
        if category == "signature":
            return True  # Signatures are trail-level, all steps are signed
        if category == "explainability" and step.step_type == "model":
            return getattr(step, "explain", False)

        return False
