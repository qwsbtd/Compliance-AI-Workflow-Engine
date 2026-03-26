"""
Workflow step types for compliance-regulated AI pipelines.

Provides: WorkflowStep (ABC), ModelStep (AI with versioning + explainability),
HumanCheckpoint (multi-party approval), FairnessGate (bias detection),
and ProcessingStep (general-purpose logic).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class StepStatus(Enum):
    """Execution status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    AWAITING_APPROVAL = "awaiting_approval"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"


@dataclass
class ModelInfo:
    """Captures model identity for reproducibility.

    Every AI step records which model version, what input, and what
    configuration produced the decision — essential for regulatory
    reproducibility audits.
    """

    model_id: str
    model_version: str
    model_hash: str | None = None
    input_hash: str = ""
    config: dict = field(default_factory=dict)
    framework: str = "custom"

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "input_hash": self.input_hash,
            "config": self.config,
            "framework": self.framework,
        }


@dataclass
class ExplainabilityInfo:
    """Captures reasoning traces for explainability.

    Stores feature importances, reasoning steps, and human-readable
    summaries — required for clinician review and regulatory transparency.
    """

    method: str  # "feature_importance", "shap", "lime", "rule_trace", "reasoning_chain"
    summary: str
    feature_importances: dict[str, float] | None = None
    reasoning_trace: list[str] | None = None
    confidence: float | None = None
    alternatives_considered: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "summary": self.summary,
            "feature_importances": self.feature_importances,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
            "alternatives_considered": self.alternatives_considered,
        }


@dataclass
class AIHumanTradeoff:
    """Documents why AI was chosen over a human (or vice versa) for a step.

    This is the key deliverable for regulators — it answers "why did you
    use AI here and not a human?" with a structured rationale.
    """

    step_name: str
    performer: str  # "ai" or "human"
    rationale: str
    risk_level: str  # "low", "medium", "high", "critical"
    regulatory_reference: str = ""
    fallback_plan: str = ""
    review_frequency: str = ""

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "performer": self.performer,
            "rationale": self.rationale,
            "risk_level": self.risk_level,
            "regulatory_reference": self.regulatory_reference,
            "fallback_plan": self.fallback_plan,
            "review_frequency": self.review_frequency,
        }


@dataclass
class StepResult:
    """Result returned by a workflow step execution."""

    status: StepStatus
    output: dict = field(default_factory=dict)
    decision: str | None = None
    confidence: float | None = None
    model_info: ModelInfo | None = None
    explainability: ExplainabilityInfo | None = None
    fairness_results: list[dict] | None = None
    escalate: bool = False
    escalation_reason: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ApprovalRequirement:
    """Multi-party approval configuration (four-eyes principle).

    Defines the quorum rules: how many approvers are needed, who is
    eligible, and what role constraints apply.
    """

    min_approvals: int = 1
    eligible_approvers: list[str] = field(default_factory=list)
    require_different_people: bool = True
    require_different_roles: bool = False
    approval_timeout_seconds: float | None = None
    escalation_chain: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "min_approvals": self.min_approvals,
            "eligible_approvers": self.eligible_approvers,
            "require_different_people": self.require_different_people,
            "require_different_roles": self.require_different_roles,
            "approval_timeout_seconds": self.approval_timeout_seconds,
            "escalation_chain": self.escalation_chain,
        }


@dataclass
class HumanDecision:
    """A single human approval or rejection."""

    approver_id: str
    role: str
    approved: bool
    comment: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "approver_id": self.approver_id,
            "role": self.role,
            "approved": self.approved,
            "comment": self.comment,
            "timestamp": self.timestamp,
        }


@dataclass
class FairnessConstraint:
    """A single fairness check configuration."""

    metric: str  # "demographic_parity", "disparate_impact", "equalized_odds"
    protected_attribute: str
    threshold: float
    comparison: str = "gte"  # "gte", "lte", "between"
    upper_bound: float | None = None
    block_on_failure: bool = True

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "protected_attribute": self.protected_attribute,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "upper_bound": self.upper_bound,
            "block_on_failure": self.block_on_failure,
        }


# ─── Step Base Classes ────────────────────────────────────────────────────────


class WorkflowStep(ABC):
    """Abstract base class for all workflow steps."""

    def __init__(
        self,
        name: str,
        step_type: str,
        *,
        sla_seconds: float | None = None,
        escalation_chain: list[str] | None = None,
        tradeoff: AIHumanTradeoff | None = None,
    ) -> None:
        self.name = name
        self.step_type = step_type
        self.sla_seconds = sla_seconds
        self.escalation_chain = escalation_chain or []
        self.tradeoff = tradeoff

    @abstractmethod
    def execute(self, input_data: Any, context: dict) -> StepResult:
        """Execute this step. Must be implemented by subclasses."""
        ...


class ModelStep(WorkflowStep):
    """AI/ML model step with automatic versioning and explainability capture.

    Subclass this and implement predict(). The framework automatically:
    - Computes SHA-256 hash of input data
    - Builds ModelInfo for the audit record
    - Calls generate_explanation() if explain=True
    """

    def __init__(
        self,
        name: str,
        *,
        model_id: str,
        model_version: str,
        model_config: dict | None = None,
        model_hash: str | None = None,
        framework: str = "custom",
        explain: bool = True,
        confidence_threshold: float = 0.8,
        sla_seconds: float | None = None,
        escalation_chain: list[str] | None = None,
        tradeoff: AIHumanTradeoff | None = None,
    ) -> None:
        super().__init__(
            name,
            "model",
            sla_seconds=sla_seconds,
            escalation_chain=escalation_chain,
            tradeoff=tradeoff,
        )
        self.model_id = model_id
        self.model_version = model_version
        self.model_config = model_config or {}
        self.model_hash = model_hash
        self.framework = framework
        self.explain = explain
        self.confidence_threshold = confidence_threshold

    def execute(self, input_data: Any, context: dict) -> StepResult:
        """Execute model with versioning and explainability capture."""
        # Compute input hash for reproducibility
        input_canonical = json.dumps(
            input_data, sort_keys=True, separators=(",", ":"), default=str
        )
        input_hash = hashlib.sha256(input_canonical.encode("utf-8")).hexdigest()

        # Run prediction
        prediction, confidence = self.predict(input_data, context)

        # Build model info
        model_info = ModelInfo(
            model_id=self.model_id,
            model_version=self.model_version,
            model_hash=self.model_hash,
            input_hash=input_hash,
            config=self.model_config,
            framework=self.framework,
        )

        # Generate explanation if enabled
        explainability = None
        if self.explain:
            explainability = self.generate_explanation(
                input_data, prediction, context
            )

        # Check confidence threshold for auto-escalation
        escalate = False
        escalation_reason = None
        if confidence is not None and confidence < self.confidence_threshold:
            escalate = True
            escalation_reason = (
                f"Confidence {confidence:.2f} below "
                f"threshold {self.confidence_threshold:.2f}"
            )

        return StepResult(
            status=StepStatus.COMPLETED,
            output=prediction if isinstance(prediction, dict) else {"prediction": prediction},
            decision=str(prediction) if not isinstance(prediction, dict) else prediction.get("decision"),
            confidence=confidence,
            model_info=model_info,
            explainability=explainability,
            escalate=escalate,
            escalation_reason=escalation_reason,
        )

    @abstractmethod
    def predict(self, input_data: Any, context: dict) -> tuple[Any, float | None]:
        """Run the model prediction.

        Returns (prediction_output, confidence_score).
        Confidence is 0.0-1.0, or None if not applicable.
        """
        ...

    def generate_explanation(
        self, input_data: Any, prediction: Any, context: dict
    ) -> ExplainabilityInfo:
        """Generate explanation for the prediction.

        Override to provide model-specific explanations (SHAP, LIME, etc.).
        Default returns a basic summary.
        """
        return ExplainabilityInfo(
            method="default",
            summary=f"Model {self.model_id} v{self.model_version} produced: {prediction}",
        )


class HumanCheckpoint(WorkflowStep):
    """Step requiring human approval before the pipeline continues.

    Supports multi-party approval (four-eyes principle) with configurable
    quorum rules. Can be used with a review_callback for programmatic
    approval (production) or interactive stdin for demos.
    """

    def __init__(
        self,
        name: str,
        *,
        approval_requirement: ApprovalRequirement | None = None,
        instructions: str = "",
        review_callback: Callable[[dict, str], HumanDecision] | None = None,
        sla_seconds: float | None = None,
        escalation_chain: list[str] | None = None,
        tradeoff: AIHumanTradeoff | None = None,
    ) -> None:
        super().__init__(
            name,
            "human_checkpoint",
            sla_seconds=sla_seconds,
            escalation_chain=escalation_chain,
            tradeoff=tradeoff,
        )
        self.approval_requirement = approval_requirement or ApprovalRequirement()
        self.instructions = instructions
        self.review_callback = review_callback
        self._approvals: list[HumanDecision] = []
        self._rejections: list[HumanDecision] = []

    def execute(self, input_data: Any, context: dict) -> StepResult:
        """Execute checkpoint using review_callback if provided.

        If review_callback is set, it's called repeatedly until quorum is
        met. If not set, returns AWAITING_APPROVAL for the engine to handle.
        """
        if self.review_callback:
            # Programmatic approval flow (for demos and automated systems)
            return self._execute_with_callback(input_data, context)
        else:
            # Return awaiting status — engine/caller handles approval collection
            return StepResult(
                status=StepStatus.AWAITING_APPROVAL,
                output={"instructions": self.instructions},
                decision=None,
                metadata={
                    "approval_requirement": self.approval_requirement.to_dict(),
                    "approvals_received": len(self._approvals),
                },
            )

    def _execute_with_callback(self, input_data: Any, context: dict) -> StepResult:
        """Collect approvals via callback until quorum is met or rejected."""
        self.reset()
        required = self.approval_requirement.min_approvals

        for i in range(required):
            decision = self.review_callback(context, self.instructions)
            quorum_met, msg = self.submit_approval(decision)

            if not decision.approved:
                return StepResult(
                    status=StepStatus.COMPLETED,
                    output={
                        "approved": False,
                        "rejections": [r.to_dict() for r in self._rejections],
                    },
                    decision="rejected",
                    metadata={"reason": decision.comment},
                )

            if quorum_met:
                return StepResult(
                    status=StepStatus.COMPLETED,
                    output={
                        "approved": True,
                        "approvals": [a.to_dict() for a in self._approvals],
                    },
                    decision="approved",
                )

        return StepResult(
            status=StepStatus.AWAITING_APPROVAL,
            output={"approvals_so_far": len(self._approvals)},
            decision=None,
        )

    def submit_approval(self, decision: HumanDecision) -> tuple[bool, str]:
        """Submit an approval or rejection.

        Returns (quorum_reached, message).
        Validates: approver eligibility, no duplicate people/roles.
        """
        req = self.approval_requirement

        # Check eligibility
        if req.eligible_approvers and decision.approver_id not in req.eligible_approvers:
            return (False, f"Approver {decision.approver_id!r} not in eligible list")

        # Check duplicate person
        if req.require_different_people:
            existing_ids = {a.approver_id for a in self._approvals}
            if decision.approver_id in existing_ids:
                return (False, f"Approver {decision.approver_id!r} already approved")

        # Check duplicate role
        if req.require_different_roles:
            existing_roles = {a.role for a in self._approvals}
            if decision.role in existing_roles:
                return (False, f"Role {decision.role!r} already represented")

        if decision.approved:
            self._approvals.append(decision)
            quorum_met = len(self._approvals) >= req.min_approvals
            return (quorum_met, f"Approval recorded ({len(self._approvals)}/{req.min_approvals})")
        else:
            self._rejections.append(decision)
            return (False, "Rejection recorded")

    def reset(self) -> None:
        """Clear all approvals and rejections."""
        self._approvals.clear()
        self._rejections.clear()

    @property
    def approval_status(self) -> dict:
        """Current approval state summary."""
        return {
            "approvals": len(self._approvals),
            "rejections": len(self._rejections),
            "required": self.approval_requirement.min_approvals,
            "quorum_met": len(self._approvals) >= self.approval_requirement.min_approvals,
        }


class FairnessGate(WorkflowStep):
    """Gate step that evaluates fairness constraints and can block the pipeline.

    Computes statistical fairness metrics (demographic parity, disparate impact)
    across protected attributes. If any blocking constraint fails, the pipeline
    halts — a biased model is a patient safety issue, not a logging concern.
    """

    def __init__(
        self,
        name: str,
        *,
        constraints: list[FairnessConstraint],
        sla_seconds: float | None = None,
        escalation_chain: list[str] | None = None,
    ) -> None:
        super().__init__(
            name,
            "fairness_gate",
            sla_seconds=sla_seconds,
            escalation_chain=escalation_chain,
        )
        self.constraints = constraints

    def execute(self, input_data: Any, context: dict) -> StepResult:
        """Evaluate fairness constraints.

        input_data should be a dict with:
          - 'predictions': list of dicts with at least an 'outcome' key (0/1 or bool)
          - 'groups': dict mapping protected_attribute -> list of group labels
                      (same length as predictions)
        """
        predictions = input_data.get("predictions", [])
        groups = input_data.get("groups", {})

        results = []
        any_blocked = False

        for constraint in self.constraints:
            metric_value = self._compute_metric(
                constraint.metric,
                predictions,
                groups.get(constraint.protected_attribute, []),
            )

            passed = self._check_threshold(constraint, metric_value)

            result = {
                "metric": constraint.metric,
                "protected_attribute": constraint.protected_attribute,
                "value": round(metric_value, 4) if metric_value is not None else None,
                "threshold": constraint.threshold,
                "passed": passed,
                "block_on_failure": constraint.block_on_failure,
            }
            results.append(result)

            if not passed and constraint.block_on_failure:
                any_blocked = True

        status = StepStatus.BLOCKED if any_blocked else StepStatus.COMPLETED
        decision = "blocked_bias_detected" if any_blocked else "passed"

        return StepResult(
            status=status,
            output={"fairness_results": results},
            decision=decision,
            fairness_results=results,
            metadata={"blocked": any_blocked},
        )

    def _compute_metric(
        self, metric: str, predictions: list[dict], group_labels: list
    ) -> float | None:
        """Compute a fairness metric."""
        if not predictions or not group_labels:
            return None

        if metric == "demographic_parity":
            return self._demographic_parity(predictions, group_labels)
        elif metric == "disparate_impact":
            return self._disparate_impact(predictions, group_labels)
        elif metric == "equalized_odds":
            return self._equalized_odds(predictions, group_labels)
        return None

    def _demographic_parity(
        self, predictions: list[dict], group_labels: list
    ) -> float:
        """Compute demographic parity ratio.

        Ratio of positive outcome rates between groups.
        1.0 = perfect parity, <1.0 = disparity.
        """
        group_rates = self._positive_rates_by_group(predictions, group_labels)
        if not group_rates:
            return 1.0
        rates = list(group_rates.values())
        max_rate = max(rates) if rates else 1.0
        min_rate = min(rates) if rates else 1.0
        if max_rate == 0:
            return 1.0
        return min_rate / max_rate

    def _disparate_impact(
        self, predictions: list[dict], group_labels: list
    ) -> float:
        """Compute disparate impact ratio (same as demographic parity ratio).

        Values >= 0.8 generally considered acceptable (80% rule).
        """
        return self._demographic_parity(predictions, group_labels)

    def _equalized_odds(
        self, predictions: list[dict], group_labels: list
    ) -> float:
        """Simplified equalized odds: ratio of true positive rates across groups.

        Requires 'actual' field in predictions alongside 'outcome'.
        """
        group_tpr = {}
        groups: dict[str, list] = {}
        for pred, label in zip(predictions, group_labels):
            groups.setdefault(str(label), []).append(pred)

        for group_name, group_preds in groups.items():
            tp = sum(
                1
                for p in group_preds
                if p.get("outcome") and p.get("actual")
            )
            actual_pos = sum(1 for p in group_preds if p.get("actual"))
            group_tpr[group_name] = tp / actual_pos if actual_pos > 0 else 0.0

        if not group_tpr:
            return 1.0
        rates = list(group_tpr.values())
        max_rate = max(rates) if rates else 1.0
        if max_rate == 0:
            return 1.0
        return min(rates) / max_rate

    def _positive_rates_by_group(
        self, predictions: list[dict], group_labels: list
    ) -> dict[str, float]:
        """Compute positive outcome rate for each group."""
        groups: dict[str, list] = {}
        for pred, label in zip(predictions, group_labels):
            groups.setdefault(str(label), []).append(pred)

        rates = {}
        for group_name, group_preds in groups.items():
            positive = sum(1 for p in group_preds if p.get("outcome"))
            rates[group_name] = positive / len(group_preds) if group_preds else 0.0
        return rates

    def _check_threshold(
        self, constraint: FairnessConstraint, value: float | None
    ) -> bool:
        """Check if a metric value meets the constraint threshold."""
        if value is None:
            return False
        if constraint.comparison == "gte":
            return value >= constraint.threshold
        elif constraint.comparison == "lte":
            return value <= constraint.threshold
        elif constraint.comparison == "between":
            upper = constraint.upper_bound if constraint.upper_bound is not None else 1.0
            return constraint.threshold <= value <= upper
        return False


class ProcessingStep(WorkflowStep):
    """General-purpose processing step for non-AI logic.

    Subclass and implement execute(), or pass a callable to the constructor.
    """

    def __init__(
        self,
        name: str,
        *,
        processor: Callable[[Any, dict], StepResult] | None = None,
        sla_seconds: float | None = None,
        escalation_chain: list[str] | None = None,
        tradeoff: AIHumanTradeoff | None = None,
    ) -> None:
        super().__init__(
            name,
            "processing",
            sla_seconds=sla_seconds,
            escalation_chain=escalation_chain,
            tradeoff=tradeoff,
        )
        self._processor = processor

    def execute(self, input_data: Any, context: dict) -> StepResult:
        if self._processor:
            return self._processor(input_data, context)
        raise NotImplementedError(
            f"ProcessingStep '{self.name}' has no processor. "
            f"Either pass a callable or subclass and override execute()."
        )
