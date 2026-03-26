"""
Workflow engine orchestrator for compliance-regulated AI pipelines.

Executes steps sequentially with audit trail generation, SLA deadline tracking,
escalation chains, and optional persistence for resumable workflows.

NOTE: This engine is a sequential step runner. It does NOT implement proof-gated
transitions (where step advancement requires verification of prior evidence
artifacts). Steps run in order based on prior step completion status.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .audit import AuditTrail
from .backends import AuditBackend
from .crypto import Signer, SigningKey
from .persistence import (
    JSONStatePersister,
    SQLiteStatePersister,
    StatePersister,
    WorkflowCheckpoint,
    WorkflowState,
)
from .regulatory import ComplianceProfile
from .steps import (
    HumanCheckpoint,
    HumanDecision,
    StepResult,
    StepStatus,
    WorkflowStep,
)


@dataclass
class EscalationEvent:
    """Record of an escalation triggered by SLA breach or other conditions."""

    step_name: str
    tier: int
    target: str
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "tier": self.tier,
            "target": self.target,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
        }


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    run_id: str
    workflow_id: str
    workflow_name: str
    state: WorkflowState
    context: dict
    audit_trail: AuditTrail
    step_results: list[StepResult]
    escalation_events: list[EscalationEvent]
    compliance_gaps: list[str]
    completed: bool = False

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "state": self.state.value,
            "completed": self.completed,
            "compliance_gaps": self.compliance_gaps,
            "escalation_events": [e.to_dict() for e in self.escalation_events],
        }


class WorkflowEngine:
    """Sequential workflow engine with audit trail, SLAs, and escalation.

    Executes steps in order. Each step produces an audit record. Human
    checkpoints can pause the workflow for approval collection. The engine
    supports persistence for resumable workflows and pre-flight compliance
    validation against regulatory profiles.
    """

    def __init__(
        self,
        *,
        workflow_name: str,
        steps: list[WorkflowStep],
        audit_trail: AuditTrail | None = None,
        compliance_profile: ComplianceProfile | None = None,
        persister: StatePersister | None = None,
        workflow_id: str | None = None,
    ) -> None:
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.workflow_name = workflow_name
        self._steps = list(steps)
        self._profile = compliance_profile
        self._persister = persister

        # Initialize audit trail
        self._audit = audit_trail if audit_trail is not None else AuditTrail(
            workflow_id=self.workflow_id,
        )
        self._audit.workflow_id = self.workflow_id

        # Execution state
        self._state = WorkflowState.CREATED
        self._current_step_index = 0
        self._step_results: list[StepResult] = []
        self._context: dict = {}
        self._escalation_events: list[EscalationEvent] = []
        self._sla_deadlines: dict[str, str] = {}  # step_name -> ISO deadline

    def validate_workflow(self) -> list[str]:
        """Pre-flight compliance check against the configured profile.

        Returns a list of compliance gaps. Empty list = compliant.
        Should be called before run() in regulated environments.
        """
        if not self._profile:
            return []
        return self._profile.validate_workflow(self._steps)

    def run(
        self,
        initial_input: Any = None,
        context: dict | None = None,
    ) -> WorkflowResult:
        """Execute the workflow from current position.

        Steps run sequentially. Each step's output merges into the shared
        context. If a HumanCheckpoint returns AWAITING_APPROVAL, the
        workflow pauses and can be resumed after approvals are collected.
        If a FairnessGate returns BLOCKED, the workflow halts.
        """
        if context:
            self._context.update(context)
        if initial_input is not None:
            self._context["_initial_input"] = initial_input

        self._state = WorkflowState.RUNNING

        # Audit workflow start
        self._audit.append(
            step_name="_workflow",
            step_type="system",
            action="workflow_started",
            actor="engine",
            input_summary={"workflow_name": self.workflow_name},
            output_summary={},
        )

        # Checkpoint if persister configured
        self._checkpoint()

        # Execute steps from current position
        return self._execute_from_current()

    def resume(self) -> WorkflowResult:
        """Resume workflow from persisted checkpoint."""
        if not self._persister:
            raise RuntimeError("No persister configured — cannot resume")

        checkpoint = self._persister.load(self.workflow_id)
        if checkpoint is None:
            raise RuntimeError(
                f"No checkpoint found for workflow {self.workflow_id}"
            )

        self._restore_checkpoint(checkpoint)
        self._state = WorkflowState.RUNNING

        self._audit.append(
            step_name="_workflow",
            step_type="system",
            action="workflow_resumed",
            actor="engine",
            input_summary={"from_step_index": self._current_step_index},
            output_summary={},
        )

        return self._execute_from_current()

    def submit_approval(
        self,
        step_name: str,
        approver_id: str,
        role: str,
        approved: bool,
        comment: str = "",
    ) -> dict:
        """Submit an approval for a pending HumanCheckpoint.

        Returns a status dict with quorum state and whether the workflow
        can be resumed.
        """
        # Find the checkpoint step
        step = None
        for s in self._steps:
            if s.name == step_name and isinstance(s, HumanCheckpoint):
                step = s
                break

        if step is None:
            return {"error": f"No HumanCheckpoint found with name {step_name!r}"}

        decision = HumanDecision(
            approver_id=approver_id,
            role=role,
            approved=approved,
            comment=comment,
        )

        quorum_met, message = step.submit_approval(decision)

        # Audit the approval/rejection
        self._audit.append(
            step_name=step_name,
            step_type="human_checkpoint",
            action="approval_submitted" if approved else "rejection_submitted",
            actor=approver_id,
            input_summary={"role": role, "comment": comment},
            output_summary={"quorum_met": quorum_met},
            human_decision=decision.to_dict(),
        )

        result = {
            "step_name": step_name,
            "quorum_met": quorum_met,
            "message": message,
            "approval_status": step.approval_status,
            "can_resume": quorum_met,
        }

        if quorum_met:
            # Record the completed approval in step results
            approval_result = StepResult(
                status=StepStatus.COMPLETED,
                output={
                    "approved": True,
                    "approvals": [
                        a.to_dict() for a in step._approvals
                    ],
                },
                decision="approved",
            )
            self._step_results.append(approval_result)
            self._context.update(approval_result.output)
            self._current_step_index += 1
            self._checkpoint()

        if not approved:
            self._audit.append(
                step_name=step_name,
                step_type="human_checkpoint",
                action="checkpoint_rejected",
                actor=approver_id,
                input_summary={},
                output_summary={"approved": False},
                human_decision=decision.to_dict(),
            )
            # Don't auto-fail the workflow on a single rejection
            # unless quorum can no longer be met
            result["rejected"] = True

        return result

    def check_sla(self, step_name: str) -> dict:
        """Check if a step has breached its SLA.

        SLAs are deadline-based (no background threads). The caller
        is responsible for polling this method.
        """
        deadline_str = self._sla_deadlines.get(step_name)
        if deadline_str is None:
            return {"step_name": step_name, "has_sla": False}

        deadline = datetime.fromisoformat(deadline_str)
        now = datetime.now(timezone.utc)
        remaining = (deadline - now).total_seconds()

        return {
            "step_name": step_name,
            "has_sla": True,
            "deadline": deadline_str,
            "remaining_seconds": max(0, remaining),
            "breached": remaining <= 0,
        }

    def trigger_escalation(self, step_name: str, reason: str) -> EscalationEvent | None:
        """Escalate to the next tier in a step's escalation chain.

        Returns the EscalationEvent, or None if no more escalation targets.
        """
        step = None
        for s in self._steps:
            if s.name == step_name:
                step = s
                break

        if step is None or not step.escalation_chain:
            return None

        # Determine current tier
        current_tier = sum(
            1 for e in self._escalation_events
            if e.step_name == step_name
        )

        if current_tier >= len(step.escalation_chain):
            return None  # No more escalation targets

        target = step.escalation_chain[current_tier]
        event = EscalationEvent(
            step_name=step_name,
            tier=current_tier,
            target=target,
            reason=reason,
        )
        self._escalation_events.append(event)

        # Audit the escalation
        self._audit.append(
            step_name=step_name,
            step_type="system",
            action="escalation_triggered",
            actor="engine",
            input_summary={"reason": reason, "tier": current_tier},
            output_summary={"target": target},
            escalated=True,
            escalation_reason=reason,
        )

        self._checkpoint()
        return event

    @property
    def status(self) -> dict:
        """Current workflow status summary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "state": self._state.value,
            "current_step_index": self._current_step_index,
            "total_steps": len(self._steps),
            "current_step": (
                self._steps[self._current_step_index].name
                if self._current_step_index < len(self._steps)
                else None
            ),
            "audit_records": len(self._audit),
            "escalation_events": len(self._escalation_events),
        }

    # ─── Internal Methods ─────────────────────────────────────────────────

    def _execute_from_current(self) -> WorkflowResult:
        """Execute steps starting from _current_step_index."""
        while self._current_step_index < len(self._steps):
            step = self._steps[self._current_step_index]

            # Set SLA deadline if configured
            if step.sla_seconds is not None:
                deadline = datetime.now(timezone.utc).isoformat()
                # Store as offset from now
                from datetime import timedelta

                deadline_dt = datetime.now(timezone.utc) + timedelta(
                    seconds=step.sla_seconds
                )
                self._sla_deadlines[step.name] = deadline_dt.isoformat()

            # Build input for this step
            input_data = self._context.get("_initial_input")

            # Audit step start
            self._audit.append(
                step_name=step.name,
                step_type=step.step_type,
                action="step_started",
                actor="engine",
                input_summary=_sanitize_context(self._context),
                output_summary={},
            )

            # Execute the step
            try:
                result = step.execute(input_data, self._context)
            except Exception as exc:
                result = StepResult(
                    status=StepStatus.FAILED,
                    output={"error": str(exc)},
                    decision="error",
                )

            # Build audit metadata
            model_info_dict = (
                result.model_info.to_dict() if result.model_info else None
            )
            explain_dict = (
                result.explainability.to_dict()
                if result.explainability
                else None
            )
            tradeoff_dict = (
                step.tradeoff.to_dict() if getattr(step, "tradeoff", None) else None
            )

            # Audit step completion
            self._audit.append(
                step_name=step.name,
                step_type=step.step_type,
                action="step_completed",
                actor="engine",
                input_summary=_sanitize_context(self._context),
                output_summary=result.output,
                decision=result.decision,
                confidence=result.confidence,
                model_info=model_info_dict,
                explainability=explain_dict,
                tradeoff=tradeoff_dict,
                escalated=result.escalate,
                escalation_reason=result.escalation_reason,
            )

            self._step_results.append(result)

            # Handle step status
            if result.status == StepStatus.AWAITING_APPROVAL:
                self._state = WorkflowState.AWAITING_APPROVAL
                self._checkpoint()
                return self._build_result()

            if result.status == StepStatus.BLOCKED:
                self._state = WorkflowState.FAILED
                self._audit.append(
                    step_name=step.name,
                    step_type=step.step_type,
                    action="pipeline_blocked",
                    actor="engine",
                    input_summary={},
                    output_summary=result.output,
                    decision=result.decision,
                )
                self._checkpoint()
                return self._build_result()

            if result.status == StepStatus.FAILED:
                self._state = WorkflowState.FAILED
                self._checkpoint()
                return self._build_result()

            # Check for rejection from human checkpoint
            if (
                step.step_type == "human_checkpoint"
                and result.decision == "rejected"
            ):
                self._state = WorkflowState.FAILED
                self._audit.append(
                    step_name=step.name,
                    step_type="human_checkpoint",
                    action="workflow_halted_rejection",
                    actor="engine",
                    input_summary={},
                    output_summary=result.output,
                    decision="rejected",
                )
                self._checkpoint()
                return self._build_result()

            # Merge output into context
            if result.output:
                self._context.update(result.output)

            self._current_step_index += 1
            self._checkpoint()

        # All steps completed
        self._state = WorkflowState.COMPLETED

        self._audit.append(
            step_name="_workflow",
            step_type="system",
            action="workflow_completed",
            actor="engine",
            input_summary={},
            output_summary={"total_steps": len(self._steps)},
        )

        self._checkpoint()
        return self._build_result()

    def _build_result(self) -> WorkflowResult:
        """Build the WorkflowResult from current state."""
        compliance_gaps = self.validate_workflow() if self._profile else []

        return WorkflowResult(
            run_id=self._audit.run_id,
            workflow_id=self.workflow_id,
            workflow_name=self.workflow_name,
            state=self._state,
            context=dict(self._context),
            audit_trail=self._audit,
            step_results=list(self._step_results),
            escalation_events=list(self._escalation_events),
            compliance_gaps=compliance_gaps,
            completed=self._state == WorkflowState.COMPLETED,
        )

    def _checkpoint(self) -> None:
        """Persist current state if persister is configured."""
        if not self._persister:
            return

        # Serialize step results
        serialized_results = {}
        for i, result in enumerate(self._step_results):
            step_name = (
                self._steps[i].name if i < len(self._steps) else f"step_{i}"
            )
            serialized_results[step_name] = {
                "status": result.status.value,
                "output": result.output,
                "decision": result.decision,
                "confidence": result.confidence,
            }

        checkpoint = WorkflowCheckpoint(
            workflow_id=self.workflow_id,
            workflow_name=self.workflow_name,
            state=self._state,
            current_step_index=self._current_step_index,
            step_results=serialized_results,
            context=_sanitize_context(self._context),
            pending_approvals={},
        )
        checkpoint.checkpoint_hash = checkpoint.compute_hash()
        self._persister.save(checkpoint)

    def _restore_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        """Restore engine state from checkpoint."""
        self._state = checkpoint.state
        self._current_step_index = checkpoint.current_step_index
        self._context = dict(checkpoint.context)


def _sanitize_context(context: dict) -> dict:
    """Strip sensitive keys and truncate large values for audit logging."""
    sensitive_patterns = {"password", "token", "secret", "key", "credential", "api_key"}
    sanitized = {}
    for k, v in context.items():
        if k.startswith("_"):
            continue  # Skip internal keys
        if any(pat in k.lower() for pat in sensitive_patterns):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, str) and len(v) > 1000:
            sanitized[k] = v[:1000] + "...[truncated]"
        elif isinstance(v, (dict, list)):
            try:
                import json
                serialized = json.dumps(v, default=str)
                if len(serialized) > 5000:
                    sanitized[k] = "[large_object]"
                else:
                    sanitized[k] = v
            except (TypeError, ValueError):
                sanitized[k] = str(v)[:500]
        else:
            sanitized[k] = v
    return sanitized
