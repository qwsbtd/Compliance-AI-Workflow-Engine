"""
Compliance AI Workflow Engine

A Python framework for deploying AI in regulated environments with
human-in-the-loop checkpoints, cryptographic audit trails, fairness
gates, and regulatory compliance validation.
"""

__version__ = "0.2.0"

from .audit import AuditRecord, AuditTrail
from .backends import AuditBackend, CompositeBackend, FileBackend, SQLiteBackend
from .crypto import HMACSigner, KeyRing, Signer, SigningKey
from .engine import EscalationEvent, WorkflowEngine, WorkflowResult
from .persistence import (
    JSONStatePersister,
    SQLiteStatePersister,
    StatePersister,
    WorkflowCheckpoint,
    WorkflowState,
)
from .regulatory import (
    EU_AI_ACT_HIGH_RISK,
    FDA_PROFILE,
    HIPAA_PROFILE,
    NIST_AI_RMF_PROFILE,
    NRC_PROFILE,
    ComplianceProfile,
    RegulatoryFramework,
    RegulatoryRequirement,
)
from .reports import ReportFormat, ReportGenerator
from .steps import (
    AIHumanTradeoff,
    ApprovalRequirement,
    ExplainabilityInfo,
    FairnessConstraint,
    FairnessGate,
    HumanCheckpoint,
    HumanDecision,
    ModelInfo,
    ModelStep,
    ProcessingStep,
    StepResult,
    StepStatus,
    WorkflowStep,
)
from .tradeoff import TradeoffAnalyzer

__all__ = [
    # Engine
    "WorkflowEngine",
    "WorkflowResult",
    "EscalationEvent",
    # Steps
    "WorkflowStep",
    "ModelStep",
    "HumanCheckpoint",
    "FairnessGate",
    "ProcessingStep",
    "StepResult",
    "StepStatus",
    "ModelInfo",
    "ExplainabilityInfo",
    "ApprovalRequirement",
    "HumanDecision",
    "FairnessConstraint",
    "AIHumanTradeoff",
    # Audit
    "AuditTrail",
    "AuditRecord",
    # Crypto
    "HMACSigner",
    "KeyRing",
    "SigningKey",
    "Signer",
    # Backends
    "AuditBackend",
    "FileBackend",
    "SQLiteBackend",
    "CompositeBackend",
    # Persistence
    "StatePersister",
    "SQLiteStatePersister",
    "JSONStatePersister",
    "WorkflowCheckpoint",
    "WorkflowState",
    # Regulatory
    "ComplianceProfile",
    "RegulatoryFramework",
    "RegulatoryRequirement",
    "NRC_PROFILE",
    "FDA_PROFILE",
    "HIPAA_PROFILE",
    "NIST_AI_RMF_PROFILE",
    "EU_AI_ACT_HIGH_RISK",
    # Reports
    "ReportGenerator",
    "ReportFormat",
    # Tradeoff
    "TradeoffAnalyzer",
]
