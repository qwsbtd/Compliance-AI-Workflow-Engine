"""
Regulatory compliance profiles for high-stakes environments.

Built-in profiles for NRC (nuclear), FDA 21 CFR Part 11 (medical devices/pharma),
HIPAA (healthcare privacy), NIST AI RMF, and EU AI Act. Profiles define
requirements that workflows must satisfy. The compose() method merges profiles
using strictest-wins semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RegulatoryRequirement:
    """A single regulatory requirement that a workflow step may satisfy."""

    req_id: str
    description: str
    category: str  # "signature", "audit", "human_oversight", "access_control",
    # "data_integrity", "explainability", "fairness", "retention"
    severity: str  # "mandatory", "recommended", "informational"


@dataclass
class ComplianceProfile:
    """A regulatory compliance profile defining requirements for workflows.

    Policy flags drive automated validation. The validate_workflow() method
    checks a list of steps against these requirements and returns gaps.
    """

    name: str
    framework_ids: list[str] = field(default_factory=list)
    requirements: list[RegulatoryRequirement] = field(default_factory=list)

    # Policy flags
    require_signatures: bool = False
    require_human_checkpoint_after_model: bool = False
    min_approvers: int = 1
    require_explainability: bool = False
    require_fairness_check: bool = False
    require_phi_redaction: bool = False
    require_model_versioning: bool = False
    allowed_signature_algorithms: list[str] = field(
        default_factory=lambda: ["hmac-sha256"]
    )

    def validate_workflow(self, steps: list) -> list[str]:
        """Validate a workflow's steps against this profile.

        Returns a list of compliance gap descriptions (empty = compliant).
        """
        gaps = []

        step_types = [getattr(s, "step_type", "") for s in steps]
        step_names = [getattr(s, "name", f"step_{i}") for i, s in enumerate(steps)]

        # Check: human checkpoint must follow every model step
        if self.require_human_checkpoint_after_model:
            for i, st in enumerate(step_types):
                if st == "model":
                    has_human_after = (
                        i + 1 < len(step_types)
                        and step_types[i + 1] == "human_checkpoint"
                    )
                    if not has_human_after:
                        gaps.append(
                            f"[{self.name}] Step '{step_names[i]}' (model) must be "
                            f"followed by a HumanCheckpoint "
                            f"(required by {', '.join(self.framework_ids)})"
                        )

        # Check: min approvers on human checkpoints
        if self.min_approvers > 1:
            for i, s in enumerate(steps):
                if getattr(s, "step_type", "") == "human_checkpoint":
                    approval_req = getattr(s, "approval_requirement", None)
                    if approval_req is None:
                        gaps.append(
                            f"[{self.name}] Step '{step_names[i]}' requires "
                            f"ApprovalRequirement with min_approvals >= {self.min_approvers}"
                        )
                    elif getattr(approval_req, "min_approvals", 1) < self.min_approvers:
                        gaps.append(
                            f"[{self.name}] Step '{step_names[i]}' has "
                            f"min_approvals={approval_req.min_approvals}, "
                            f"but profile requires >= {self.min_approvers}"
                        )

        # Check: explainability required
        if self.require_explainability:
            has_model_step = any(st == "model" for st in step_types)
            if has_model_step:
                for i, s in enumerate(steps):
                    if getattr(s, "step_type", "") == "model":
                        if not getattr(s, "explain", True):
                            gaps.append(
                                f"[{self.name}] Step '{step_names[i]}' must have "
                                f"explain=True (explainability required)"
                            )

        # Check: fairness check required
        if self.require_fairness_check:
            has_fairness = any(st == "fairness_gate" for st in step_types)
            has_model = any(st == "model" for st in step_types)
            if has_model and not has_fairness:
                gaps.append(
                    f"[{self.name}] Workflow contains model steps but no "
                    f"FairnessGate (required by {', '.join(self.framework_ids)})"
                )

        # Check: model versioning required
        if self.require_model_versioning:
            for i, s in enumerate(steps):
                if getattr(s, "step_type", "") == "model":
                    if not getattr(s, "model_id", None):
                        gaps.append(
                            f"[{self.name}] Step '{step_names[i]}' must declare "
                            f"model_id for model versioning"
                        )

        return gaps


class RegulatoryFramework:
    """Registry of known compliance profiles with composition support."""

    _profiles: dict[str, ComplianceProfile] = {}

    @classmethod
    def register(cls, profile: ComplianceProfile) -> None:
        """Register a compliance profile."""
        cls._profiles[profile.name] = profile

    @classmethod
    def get(cls, name: str) -> ComplianceProfile:
        """Retrieve a profile by name."""
        if name not in cls._profiles:
            raise KeyError(f"Unknown compliance profile: {name!r}")
        return cls._profiles[name]

    @classmethod
    def list_profiles(cls) -> list[str]:
        """List all registered profile names."""
        return list(cls._profiles.keys())

    @classmethod
    def compose(cls, *names: str) -> ComplianceProfile:
        """Merge multiple profiles using strictest-wins semantics.

        Boolean flags use OR (if either requires it, it's required).
        Numeric thresholds use the maximum (strictest).
        Requirements are unioned.
        """
        if not names:
            raise ValueError("At least one profile name required")

        profiles = [cls.get(name) for name in names]

        # Union all requirements (deduplicate by req_id)
        seen_reqs = {}
        for p in profiles:
            for req in p.requirements:
                seen_reqs[req.req_id] = req

        # Union all allowed algorithms
        all_algorithms = set()
        for p in profiles:
            all_algorithms.update(p.allowed_signature_algorithms)

        return ComplianceProfile(
            name=" + ".join(names),
            framework_ids=sum((p.framework_ids for p in profiles), []),
            requirements=list(seen_reqs.values()),
            require_signatures=any(p.require_signatures for p in profiles),
            require_human_checkpoint_after_model=any(
                p.require_human_checkpoint_after_model for p in profiles
            ),
            min_approvers=max(p.min_approvers for p in profiles),
            require_explainability=any(
                p.require_explainability for p in profiles
            ),
            require_fairness_check=any(
                p.require_fairness_check for p in profiles
            ),
            require_phi_redaction=any(
                p.require_phi_redaction for p in profiles
            ),
            require_model_versioning=any(
                p.require_model_versioning for p in profiles
            ),
            allowed_signature_algorithms=sorted(all_algorithms),
        )


# ─── Built-in Profiles ───────────────────────────────────────────────────────

NRC_PROFILE = ComplianceProfile(
    name="NRC Nuclear Safety",
    framework_ids=["10CFR73", "10CFR74", "IAEA_SG"],
    requirements=[
        RegulatoryRequirement(
            "NRC-73.55",
            "Physical protection of plants and materials — access controls required",
            "access_control",
            "mandatory",
        ),
        RegulatoryRequirement(
            "NRC-74.19",
            "Material control & accounting records must be tamper-evident",
            "audit",
            "mandatory",
        ),
        RegulatoryRequirement(
            "NRC-HUMAN-1",
            "Human oversight mandatory on all safety-critical AI decisions",
            "human_oversight",
            "mandatory",
        ),
        RegulatoryRequirement(
            "IAEA-SG-1",
            "Independent verification of material quantities",
            "audit",
            "mandatory",
        ),
        RegulatoryRequirement(
            "NRC-SIG-1",
            "All safety-related records must be cryptographically signed",
            "signature",
            "mandatory",
        ),
    ],
    require_signatures=True,
    require_human_checkpoint_after_model=True,
    min_approvers=2,
    require_explainability=True,
    require_model_versioning=True,
)

FDA_PROFILE = ComplianceProfile(
    name="FDA 21 CFR Part 11",
    framework_ids=["21CFR11"],
    requirements=[
        RegulatoryRequirement(
            "FDA-11.10(a)",
            "Controls for closed systems — system validation required",
            "data_integrity",
            "mandatory",
        ),
        RegulatoryRequirement(
            "FDA-11.10(e)",
            "Audit trail for record creation, modification, and deletion",
            "audit",
            "mandatory",
        ),
        RegulatoryRequirement(
            "FDA-11.50",
            "Signature manifestations — signer name, date, and meaning",
            "signature",
            "mandatory",
        ),
        RegulatoryRequirement(
            "FDA-11.70",
            "Signatures must be cryptographically linked to their records",
            "signature",
            "mandatory",
        ),
        RegulatoryRequirement(
            "FDA-11.200",
            "Electronic signatures must be unique to one individual",
            "signature",
            "mandatory",
        ),
    ],
    require_signatures=True,
    min_approvers=1,
    require_model_versioning=True,
)

HIPAA_PROFILE = ComplianceProfile(
    name="HIPAA Privacy & Security",
    framework_ids=["HIPAA"],
    requirements=[
        RegulatoryRequirement(
            "HIPAA-164.312(b)",
            "Audit controls — hardware, software, and procedural mechanisms",
            "audit",
            "mandatory",
        ),
        RegulatoryRequirement(
            "HIPAA-164.502(b)",
            "Minimum necessary standard — limit PHI disclosure",
            "data_integrity",
            "mandatory",
        ),
        RegulatoryRequirement(
            "HIPAA-164.312(a)(1)",
            "Access control — unique user identification",
            "access_control",
            "mandatory",
        ),
        RegulatoryRequirement(
            "HIPAA-164.312(c)(1)",
            "Integrity controls — protect electronic PHI from alteration",
            "data_integrity",
            "mandatory",
        ),
    ],
    require_phi_redaction=True,
    require_signatures=True,
)

NIST_AI_RMF_PROFILE = ComplianceProfile(
    name="NIST AI Risk Management Framework",
    framework_ids=["NIST_AI_RMF"],
    requirements=[
        RegulatoryRequirement(
            "NIST-MAP-1",
            "Context is established and understood",
            "data_integrity",
            "recommended",
        ),
        RegulatoryRequirement(
            "NIST-MEASURE-2",
            "AI systems are evaluated for trustworthy characteristics",
            "fairness",
            "recommended",
        ),
        RegulatoryRequirement(
            "NIST-MANAGE-1",
            "AI risks are prioritized and responded to",
            "human_oversight",
            "recommended",
        ),
        RegulatoryRequirement(
            "NIST-GOVERN-1",
            "Policies and procedures are in place for AI risk management",
            "audit",
            "recommended",
        ),
    ],
    require_explainability=True,
    require_fairness_check=True,
    require_model_versioning=True,
)

EU_AI_ACT_HIGH_RISK = ComplianceProfile(
    name="EU AI Act - High Risk",
    framework_ids=["EU_AI_ACT_HR"],
    requirements=[
        RegulatoryRequirement(
            "EUAIA-Art9",
            "Risk management system must be established and maintained",
            "audit",
            "mandatory",
        ),
        RegulatoryRequirement(
            "EUAIA-Art13",
            "Transparency — users must be informed about AI system operation",
            "explainability",
            "mandatory",
        ),
        RegulatoryRequirement(
            "EUAIA-Art14",
            "Human oversight measures must be built into high-risk AI",
            "human_oversight",
            "mandatory",
        ),
        RegulatoryRequirement(
            "EUAIA-Art10",
            "Data governance — training data quality and bias monitoring",
            "fairness",
            "mandatory",
        ),
    ],
    require_explainability=True,
    require_fairness_check=True,
    require_human_checkpoint_after_model=True,
    require_model_versioning=True,
)

# Register all built-in profiles
for _profile in [
    NRC_PROFILE,
    FDA_PROFILE,
    HIPAA_PROFILE,
    NIST_AI_RMF_PROFILE,
    EU_AI_ACT_HIGH_RISK,
]:
    RegulatoryFramework.register(_profile)
