# Compliance AI Workflow Engine

A reusable Python framework for deploying AI in regulated environments. Built for high-stakes settings — nuclear enrichment (NRC/IAEA) and hospital/healthcare (FDA/HIPAA) — with configurable human-in-the-loop checkpoints, cryptographic audit trails, fairness gates, and regulatory compliance validation.

**Zero external dependencies.** Pure Python 3.10+ standard library.

## Key Features

- **Chainable workflow steps** — sequential pipeline with shared context
- **Human-in-the-loop checkpoints** — multi-party approval (four-eyes principle) with configurable quorum rules
- **SHA-256 hash-chained audit trail** — tamper-evident records with cryptographic signatures (HMAC-SHA256)
- **Model versioning & explainability** — captures model ID, version, input hash, and feature importance for every AI decision
- **Fairness gates** — statistical bias detection (demographic parity, disparate impact) that blocks pipelines on violation
- **PHI redaction** — protected health information stripped at write time (HIPAA minimum-necessary)
- **Regulatory compliance profiles** — built-in profiles for NRC, FDA 21 CFR Part 11, HIPAA, NIST AI RMF, EU AI Act
- **Pluggable audit backends** — file (JSONL), SQLite (with immutability triggers), composite (defense-in-depth)
- **Persistent workflow state** — resume workflows after process restarts (SQLite or air-gapped JSON)
- **Compliance report generation** — auto-generated reports for regulatory inspectors

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  WorkflowEngine                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ModelStep │→ │  Human   │→ │Fairness  │→ ...      │
│  │(AI/ML)   │  │Checkpoint│  │  Gate    │           │
│  └──────────┘  └──────────┘  └──────────┘          │
│       │              │             │                 │
│       ▼              ▼             ▼                 │
│  ┌─────────────────────────────────────────────┐    │
│  │           AuditTrail (SHA-256 chain)         │    │
│  │  Record₁ ──hash──→ Record₂ ──hash──→ Record₃│    │
│  └──────────────────────┬──────────────────────┘    │
│                         │                            │
│  ┌──────────┐  ┌───────▼────────┐  ┌──────────┐    │
│  │ Crypto   │  │   Backends     │  │Regulatory│    │
│  │(signing) │  │(File/SQLite/   │  │ Profiles │    │
│  │          │  │ Composite)     │  │(NRC/FDA/ │    │
│  └──────────┘  └────────────────┘  │ HIPAA)   │    │
│                                     └──────────┘    │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```python
from compliance_engine import (
    WorkflowEngine, ModelStep, HumanCheckpoint, HumanDecision,
    ApprovalRequirement, AuditTrail, HMACSigner, KeyRing,
    StepResult, StepStatus, NRC_PROFILE, ReportGenerator,
)

# 1. Define an AI step
class MyModel(ModelStep):
    def __init__(self):
        super().__init__(name="Risk Scorer", model_id="risk-v1", model_version="1.0")

    def predict(self, input_data, context):
        score = 0.85  # Your model logic here
        return {"risk_score": score, "decision": "approve"}, score

# 2. Set up signing and audit trail
keyring = KeyRing()
key = keyring.generate_key("system")
signer = HMACSigner()
audit = AuditTrail(signer=signer, signing_key=key, workflow_id="RUN-001")

# 3. Build and run the pipeline
engine = WorkflowEngine(
    workflow_name="Risk Assessment",
    steps=[
        MyModel(),
        HumanCheckpoint(
            name="Manager Review",
            review_callback=lambda ctx, instr: HumanDecision(
                approver_id="MGR-1", role="manager", approved=True
            ),
            approval_requirement=ApprovalRequirement(min_approvals=1),
        ),
    ],
    audit_trail=audit,
)

result = engine.run(initial_input={"applicant": "A-123"})
print(f"Completed: {result.completed}")
print(f"Chain valid: {result.audit_trail.verify_chain()[0]}")
print(f"Signatures valid: {result.audit_trail.verify_signatures(keyring)[0]}")
```

## Examples

### Nuclear Centrifuge Anomaly Detection

```bash
python examples/nuclear_monitoring.py
```

NRC-regulated pipeline with:
- AI anomaly detection with explainability
- Four-eyes principle (2 approvers from different roles)
- Dual audit storage (CompositeBackend)
- NRC compliance profile validation
- Inspector report generation

### ICU Sepsis Risk Scoring

```bash
python examples/clinical_decision.py
```

FDA/HIPAA-regulated pipeline with:
- AI sepsis risk scoring with feature importance
- Fairness gate (demographic parity + disparate impact)
- PHI redaction in audit trail
- Composed FDA + HIPAA compliance profile
- Electronic signatures per FDA 21 CFR Part 11

## Regulatory Profiles

| Profile | Frameworks | Key Requirements |
|---------|-----------|-----------------|
| `NRC_PROFILE` | 10 CFR 73/74, IAEA | Human after every AI step, 2+ approvers, signatures, explainability |
| `FDA_PROFILE` | 21 CFR Part 11 | Electronic signatures bound to records, audit trail |
| `HIPAA_PROFILE` | HIPAA | PHI redaction, audit controls, access control |
| `NIST_AI_RMF_PROFILE` | NIST AI RMF | Explainability, fairness checks, model versioning |
| `EU_AI_ACT_HIGH_RISK` | EU AI Act | Explainability, fairness, human oversight |

Compose profiles for multi-framework environments:
```python
from compliance_engine import RegulatoryFramework
profile = RegulatoryFramework.compose("FDA 21 CFR Part 11", "HIPAA Privacy & Security")
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

111 tests covering: cryptographic signing, hash chain integrity, tamper detection, PHI redaction, fairness gates, multi-party approval, workflow persistence, regulatory validation, report generation, and full nuclear/clinical integration scenarios.

## Project Structure

```
compliance_engine/
├── __init__.py       # Public API (40+ exports)
├── crypto.py         # HMAC-SHA256 signing, KeyRing, pluggable Signer protocol
├── audit.py          # AuditRecord, AuditTrail (SHA-256 hash chain)
├── backends.py       # FileBackend, SQLiteBackend (immutable), CompositeBackend
├── steps.py          # ModelStep, HumanCheckpoint, FairnessGate, ProcessingStep
├── engine.py         # WorkflowEngine (orchestration, SLAs, escalation)
├── persistence.py    # SQLiteStatePersister, JSONStatePersister
├── regulatory.py     # ComplianceProfile, built-in profiles, composition
├── reports.py        # ReportGenerator (text, JSON)
└── tradeoff.py       # TradeoffAnalyzer, AI vs human documentation
```

## License

MIT
