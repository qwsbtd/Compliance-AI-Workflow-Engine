"""
Tamper-evident audit trail with SHA-256 hash chains and cryptographic signatures.

Every workflow step execution produces an AuditRecord linked into a hash chain.
Records can be cryptographically signed for non-repudiation. PHI fields are
redacted at write time (before the record exists) for HIPAA compliance.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import AuditBackend
from .crypto import KeyRing, Signer, SigningKey

GENESIS_HASH = "GENESIS"
PHI_REDACTED_MARKER = "[REDACTED-PHI]"


def _safe_serialize(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to string representations."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    return str(obj)


@dataclass(frozen=True)
class AuditRecord:
    """A single immutable audit record in the hash chain.

    Fields are frozen after creation. The record_hash is computed over
    all fields except record_hash, signature, and signed_by.
    """

    record_id: str
    run_id: str
    workflow_id: str
    step_name: str
    step_type: str
    action: str
    actor: str
    timestamp: str
    input_summary: dict
    output_summary: dict
    decision: str | None
    confidence: float | None
    model_info: dict | None
    explainability: dict | None
    tradeoff: dict | None
    escalated: bool
    escalation_reason: str | None
    human_decision: dict | None
    phi_redacted: bool
    previous_hash: str
    record_hash: str = ""
    signature: str | None = None
    signed_by: str | None = None

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization for hashing and signing.

        Excludes record_hash, signature, and signed_by — these are
        computed over this output.
        """
        d = {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "action": self.action,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "decision": self.decision,
            "confidence": self.confidence,
            "model_info": self.model_info,
            "explainability": self.explainability,
            "tradeoff": self.tradeoff,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "human_decision": self.human_decision,
            "phi_redacted": self.phi_redacted,
            "previous_hash": self.previous_hash,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of canonical representation."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_dict(self) -> dict:
        """Full serialization including signature fields."""
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "action": self.action,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "decision": self.decision,
            "confidence": self.confidence,
            "model_info": self.model_info,
            "explainability": self.explainability,
            "tradeoff": self.tradeoff,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "human_decision": self.human_decision,
            "phi_redacted": self.phi_redacted,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
            "signature": self.signature,
            "signed_by": self.signed_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AuditRecord:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AuditTrail:
    """Ordered, hash-chained, optionally signed audit trail.

    Each record's previous_hash links to the prior record's record_hash,
    forming a tamper-evident chain. Any modification to any record breaks
    the chain from that point forward.
    """

    def __init__(
        self,
        *,
        signer: Signer | None = None,
        signing_key: SigningKey | None = None,
        backend: AuditBackend | None = None,
        phi_fields: set[str] | None = None,
        run_id: str | None = None,
        workflow_id: str = "",
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.workflow_id = workflow_id
        self._records: list[AuditRecord] = []
        self._signer = signer
        self._signing_key = signing_key
        self._backend = backend
        self._phi_fields = phi_fields or set()

    @property
    def records(self) -> list[AuditRecord]:
        """Read-only access to the record chain."""
        return list(self._records)

    def append(
        self,
        *,
        step_name: str,
        step_type: str,
        action: str,
        actor: str,
        input_summary: dict | None = None,
        output_summary: dict | None = None,
        decision: str | None = None,
        confidence: float | None = None,
        model_info: dict | None = None,
        explainability: dict | None = None,
        tradeoff: dict | None = None,
        escalated: bool = False,
        escalation_reason: str | None = None,
        human_decision: dict | None = None,
    ) -> AuditRecord:
        """Create a new audit record, chain it, sign it, and persist it."""
        # Sanitize and redact PHI from summaries
        input_safe = _safe_serialize(input_summary or {})
        output_safe = _safe_serialize(output_summary or {})
        phi_redacted = False

        if self._phi_fields:
            input_safe = self._redact_phi(input_safe)
            output_safe = self._redact_phi(output_safe)
            phi_redacted = True

        # Chain link
        previous_hash = (
            self._records[-1].record_hash if self._records else GENESIS_HASH
        )

        # Build record (without hash/signature yet)
        record = AuditRecord(
            record_id=str(uuid.uuid4()),
            run_id=self.run_id,
            workflow_id=self.workflow_id,
            step_name=step_name,
            step_type=step_type,
            action=action,
            actor=actor,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=input_safe,
            output_summary=output_safe,
            decision=decision,
            confidence=confidence,
            model_info=model_info,
            explainability=explainability,
            tradeoff=tradeoff,
            escalated=escalated,
            escalation_reason=escalation_reason,
            human_decision=human_decision,
            phi_redacted=phi_redacted,
            previous_hash=previous_hash,
        )

        # Compute hash
        record_hash = record.compute_hash()

        # Sign if signer configured
        signature = None
        signed_by = None
        if self._signer and self._signing_key:
            signature = self._signer.sign(
                record_hash.encode("utf-8"), self._signing_key
            )
            signed_by = self._signing_key.key_id

        # Create final immutable record with hash and signature
        record = AuditRecord(
            record_id=record.record_id,
            run_id=record.run_id,
            workflow_id=record.workflow_id,
            step_name=record.step_name,
            step_type=record.step_type,
            action=record.action,
            actor=record.actor,
            timestamp=record.timestamp,
            input_summary=record.input_summary,
            output_summary=record.output_summary,
            decision=record.decision,
            confidence=record.confidence,
            model_info=record.model_info,
            explainability=record.explainability,
            tradeoff=record.tradeoff,
            escalated=record.escalated,
            escalation_reason=record.escalation_reason,
            human_decision=record.human_decision,
            phi_redacted=record.phi_redacted,
            previous_hash=record.previous_hash,
            record_hash=record_hash,
            signature=signature,
            signed_by=signed_by,
        )

        self._records.append(record)

        # Persist to backend
        if self._backend:
            self._backend.write(record)

        return record

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the entire hash chain integrity.

        Walks every record, recomputes its hash, and checks the chain links.
        Returns (valid, list_of_error_descriptions).
        """
        errors = []
        for i, record in enumerate(self._records):
            # Check chain link
            expected_prev = (
                self._records[i - 1].record_hash if i > 0 else GENESIS_HASH
            )
            if record.previous_hash != expected_prev:
                errors.append(
                    f"Record {i} ({record.record_id}): broken chain link. "
                    f"Expected previous_hash={expected_prev[:16]}..., "
                    f"got {record.previous_hash[:16]}..."
                )

            # Recompute hash
            computed = record.compute_hash()
            if computed != record.record_hash:
                errors.append(
                    f"Record {i} ({record.record_id}): hash mismatch "
                    f"(record may be tampered). Expected {computed[:16]}..., "
                    f"stored {record.record_hash[:16]}..."
                )

        return (len(errors) == 0, errors)

    def verify_signatures(self, keyring: KeyRing) -> tuple[bool, list[str]]:
        """Verify all signatures using keys from the keyring.

        Returns (all_valid, list_of_error_descriptions).
        """
        if not self._signer:
            return (False, ["No signer configured on this audit trail"])

        errors = []
        for i, record in enumerate(self._records):
            if not record.signature or not record.signed_by:
                errors.append(
                    f"Record {i} ({record.record_id}): missing signature"
                )
                continue

            if not keyring.has_key(record.signed_by):
                errors.append(
                    f"Record {i} ({record.record_id}): unknown signer "
                    f"key_id={record.signed_by!r}"
                )
                continue

            key = keyring.get_key(record.signed_by)
            valid = self._signer.verify(
                record.record_hash.encode("utf-8"), record.signature, key
            )
            if not valid:
                errors.append(
                    f"Record {i} ({record.record_id}): invalid signature"
                )

        return (len(errors) == 0, errors)

    def get_records(
        self,
        *,
        step_name: str | None = None,
        action: str | None = None,
    ) -> list[AuditRecord]:
        """Filter records by step_name and/or action."""
        results = self._records
        if step_name:
            results = [r for r in results if r.step_name == step_name]
        if action:
            results = [r for r in results if r.action == action]
        return results

    def to_json(self, indent: int = 2) -> str:
        """Serialize entire trail as JSON array."""
        return json.dumps(
            [r.to_dict() for r in self._records], indent=indent
        )

    def save(self, path: str | Path) -> None:
        """Save trail to a JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def _redact_phi(self, data: Any) -> Any:
        """Recursively redact PHI fields from data.

        PHI is replaced BEFORE the audit record is created — the original
        data never enters the trail. This satisfies HIPAA minimum-necessary.
        """
        if isinstance(data, dict):
            return {
                k: (
                    PHI_REDACTED_MARKER
                    if k in self._phi_fields
                    else self._redact_phi(v)
                )
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [self._redact_phi(item) for item in data]
        return data

    def __len__(self) -> int:
        return len(self._records)
