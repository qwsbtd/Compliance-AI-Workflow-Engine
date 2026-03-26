"""
Cryptographic signing and key management for audit trail integrity.

Provides HMAC-SHA256 signatures by default (stdlib-only). The Signer protocol
allows organizations to plug in RSA/ECDSA via their own PKI infrastructure.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass
class SigningKey:
    """A cryptographic key used for signing audit records."""

    key_id: str
    key_material: bytes
    algorithm: str = "hmac-sha256"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        # Never leak key material in repr
        return (
            f"SigningKey(key_id={self.key_id!r}, algorithm={self.algorithm!r}, "
            f"created_at={self.created_at!r})"
        )


@runtime_checkable
class Signer(Protocol):
    """Abstract signing protocol.

    Implementations must provide sign() and verify(). Organizations can
    implement this with RSA/ECDSA by wrapping their PKI infrastructure.
    """

    @property
    def algorithm(self) -> str: ...

    def sign(self, data: bytes, key: SigningKey) -> str:
        """Return a base64-encoded signature string."""
        ...

    def verify(self, data: bytes, signature: str, key: SigningKey) -> bool:
        """Return True if signature is valid for data under key."""
        ...


class HMACSigner:
    """Default signer using HMAC-SHA256. Timing-safe verification."""

    algorithm = "hmac-sha256"

    def sign(self, data: bytes, key: SigningKey) -> str:
        """Compute HMAC-SHA256 and return as base64."""
        mac = hmac.new(key.key_material, data, hashlib.sha256)
        return base64.b64encode(mac.digest()).decode("ascii")

    def verify(self, data: bytes, signature: str, key: SigningKey) -> bool:
        """Timing-safe verification of HMAC-SHA256 signature."""
        try:
            expected = base64.b64decode(signature)
        except Exception:
            return False
        mac = hmac.new(key.key_material, data, hashlib.sha256)
        return hmac.compare_digest(mac.digest(), expected)


class KeyRing:
    """In-memory key store mapping key_id -> SigningKey.

    In production, this would be backed by an HSM or key vault.
    The export_public_manifest() method returns key metadata without
    exposing key material, suitable for audit inspection.
    """

    def __init__(self) -> None:
        self._keys: dict[str, SigningKey] = {}

    def generate_key(
        self,
        key_id: str,
        algorithm: str = "hmac-sha256",
        metadata: dict | None = None,
    ) -> SigningKey:
        """Generate a new signing key using cryptographically secure randomness."""
        key = SigningKey(
            key_id=key_id,
            key_material=secrets.token_bytes(32),
            algorithm=algorithm,
            metadata=metadata or {},
        )
        self._keys[key_id] = key
        return key

    def register_key(self, key: SigningKey) -> None:
        """Register an externally created key."""
        self._keys[key.key_id] = key

    def get_key(self, key_id: str) -> SigningKey:
        """Retrieve a key by ID. Raises KeyError if not found."""
        if key_id not in self._keys:
            raise KeyError(f"No key found with id: {key_id!r}")
        return self._keys[key_id]

    def has_key(self, key_id: str) -> bool:
        """Check if a key exists in the ring."""
        return key_id in self._keys

    def list_key_ids(self) -> list[str]:
        """Return all registered key IDs."""
        return list(self._keys.keys())

    def export_public_manifest(self) -> list[dict]:
        """Export key metadata for audit inspection.

        Returns key_id, algorithm, created_at, and metadata.
        NEVER includes key_material.
        """
        return [
            {
                "key_id": k.key_id,
                "algorithm": k.algorithm,
                "created_at": k.created_at,
                "metadata": k.metadata,
            }
            for k in self._keys.values()
        ]
