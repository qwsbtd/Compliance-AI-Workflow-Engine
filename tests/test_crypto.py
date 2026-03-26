"""Tests for cryptographic signing and key management."""

import pytest

from compliance_engine.crypto import HMACSigner, KeyRing, SigningKey


class TestSigningKey:
    def test_repr_hides_key_material(self):
        key = SigningKey(key_id="test", key_material=b"secret")
        assert "secret" not in repr(key)
        assert "test" in repr(key)


class TestHMACSigner:
    def setup_method(self):
        self.signer = HMACSigner()
        self.keyring = KeyRing()
        self.key = self.keyring.generate_key("signer-1")

    def test_sign_returns_base64_string(self):
        sig = self.signer.sign(b"hello", self.key)
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_verify_valid_signature(self):
        data = b"test data"
        sig = self.signer.sign(data, self.key)
        assert self.signer.verify(data, sig, self.key) is True

    def test_verify_rejects_tampered_data(self):
        sig = self.signer.sign(b"original", self.key)
        assert self.signer.verify(b"tampered", sig, self.key) is False

    def test_verify_rejects_wrong_key(self):
        other_key = self.keyring.generate_key("other")
        sig = self.signer.sign(b"data", self.key)
        assert self.signer.verify(b"data", sig, other_key) is False

    def test_verify_rejects_invalid_base64(self):
        assert self.signer.verify(b"data", "not-valid-base64!!!", self.key) is False

    def test_deterministic_signatures(self):
        data = b"deterministic"
        sig1 = self.signer.sign(data, self.key)
        sig2 = self.signer.sign(data, self.key)
        assert sig1 == sig2

    def test_algorithm_property(self):
        assert self.signer.algorithm == "hmac-sha256"


class TestKeyRing:
    def setup_method(self):
        self.keyring = KeyRing()

    def test_generate_key(self):
        key = self.keyring.generate_key("key-1", metadata={"role": "admin"})
        assert key.key_id == "key-1"
        assert len(key.key_material) == 32
        assert key.metadata == {"role": "admin"}

    def test_get_key(self):
        self.keyring.generate_key("k1")
        key = self.keyring.get_key("k1")
        assert key.key_id == "k1"

    def test_get_key_missing_raises(self):
        with pytest.raises(KeyError):
            self.keyring.get_key("nonexistent")

    def test_has_key(self):
        self.keyring.generate_key("exists")
        assert self.keyring.has_key("exists") is True
        assert self.keyring.has_key("nope") is False

    def test_register_key(self):
        key = SigningKey(key_id="ext", key_material=b"x" * 32)
        self.keyring.register_key(key)
        assert self.keyring.get_key("ext") is key

    def test_list_key_ids(self):
        self.keyring.generate_key("a")
        self.keyring.generate_key("b")
        assert sorted(self.keyring.list_key_ids()) == ["a", "b"]

    def test_export_public_manifest_excludes_key_material(self):
        self.keyring.generate_key("pub", metadata={"dept": "ops"})
        manifest = self.keyring.export_public_manifest()
        assert len(manifest) == 1
        assert manifest[0]["key_id"] == "pub"
        assert "key_material" not in manifest[0]
        assert manifest[0]["metadata"] == {"dept": "ops"}
