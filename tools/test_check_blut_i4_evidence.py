#!/usr/bin/env python3

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_blut_i4_evidence import (
    EvidenceError,
    GATE_SPECS,
    discover_candidate,
    validate_record,
)


SHA = {
    "meta": "1" * 40,
    "engine": "2" * 40,
    "cookbook": "3" * 40,
    "backends": "4" * 40,
}


def setUpModule() -> None:
    """Fail with a readable message when OpenSSL is missing, never skip.

    These tests exercise Ed25519 signature verification -- the trust boundary of
    the I4 evidence gate. A skipUnless would let the whole signature suite go
    quiet on a machine without openssl and still report green, which is worse
    than a failure. Without the guard the first subprocess raises a bare
    FileNotFoundError from inside setUp, which says nothing about what is
    actually missing.
    """
    if shutil.which("openssl") is None:
        raise RuntimeError(
            "openssl is required: these tests generate an Ed25519 keypair and "
            "verify real signatures. Install openssl (3.x -- `pkeyutl -rawin` "
            "is not in 1.1.x) rather than skipping them."
        )


class I4EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="blut-i4-evidence-test-")
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        trust = self.root / "docs/release/blut-1.0"
        trust.mkdir(parents=True)
        keys = self.root / "keys"
        keys.mkdir()
        self.private_key = keys / "operator-private.pem"
        self.public_key = keys / "operator-public.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(self.private_key)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "openssl",
                "pkey",
                "-in",
                str(self.private_key),
                "-pubout",
                "-out",
                str(self.public_key),
            ],
            check=True,
            capture_output=True,
        )
        key_hash = hashlib.sha256(self.public_key.read_bytes()).hexdigest()
        (trust / "trusted-signers.toml").write_text(
            "\n".join(
                [
                    "schema_version = 1",
                    'campaign = "blut-1.0-i4"',
                    "",
                    "[[signers]]",
                    'id = "fixture-operator"',
                    'status = "trusted"',
                    'algorithm = "ed25519"',
                    'public_key_path = "keys/operator-public.pem"',
                    f'public_key_sha256 = "{key_hash}"',
                    'authorized_subjects = ["blut-i4-live"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self.now = dt.datetime(2026, 7, 22, 12, 0, tzinfo=dt.UTC)
        self.record = self._record()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _gate(self, name: str) -> dict[str, object]:
        spec = GATE_SPECS[name]
        infrastructure = {
            field: [f"{field}-{index}" for index in range(minimum)]
            for field, minimum in spec["lists"].items()
        }
        measurements = {
            field: minimum for field, (minimum, _maximum) in spec["measurements"].items()
        }
        if name == "mesh_two_host":
            measurements.update(blob_bytes=100, resume_bytes=40, repeat_bytes=0)
        if name == "async_gpu":
            measurements.update(
                default_throughput_milli_items_per_second=2,
                control_throughput_milli_items_per_second=1,
            )
        if name == "multiprovider_two_account":
            measurements.update(
                actual_microusd=10,
                max_price_microusd=20,
                billed_microusd=10,
                provider_console_microusd=10,
                elapsed_seconds=10,
                deadline_seconds=20,
            )
        artifact = self.bundle / f"{name}.log"
        artifact.write_text(f"{name}: live PASS\n", encoding="utf-8")
        return {
            "status": "pass",
            "command": f"run {name}",
            "infrastructure": infrastructure,
            "measurements": measurements,
            "checks": {check: True for check in spec["checks"]},
            "artifacts": [
                {
                    "path": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        }

    def _record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign": "blut-1.0-i4",
            "recorded_at": "2026-07-22T12:00:00+00:00",
            "candidate": dict(SHA),
            "data_handling": {
                "phi_used": False,
                "fixtures": [
                    {"name": "synthetic", "classification": "Synthetic", "sha256": "a" * 64}
                ],
            },
            "gates": {name: self._gate(name) for name in GATE_SPECS},
            "attestation": {
                "signer_id": "fixture-operator",
                "signature_algorithm": "ed25519",
                "signature_path": "evidence.sig",
            },
        }

    def _write_and_sign(self, record: dict[str, object] | None = None) -> Path:
        path = self.bundle / "evidence.json"
        path.write_text(
            json.dumps(record or self.record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(self.private_key),
                "-rawin",
                "-in",
                str(path),
                "-out",
                str(self.bundle / "evidence.sig"),
            ],
            check=True,
            capture_output=True,
        )
        return path

    def test_complete_signed_bundle_passes(self) -> None:
        validate_record(self.root, self._write_and_sign(), SHA, now=self.now)

    def test_candidate_must_match_exact_source(self) -> None:
        record = copy.deepcopy(self.record)
        record["candidate"]["engine"] = "9" * 40
        with self.assertRaisesRegex(EvidenceError, "must equal current source"):
            validate_record(self.root, self._write_and_sign(record), SHA, now=self.now)

    def test_every_named_check_is_required(self) -> None:
        record = copy.deepcopy(self.record)
        del record["gates"]["ha_three_node"]["checks"]["quorum_loss_refused"]
        with self.assertRaisesRegex(EvidenceError, "keys differ"):
            validate_record(self.root, self._write_and_sign(record), SHA, now=self.now)

    def test_artifact_digest_is_revalidated(self) -> None:
        path = self._write_and_sign()
        (self.bundle / "geo_two_region.log").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "digest mismatch"):
            validate_record(self.root, path, SHA, now=self.now)

    def test_record_tampering_breaks_signature(self) -> None:
        path = self._write_and_sign()
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "signature verification failed"):
            validate_record(self.root, path, SHA, now=self.now)

    def test_untrusted_signer_fails_closed(self) -> None:
        path = self._write_and_sign()
        (self.root / "docs/release/blut-1.0/trusted-signers.toml").write_text(
            'schema_version = 1\ncampaign = "blut-1.0-i4"\nsigners = []\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "not uniquely trusted"):
            validate_record(self.root, path, SHA, now=self.now)

    def test_stale_record_fails_closed(self) -> None:
        with self.assertRaisesRegex(EvidenceError, "older than 30 days"):
            validate_record(
                self.root,
                self._write_and_sign(),
                SHA,
                now=self.now + dt.timedelta(days=31),
            )


class CandidateDiscoveryTests(unittest.TestCase):
    """The four bound revisions must come from four real checkouts.

    Before the 2026-09-07 move this function derived all four from one LamQuant
    meta root through its submodule layout. Here the engine is this repository
    and the others are separate repositories, so they have to be NAMED. The
    plant is the whole point: a component nobody supplied must stop the check,
    because a record that binds four revisions and was verified against two
    would otherwise print PASS.
    """

    def test_an_unsupplied_component_refuses_rather_than_defaulting(self) -> None:
        engine = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, engine, True)
        for missing, supplied in (
            ("meta", {"cookbook": engine}),
            ("cookbook", {"meta": engine}),
        ):
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(EvidenceError, f"--{missing} must name"):
                    discover_candidate(
                        engine, supplied.get("meta"), supplied.get("cookbook")
                    )


if __name__ == "__main__":
    unittest.main()
