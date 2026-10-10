"""Bounded process controls; native validation is covered by Rust fixtures.

Usage: python tools/check_openwarrant_transport.py PATH_TO_VERIFIER_EXAMPLE
Uses a public RFC 8032 test key only, never a private/authority key.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check(executable):
    with tempfile.TemporaryDirectory(prefix="blut-receipt-transport-") as scratch:
        root = Path(scratch)
        key = root / "public-key.bin"
        key.write_bytes(bytes.fromhex(
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        ))
        receipt = root / "receipt.json"
        args = [str(executable), "openwarrant", "verify", "--receipt", str(receipt),
                "--public-key", str(key), "--binding", str(root / "binding.json"),
                "--plan", str(root / "plan.json"), "--job", str(root / "job"),
                "--producer-executable", str(executable)]

        def refusal(expected):
            response = subprocess.run(args, capture_output=True, text=True, timeout=20)
            assert response.returncode != 0, response.stdout
            report = json.loads(response.stdout)
            assert report["schema"] == "blut/openwarrant-verification/v1", report
            assert report["status"] == expected, report
            assert report["assurance"] == "not-established", report
            assert "native" not in report, report
            assert not (root / "job").exists(), "verification created a job"
            print(f"PASS process {expected}; no execution/assurance")

        refusal("unavailable")
        receipt.write_text("{broken}", encoding="utf-8")
        refusal("rejected")
        with receipt.open("wb") as oversized:
            oversized.truncate(8 * 1024 * 1024 + 1025)
        refusal("rejected")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_openwarrant_transport.py PATH_TO_VERIFIER_EXAMPLE")
    check(Path(sys.argv[1]).resolve(strict=True))
