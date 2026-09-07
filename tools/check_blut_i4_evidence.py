#!/usr/bin/env python3
"""Validate the signed, source-bound BLUT I4 live-scale evidence bundle.

WHERE THIS CAME FROM AND WHY IT IS HERE NOW. It lived in Quitetall/LamQuant as
`tools/check_blut_i4_evidence.py` until 2026-09-07. The campaign it verifies is
`blut-1.0-i4` and the subject is `blut-i4-live`; nothing it asserts is a fact
about that repository. It reached the components through that repository's
submodule layout -- `training/engine`, `training/cookbooks/lamquant` -- which is
one checkout shape, not the identity of the components. LQ-WAR-0001 M4 named the
move as owed and could not perform it, because that session could not push here.

WHAT CHANGED IN THE MOVE, and nothing else did: `discover_candidate` takes the
ENGINE as this repository and each other component as a supplied checkout
(`--meta`, `--cookbook`), instead of deriving all four from one meta root. An
unsupplied component is a FAILURE, not a skipped field: the record binds four
source revisions and a bundle verified against three of them is not verified.
Every structural, signature, freshness and gate-content rule is untouched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping


SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CAMPAIGN = "blut-1.0-i4"
SUBJECT = "blut-i4-live"
MAX_RECORD_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class EvidenceError(ValueError):
    """One fail-closed evidence violation."""


GATE_SPECS: dict[str, dict[str, Any]] = {
    "mesh_two_host": {
        "lists": {"hosts": 2},
        "checks": {
            "mutual_tls_identity",
            "any_to_any_dispatch",
            "interrupted_transfer_resumed",
            "unchanged_content_zero_chunks",
        },
        "measurements": {
            "blob_bytes": (1, None),
            "resume_bytes": (0, None),
            "repeat_bytes": (0, 0),
        },
    },
    "dp_multinode": {
        "lists": {"hosts": 2, "participants": 2},
        "checks": {
            "dp_sgd_local",
            "epsilon_debited",
            "epsilon_exhaustion_refused",
            "restricted_dispatch_refused",
            "resume_without_round_recompute",
        },
        "measurements": {"rounds": (2, None), "epsilon_curve_points": (3, None)},
    },
    "cloud_provider": {
        "lists": {"providers": 1, "regions": 1},
        "checks": {
            "provider_job_completed",
            "object_roundtrip_verified",
            "restricted_worker_refused",
            "provider_bill_reconciled",
        },
        "measurements": {"completed_jobs": (1, None)},
    },
    "async_gpu": {
        "lists": {"hosts": 1, "device_uuids": 1},
        "checks": {
            "default_profile_not_downgraded",
            "control_profile_user_forced_inline",
            "same_seed_data_and_cache_conditioning",
            "training_loop_only_metric",
        },
        "measurements": {
            "interleaved_pairs": (3, None),
            "default_throughput_milli_items_per_second": (1, None),
            "control_throughput_milli_items_per_second": (1, None),
        },
    },
    "fsdp_two_gpu": {
        "lists": {"hosts": 1, "device_uuids": 2},
        "checks": {
            "two_physical_gpus",
            "effective_batch_invariant",
            "fsdp_ddp_parity",
            "fsdp_single_gpu_parity",
        },
        "measurements": {
            "steps": (200, None),
            "fsdp_vs_ddp_relative_loss_ppm": (0, 1000),
            "fsdp_vs_single_relative_loss_ppm": (0, 1000),
        },
    },
    "multinode_two_gpu": {
        "lists": {"hosts": 2, "device_uuids": 2},
        "checks": {
            "nccl_mesh_interface_initialized",
            "cross_node_loss_parity",
            "missing_only_cache_sync",
            "restricted_multinode_refused",
        },
        "measurements": {
            "steps": (200, None),
            "cross_node_relative_loss_ppm": (0, 1000),
            "full_redownloads": (0, 0),
        },
    },
    "ha_three_node": {
        "lists": {"hosts": 3},
        "checks": {
            "leader_killed_mid_dag",
            "follower_elected",
            "committed_results_adopted",
            "orphans_only_redispatched",
            "quorum_loss_refused",
        },
        "measurements": {
            "completed_stages": (1, None),
            "duplicate_stage_completions": (0, 0),
            "double_bills": (0, 0),
        },
    },
    "geo_two_region": {
        "lists": {"hosts": 4, "regions": 2, "zones": 2},
        "checks": {
            "signed_zone_attestations",
            "dispatch_boundary_refused_out_of_zone",
            "transfer_boundary_refused_out_of_zone",
            "nearest_public_replica_used",
        },
        "measurements": {
            "restricted_bytes_outside_zone": (0, 0),
            "restricted_artifacts_outside_zone": (0, 0),
            "public_full_refetches": (0, 0),
        },
    },
    "multiprovider_two_account": {
        "lists": {"providers": 2, "accounts": 2},
        "checks": {
            "price_and_deadline_policy_met",
            "real_spot_preemption_observed",
            "durable_resume_used",
            "restricted_short_circuit",
            "provider_bills_reconciled",
        },
        "measurements": {
            "completed_jobs": (1, None),
            "observed_preemptions": (1, None),
            "recomputed_completed_stages": (0, 0),
            "duplicate_bills": (0, 0),
            "actual_microusd": (0, None),
            "max_price_microusd": (0, None),
            "billed_microusd": (0, None),
            "provider_console_microusd": (0, None),
            "elapsed_seconds": (1, None),
            "deadline_seconds": (1, None),
        },
    },
}


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{context} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvidenceError(f"{context} keys differ: missing={missing}, extra={extra}")


def _text(value: Any, context: str, *, max_len: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise EvidenceError(f"{context} must be non-empty text")
    if len(value) > max_len:
        raise EvidenceError(f"{context} exceeds {max_len} characters")
    return value


def _unique_strings(value: Any, minimum: int, context: str) -> list[str]:
    if not isinstance(value, list):
        raise EvidenceError(f"{context} must be a list")
    values = [_text(item, f"{context} item", max_len=256) for item in value]
    if len(values) < minimum or len(values) != len(set(values)):
        raise EvidenceError(f"{context} needs at least {minimum} unique values")
    return values


def _uint(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{context} must be an unsigned integer")
    return value


def _safe_relative(base: Path, value: Any, context: str) -> Path:
    raw = _text(value, context, max_len=1024)
    relative = Path(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise EvidenceError(f"{context} must be a confined relative path")
    candidate = base / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise EvidenceError(f"{context} must name a regular non-symlink file")
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise EvidenceError(f"{context} escapes its evidence root") from exc
    return candidate


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_gate(name: str, value: Any, evidence_root: Path) -> None:
    row = _object(value, f"gate {name}")
    _exact_keys(
        row,
        {"status", "command", "infrastructure", "measurements", "checks", "artifacts"},
        f"gate {name}",
    )
    if row["status"] != "pass":
        raise EvidenceError(f"gate {name} status must be pass")
    _text(row["command"], f"gate {name} command")

    spec = GATE_SPECS[name]
    infrastructure = _object(row["infrastructure"], f"gate {name} infrastructure")
    _exact_keys(infrastructure, set(spec["lists"]), f"gate {name} infrastructure")
    for field, minimum in spec["lists"].items():
        _unique_strings(infrastructure[field], minimum, f"gate {name} infrastructure.{field}")

    measurements = _object(row["measurements"], f"gate {name} measurements")
    _exact_keys(measurements, set(spec["measurements"]), f"gate {name} measurements")
    measured: dict[str, int] = {}
    for field, (minimum, maximum) in spec["measurements"].items():
        number = _uint(measurements[field], f"gate {name} measurements.{field}")
        if number < minimum or (maximum is not None and number > maximum):
            raise EvidenceError(
                f"gate {name} measurements.{field}={number} outside [{minimum}, {maximum}]"
            )
        measured[field] = number

    if name == "mesh_two_host" and measured["resume_bytes"] >= measured["blob_bytes"]:
        raise EvidenceError("mesh resume_bytes must be smaller than blob_bytes")
    if name == "async_gpu" and (
        measured["default_throughput_milli_items_per_second"]
        <= measured["control_throughput_milli_items_per_second"]
    ):
        raise EvidenceError("async GPU default throughput must exceed the synchronous control")
    if name == "multiprovider_two_account":
        if measured["actual_microusd"] > measured["max_price_microusd"]:
            raise EvidenceError("multi-provider actual cost exceeds max price")
        if measured["elapsed_seconds"] > measured["deadline_seconds"]:
            raise EvidenceError("multi-provider elapsed time exceeds deadline")
        if measured["billed_microusd"] != measured["provider_console_microusd"]:
            raise EvidenceError("multi-provider bill does not match provider consoles")

    checks = _object(row["checks"], f"gate {name} checks")
    _exact_keys(checks, spec["checks"], f"gate {name} checks")
    for check, passed in checks.items():
        if passed is not True:
            raise EvidenceError(f"gate {name} check {check} must be true")

    artifacts = row["artifacts"]
    if not isinstance(artifacts, list) or not artifacts or len(artifacts) > 16:
        raise EvidenceError(f"gate {name} needs 1..16 artifacts")
    seen: set[str] = set()
    for index, value in enumerate(artifacts):
        artifact = _object(value, f"gate {name} artifact {index}")
        _exact_keys(artifact, {"path", "sha256"}, f"gate {name} artifact {index}")
        raw_path = _text(artifact["path"], f"gate {name} artifact {index} path")
        if raw_path in seen:
            raise EvidenceError(f"gate {name} repeats artifact {raw_path}")
        seen.add(raw_path)
        path = _safe_relative(evidence_root, raw_path, f"gate {name} artifact {index} path")
        if path.stat().st_size == 0 or path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise EvidenceError(f"gate {name} artifact {raw_path} has invalid size")
        expected = _text(artifact["sha256"], f"gate {name} artifact {index} sha256")
        if not SHA256_RE.fullmatch(expected) or _digest(path) != expected:
            raise EvidenceError(f"gate {name} artifact {raw_path} digest mismatch")


def _trusted_signer(root: Path, signer_id: str) -> Mapping[str, Any]:
    catalog_path = root / "docs/release/blut-1.0/trusted-signers.toml"
    try:
        catalog = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EvidenceError("trusted signer catalog is unavailable or invalid") from exc
    if catalog.get("schema_version") != 1 or catalog.get("campaign") != CAMPAIGN:
        raise EvidenceError("trusted signer catalog identity is invalid")
    rows = catalog.get("signers")
    if not isinstance(rows, list):
        raise EvidenceError("trusted signer catalog signers must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == signer_id]
    if len(matches) != 1:
        raise EvidenceError(f"evidence signer {signer_id!r} is not uniquely trusted")
    signer = matches[0]
    _exact_keys(
        signer,
        {"id", "status", "algorithm", "public_key_path", "public_key_sha256", "authorized_subjects"},
        f"trusted signer {signer_id}",
    )
    if signer["status"] != "trusted" or signer["algorithm"] != "ed25519":
        raise EvidenceError(f"trusted signer {signer_id} is not active Ed25519")
    authorized = _unique_strings(
        signer["authorized_subjects"], 1, f"trusted signer {signer_id} authorized_subjects"
    )
    if SUBJECT not in authorized or "*" in authorized:
        raise EvidenceError(f"trusted signer {signer_id} is not narrowly authorized")
    return signer


def _verify_signature(
    root: Path,
    evidence_path: Path,
    evidence_blob: bytes,
    attestation: Mapping[str, Any],
) -> None:
    signer_id = _text(attestation["signer_id"], "attestation signer_id", max_len=128)
    if attestation["signature_algorithm"] != "ed25519":
        raise EvidenceError("attestation signature_algorithm must be ed25519")
    signer = _trusted_signer(root, signer_id)
    public_key = _safe_relative(root, signer["public_key_path"], "trusted signer public_key_path")
    expected_key_hash = _text(signer["public_key_sha256"], "trusted signer public_key_sha256")
    if not SHA256_RE.fullmatch(expected_key_hash) or _digest(public_key) != expected_key_hash:
        raise EvidenceError("trusted signer public key digest mismatch")
    signature = _safe_relative(
        evidence_path.parent, attestation["signature_path"], "attestation signature_path"
    )
    if signature.stat().st_size != 64:
        raise EvidenceError("detached Ed25519 signature must be exactly 64 bytes")
    try:
        verified = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(public_key),
                "-rawin",
                "-in",
                str(evidence_path),
                "-sigfile",
                str(signature),
            ],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError("OpenSSL Ed25519 verification is unavailable") from exc
    if verified.returncode != 0:
        raise EvidenceError("detached evidence signature verification failed")


def validate_record(
    root: Path,
    evidence_path: Path,
    expected_candidate: Mapping[str, str],
    *,
    now: dt.datetime | None = None,
) -> None:
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise EvidenceError("evidence record must be a regular non-symlink file")
    evidence_blob = evidence_path.read_bytes()
    if not evidence_blob or len(evidence_blob) > MAX_RECORD_BYTES:
        raise EvidenceError("evidence record size is invalid")
    try:
        record = _object(json.loads(evidence_blob), "evidence record")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("evidence record is not valid UTF-8 JSON") from exc
    _exact_keys(
        record,
        {"schema_version", "campaign", "recorded_at", "candidate", "data_handling", "gates", "attestation"},
        "evidence record",
    )
    if record["schema_version"] != 1 or record["campaign"] != CAMPAIGN:
        raise EvidenceError("evidence record identity is invalid")

    candidate = _object(record["candidate"], "candidate")
    _exact_keys(candidate, {"meta", "engine", "cookbook", "backends"}, "candidate")
    for component, expected in expected_candidate.items():
        actual = _text(candidate.get(component), f"candidate {component}")
        if not SHA_RE.fullmatch(actual) or actual != expected:
            raise EvidenceError(
                f"candidate {component} must equal current source {expected}, got {actual}"
            )

    try:
        recorded_at = dt.datetime.fromisoformat(_text(record["recorded_at"], "recorded_at"))
    except ValueError as exc:
        raise EvidenceError("recorded_at must be an ISO-8601 timestamp") from exc
    if recorded_at.tzinfo is None or recorded_at.utcoffset() != dt.timedelta(0):
        raise EvidenceError("recorded_at must carry an explicit UTC offset")
    current = now or dt.datetime.now(dt.UTC)
    if recorded_at > current + dt.timedelta(minutes=5):
        raise EvidenceError("evidence record is dated in the future")
    if current - recorded_at > dt.timedelta(days=30):
        raise EvidenceError("evidence record is older than 30 days")

    data = _object(record["data_handling"], "data_handling")
    _exact_keys(data, {"phi_used", "fixtures"}, "data_handling")
    if data["phi_used"] is not False:
        raise EvidenceError("I4 evidence must never use PHI")
    fixtures = data["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise EvidenceError("data_handling.fixtures must be non-empty")
    for index, value in enumerate(fixtures):
        fixture = _object(value, f"fixture {index}")
        _exact_keys(fixture, {"name", "classification", "sha256"}, f"fixture {index}")
        _text(fixture["name"], f"fixture {index} name", max_len=256)
        if fixture["classification"] not in {"Public", "Synthetic"}:
            raise EvidenceError(f"fixture {index} classification must be Public or Synthetic")
        if not SHA256_RE.fullmatch(_text(fixture["sha256"], f"fixture {index} sha256")):
            raise EvidenceError(f"fixture {index} sha256 must be 64 lowercase hex characters")

    gates = _object(record["gates"], "gates")
    _exact_keys(gates, set(GATE_SPECS), "gates")
    for name, value in gates.items():
        _validate_gate(name, value, evidence_path.parent)

    attestation = _object(record["attestation"], "attestation")
    _exact_keys(attestation, {"signer_id", "signature_algorithm", "signature_path"}, "attestation")
    _verify_signature(root, evidence_path, evidence_blob, attestation)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvidenceError(f"git {' '.join(args)} failed in {root}")
    return result.stdout.strip()


def discover_candidate(engine: Path, meta: Path | None, cookbook: Path | None) -> dict[str, str]:
    """The four source revisions the record must bind, read from real checkouts.

    `engine` is THIS repository. `meta` and `cookbook` are separate
    repositories and must be named; refusing here rather than defaulting is the
    point, because a record that binds four revisions and was checked against
    two would report PASS.
    """
    for label, path in (("meta", meta), ("cookbook", cookbook)):
        if path is None:
            raise EvidenceError(
                f"--{label} must name the {label} checkout: the I4 record binds its "
                "revision and it cannot be read from this repository"
            )
    assert meta is not None and cookbook is not None  # narrowed by the loop above
    candidate = {
        "meta": _git(meta, "rev-parse", "HEAD"),
        "engine": _git(engine, "rev-parse", "HEAD"),
        "cookbook": _git(cookbook, "rev-parse", "HEAD"),
    }
    manifest = tomllib.loads((cookbook / "Cargo.toml").read_text(encoding="utf-8"))
    dependencies = _object(manifest.get("dependencies"), "cookbook dependencies")
    for dependency in ("blut", "blut-types", "blut-tui"):
        row = _object(dependencies.get(dependency), f"cookbook dependency {dependency}")
        if row.get("rev") != candidate["engine"]:
            raise EvidenceError(
                f"cookbook dependency {dependency} is not pinned to engine {candidate['engine']}"
            )
    backend = _object(dependencies.get("blut-backends"), "cookbook dependency blut-backends")
    backend_sha = _text(backend.get("rev"), "cookbook blut-backends revision")
    if not SHA_RE.fullmatch(backend_sha):
        raise EvidenceError("cookbook blut-backends revision must be a full SHA")
    candidate["backends"] = backend_sha
    for component, value in candidate.items():
        if not SHA_RE.fullmatch(value):
            raise EvidenceError(f"{component} source is not a full Git SHA")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="signed I4 evidence JSON")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the blut repository root (default: this checkout)",
    )
    parser.add_argument(
        "--meta", type=Path, default=None,
        help="checkout of the LamQuant meta-repository whose revision the record binds",
    )
    parser.add_argument(
        "--cookbook", type=Path, default=None,
        help="checkout of blut-cookbook-lamquant whose revision the record binds",
    )
    args = parser.parse_args(argv)
    try:
        candidate = discover_candidate(
            args.repo.resolve(),
            args.meta.resolve() if args.meta else None,
            args.cookbook.resolve() if args.cookbook else None,
        )
        validate_record(args.repo.resolve(), args.evidence.absolute(), candidate)
    except (EvidenceError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"BLUT I4 evidence: FAIL: {exc}", file=sys.stderr)
        return 1
    print("BLUT I4 evidence: PASS")
    for component, sha in candidate.items():
        print(f"  {component}: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
