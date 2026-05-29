"""RUN_MANIFEST.json writer — captures every training run's provenance.

Per `pccp/02-protocol.md` Section 2.2, every training run must emit a
manifest under `ai_models/training_logs/<run_id>/RUN_MANIFEST.json` with:
  - ULID run_id
  - Started / ended timestamps
  - git_sha
  - Literal argv
  - Resolved config hash (SHA-256)
  - Dataset manifest SHAs from registry.yaml
  - Library versions (torch, numpy, cuda)
  - Hardware identifier (GPU model, driver)
  - Random seed
  - Final checkpoint SHA-256 (set on close)

Usage from a training entry point:

    from lamquant.common.run_manifest import RunManifest

    with RunManifest.start(out_dir="ai_models/training_logs",
                           argv=sys.argv,
                           seed=42,
                           config=config_dict) as manifest:
        train(...)
        manifest.set_checkpoint(final_ckpt_path)
        # On clean exit: manifest.exit_code = 0 and ended_at is set.
        # On exception: exit_code = 1, exception text recorded, raised.

The context manager guarantees the manifest is written even on crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

__all__ = ["RunManifest"]


def _ulid() -> str:
    """RFC-style monotonic-ish ID. uuid4 is fine for now; replace with
    real ULID library if temporal sortability becomes critical."""
    return uuid.uuid4().hex


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for mod_name in ("torch", "numpy", "scipy", "sklearn"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            versions["cuda_runtime"] = torch.version.cuda or "unknown"
            versions["cudnn"] = str(torch.backends.cudnn.version() or "unknown")
    except Exception:
        pass
    return versions


def _hardware_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["gpu_count"] = str(torch.cuda.device_count())
    except Exception:
        pass
    return info


def _config_sha(config: Any) -> str:
    """SHA-256 of canonical-form JSON of the config dict.
    Sorts keys, no extra whitespace, UTF-8."""
    if not isinstance(config, (dict, list)):
        config = {"raw": str(config)}
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _checkpoint_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RunManifest:
    run_id: str
    out_dir: Path
    argv: List[str]
    started_at: str
    seed: Optional[int]
    git_sha: str
    config_hash: str
    library_versions: Dict[str, str]
    hardware: Dict[str, str]
    dataset_manifests: List[Dict[str, str]] = field(default_factory=list)
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    exception: Optional[str] = None

    @classmethod
    def start(cls, out_dir: str | Path, argv: List[str],
              seed: Optional[int] = None,
              config: Any = None,
              dataset_manifests: Optional[List[Dict[str, str]]] = None) -> "RunManifest":
        run_id = _ulid()
        out_dir_path = Path(out_dir) / run_id
        out_dir_path.mkdir(parents=True, exist_ok=True)
        manifest = cls(
            run_id=run_id,
            out_dir=out_dir_path,
            argv=list(argv),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            seed=seed,
            git_sha=_git_sha(),
            config_hash=_config_sha(config) if config is not None else "no-config",
            library_versions=_library_versions(),
            hardware=_hardware_info(),
            dataset_manifests=dataset_manifests or [],
        )
        manifest._write()
        return manifest

    def set_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        self.checkpoint_path = str(p)
        if p.exists():
            self.checkpoint_sha256 = _checkpoint_sha(p)
        self._write()

    def add_dataset(self, name: str, sha256: str, manifest_path: str = "") -> None:
        self.dataset_manifests.append({
            "name": name, "sha256": sha256, "manifest_path": manifest_path,
        })
        self._write()

    def close(self, exit_code: int = 0, exception: Optional[str] = None) -> None:
        self.ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.exit_code = exit_code
        if exception is not None:
            self.exception = exception
        self._write()

    def _write(self) -> None:
        path = self.out_dir / "RUN_MANIFEST.json"
        # Don't serialize out_dir as a Path (json can't); record as str.
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(d, f, indent=2, sort_keys=False)
        tmp.replace(path)

    # Context-manager protocol — guarantees close-on-exit.
    def __enter__(self) -> "RunManifest":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> bool:
        if exc_type is None:
            self.close(exit_code=0)
        else:
            self.close(exit_code=1, exception=f"{exc_type.__name__}: {exc}")
        return False  # do not suppress
