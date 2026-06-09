"""sysgauge — best-effort point-in-time system snapshot for the metric stream
(ADR 0044 P10, "system gauges").

``snapshot()`` returns a dict of whatever gauges are available: GPU
mem/util/temp/power (``torch.cuda`` + ``nvidia-smi``), host RAM %, host disk
free GiB. Every source is best-effort; **a missing source OMITS its key — never
a fabricated value** (same honesty contract as ``read_metric``). Meant to be
merged into a per-epoch ``MetricLog.append`` row.

stdlib + lazy ``torch`` + an ``nvidia-smi`` subprocess; importing is cheap.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict

__all__ = ["snapshot"]


def _torch_cuda() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import torch  # lazy
        if torch.cuda.is_available():
            out["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1e6
            out["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
    except Exception:
        pass
    return out


def _nvidia_smi() -> Dict[str, float]:
    out: Dict[str, float] = {}
    if shutil.which("nvidia-smi") is None:
        return out
    try:
        q = "utilization.gpu,temperature.gpu,power.draw,memory.used"
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            util, temp, power, mem = (s.strip() for s in r.stdout.splitlines()[0].split(","))
            for key, val in (("gpu_util", util), ("gpu_temp_c", temp),
                             ("gpu_power_w", power), ("gpu_mem_used_mb", mem)):
                try:
                    out[key] = float(val)
                except ValueError:
                    pass  # '[N/A]' from some drivers → omit, don't fabricate
    except Exception:
        pass
    return out


def _host() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        # /proc/meminfo: avail/total → used %, no psutil dep.
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            info[k] = float(rest.strip().split()[0])  # kB
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
        if total and avail:
            out["host_ram_pct"] = 100.0 * (1.0 - avail / total)
    except Exception:
        pass
    try:
        out["host_disk_free_gb"] = shutil.disk_usage(".").free / 1e9
    except Exception:
        pass
    return out


def snapshot() -> Dict[str, float]:
    """All available system gauges as a flat dict. Missing sources are omitted."""
    out: Dict[str, float] = {}
    out.update(_torch_cuda())
    out.update(_nvidia_smi())
    out.update(_host())
    return out
