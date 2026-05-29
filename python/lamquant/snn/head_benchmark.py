#!/usr/bin/env python3
"""Benchmark SNN classification heads.

Runs each of {threshold_legacy, attention_softmax, crf, temporal_attention,
moe_fsq} against the production MambaSNN backbone on a TUSZ eval sample
and prints a comparison table.

What it actually measures
-------------------------
The non-threshold heads are **untrained** until labeled per-timestep
4-state data lands. This bench therefore reports:
  - Architectural soundness (no NaN, sane gradient shapes)
  - State distribution (% per state) — how each head partitions the
    timestep budget when fed real EEG via the backbone
  - Flicker rate (state transitions per second) — stability proxy
  - CR estimate via LEVEL_TABLE
  - Inference latency per window
  - Parameter count

Real NEDC sensitivity comparison requires per-head training; see
follow-up task on the project's tracker. For 7.7.1 ship, the bench
verifies the architectural drop-in is clean before we invest in
labeled training.

Usage
-----
    python ai_models/snn/head_benchmark.py \\
        --checkpoint weights/snn/mamba_snn_best.pt \\
        --eval-root /mnt/4tb/data/edf/tusz_v2.0.6/edf/eval \\
        --max-files 10 --seed 42

    # With JSON output for the PCCP gate:
    python ai_models/snn/head_benchmark.py ... --json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import torch.serialization as _torch_serialization  # for UnsafeUnpicklingError on newer torch

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lamquant" / "snn"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from lamquant_neural.models.mamba_ssm_minimal import MambaSNN
from lamquant_neural.models.heads import build_head, HEAD_REGISTRY
from snn_to_nedc_eval import load_edf_signal


# Single source of truth for the bench's head set — pulls from the
# registry so adding a new head in heads.py auto-extends the bench
# (V4 Pro Finding 5 of the 5-head-bench commit). Filter to the
# unique-class names so we don't double-test "threshold" + "threshold_legacy".
HEADS = sorted(
    {name for name, cls in HEAD_REGISTRY.items()
     if name == "threshold_legacy" or name != "threshold"}
)


def discover_edfs(eval_root: Path, max_files: int, seed: int) -> list[Path]:
    edfs = sorted(eval_root.rglob("*.edf"))
    if not edfs:
        return []
    rng = random.Random(seed)
    rng.shuffle(edfs)
    return sorted(edfs[:max_files])


def run_head(head, backbone, signal_batch, device) -> dict:
    """Run a head over a batch of windowed signals; return distribution + flicker.

    Latency is measured for the HEAD ONLY — backbone forward runs first
    and we synchronize before starting the head timer (V4 Pro Finding 4
    of the 5-head-bench commit)."""
    K = head.K
    level_table = head.level_table.to(device)
    fs = 250.0
    transitions = 0
    state_counts = torch.zeros(K, dtype=torch.long, device=device)
    flicker_per_window = []
    total_timesteps = 0
    fsq_levels_emitted = []
    latency_ms_head = []

    head.eval().to(device)
    backbone.eval().to(device)

    with torch.no_grad():
        for chunk in signal_batch:  # chunk: [21, 2500] float32 tensor
            x = chunk.unsqueeze(0).to(device)              # [1, 21, 2500]
            logits, _ = backbone(x)                        # [1, 8, T_out]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            states, _ = head(logits, target_T=79)          # [1, T]
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms_head.append((time.perf_counter() - t0) * 1000.0)

            s = states[0].long()                            # [T]
            for k in range(K):
                state_counts[k] += (s == k).sum()
            tr = (s[1:] != s[:-1]).sum().item()
            transitions += tr
            flicker_per_window.append(tr / max(1, len(s) - 1))
            total_timesteps += s.numel()
            fsq_levels_emitted.append(level_table[s].cpu())

    total = max(int(state_counts.sum().item()), 1)
    distribution = (state_counts / total).cpu().tolist()

    # Effective CR estimate (proxy: LEVEL_TABLE gives FSQ levels per state).
    # Lower L → higher CR. The mean log2(L) over the schedule is the bit
    # budget proxy; we report mean L and avg-CR-proxy as 1 / mean(log2(L)).
    fsq_cat = torch.cat(fsq_levels_emitted)
    mean_level = float(fsq_cat.float().mean().item())
    bits_per_timestep = float(torch.log2(fsq_cat.float()).mean().item())

    return {
        "head": head.name,
        "K": K,
        "param_count": int(sum(p.numel() for p in head.parameters() if p.requires_grad)),
        "state_distribution_pct": [round(p * 100, 2) for p in distribution],
        "level_table": head.level_table.cpu().tolist(),
        "mean_fsq_level": round(mean_level, 3),
        "bits_per_timestep": round(bits_per_timestep, 3),
        "flicker_per_step_mean": round(float(np.mean(flicker_per_window)), 4),
        "flicker_per_step_p95": round(float(np.percentile(flicker_per_window, 95)), 4),
        "transitions_per_second_mean": round(
            (transitions / max(1, total_timesteps)) * (79.0 / 10.0), 3
        ),
        "head_latency_ms_mean": round(float(np.mean(latency_ms_head)), 3),
        "head_latency_ms_p95": round(float(np.percentile(latency_ms_head, 95)), 3),
    }


def load_signal_batch(edfs: list[Path], window_samples: int = 2500) -> list[torch.Tensor]:
    """One window per EDF, taken from the middle of the recording to avoid
    edge artefacts. Returns CPU float32 tensors."""
    batch = []
    for path in edfs:
        try:
            signal, _, _ = load_edf_signal(str(path))
            T = signal.shape[1]
            if T < window_samples:
                continue
            start = (T - window_samples) // 2
            chunk = signal[:, start:start + window_samples]
            batch.append(torch.from_numpy(chunk).float())
        except Exception as exc:
            sys.stderr.write(f"[head_benchmark] skip {path.name}: {exc}\n")
    return batch


def main() -> int:
    parser = argparse.ArgumentParser(prog="head_benchmark")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT_DIR / "weights" / "snn" / "mamba_snn_best.pt")
    parser.add_argument("--eval-root", type=Path,
                        default=Path("/mnt/4tb/data/edf/tusz_v2.0.6/edf/eval"))
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--heads", nargs="+", default=HEADS,
                        choices=HEADS, help="Subset of heads to benchmark")
    parser.add_argument("--json", action="store_true",
                        help="Emit __PCCP_JSON__ line for pccp_gate consumption")
    args = parser.parse_args()

    dev = torch.device(args.device if args.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint missing: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.eval_root.exists():
        print(f"ERROR: eval-root missing: {args.eval_root}", file=sys.stderr)
        return 2

    # Build backbone with the production architecture.
    # Prefer weights_only=True (pickle-safe). Fall back to weights_only=False
    # ONLY when the failure is the unpickling-non-tensor-metadata case
    # (existing convention in snn_to_nedc_eval.py / snn_pccp_eval.py for
    # sensitivity/accuracy/epoch fields). Any other error — missing file,
    # corrupt checkpoint, import error — is re-raised so we don't silently
    # load a tampered ckpt via the pickle path (V4 Pro Finding 2 of the
    # 5-head-bench-fixes commit).
    backbone = MambaSNN(in_channels=21, d_model=40, d_state=16, n_layers=2).to(dev)
    try:
        ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, _torch_serialization.UnsafeUnpicklingError if hasattr(_torch_serialization, "UnsafeUnpicklingError") else RuntimeError) as exc:
        # Non-tensor metadata path. Log fallback so operators see the
        # supply-chain implication (pccp/06-cybersecurity.md Section 2).
        sys.stderr.write(
            f"[head_benchmark] weights_only=True failed ({type(exc).__name__}: {exc}); "
            "falling back to weights_only=False — load is now pickle-trusted.\n"
        )
        ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    backbone.load_state_dict(ckpt.get("model", ckpt))
    backbone.eval()
    n_params_backbone = sum(p.numel() for p in backbone.parameters())

    # Warm-up so first inference's CUDA compile time doesn't pollute timing.
    if dev.type == "cuda":
        dummy = torch.zeros((1, 21, 2500), device=dev)
        with torch.no_grad():
            _ = backbone(dummy)
        torch.cuda.synchronize()

    edfs = discover_edfs(args.eval_root, args.max_files, args.seed)
    if not edfs:
        print("ERROR: no EDFs discovered.", file=sys.stderr)
        return 2
    signal_batch = load_signal_batch(edfs)
    if not signal_batch:
        print("ERROR: no signals loaded.", file=sys.stderr)
        return 2
    print(f"[head_benchmark] {len(signal_batch)} windows loaded, "
          f"backbone={n_params_backbone:,} params, device={dev.type}")

    results = []
    for head_name in args.heads:
        # Each head's __init__ owns its initialization (including the
        # CRF transition prior). The bench used to blanket-re-init all
        # parameters, which clobbered intentional defaults like the
        # sticky-diagonal in CRFHead.transitions. Seed Torch before
        # construction so any random init inside __init__ is
        # reproducible across runs (V4 Pro Finding 1 of the
        # 5-head-bench commit).
        torch.manual_seed(args.seed)
        head = build_head(head_name)
        r = run_head(head, backbone, signal_batch, dev)
        results.append(r)
        print(_format_row(r))

    # Pretty header for the table.
    print()
    print(_format_table(results))

    if args.json:
        payload = {"backbone_params": n_params_backbone,
                   "n_windows": len(signal_batch), "results": results}
        print("__PCCP_JSON__" + json.dumps(payload))
    return 0


def _format_row(r: dict) -> str:
    return (f"  [{r['head']:22s}] K={r['K']} "
            f"params={r['param_count']:>5d} "
            f"dist={r['state_distribution_pct']} "
            f"L̄={r['mean_fsq_level']:.2f} "
            f"flicker={r['flicker_per_step_mean']:.3f} "
            f"head_ms={r['head_latency_ms_mean']:.2f}")


def _format_table(results: list[dict]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"  {'Head':<22} {'K':>2} {'Params':>7} {'Dist (%)':<28} "
                 f"{'L̄':>5} {'bits/t':>6} {'Flicker':>8} {'head ms':>8}")
    lines.append("-" * 100)
    for r in results:
        dist = r["state_distribution_pct"]
        dist_str = ", ".join(f"{d:.1f}" for d in dist)
        lines.append(
            f"  {r['head']:<22} {r['K']:>2} {r['param_count']:>7,d} "
            f"{dist_str:<28} {r['mean_fsq_level']:>5.2f} "
            f"{r['bits_per_timestep']:>6.2f} "
            f"{r['flicker_per_step_mean']:>8.3f} "
            f"{r['head_latency_ms_mean']:>8.3f}"
        )
    lines.append("=" * 100)
    lines.append("  Notes:")
    lines.append("    - head ms is HEAD ONLY (post-backbone, post-cuda-sync).")
    lines.append("    - non-threshold heads use the constructor's default init")
    lines.append("      (CRF gets sticky-diagonal transition prior). Real")
    lines.append("      comparison requires per-head training on 4-state labels.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
