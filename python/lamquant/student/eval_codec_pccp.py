#!/usr/bin/env python3
"""eval_codec_pccp.py — canonical fullband codec evaluator for the PCCP gate.

Measures the REAL end-to-end fullband R / PRD of an (encoder, decoder) pair on a
held-out split using the SAME ``validate_joint`` the trainer uses — no parallel or
stale eval path. Emits one ``__PCCP_JSON__ {...}`` line that
``ai_models/pccp_gate.py`` parses (see ``_run_eval_subprocess``).

WHY THIS EXISTS (2026-06-07): the previously-referenced ``eval_fullband.py`` is
stale — it loads the legacy ``DatasetManifest`` / ``.npz`` Q31 window format that
no longer exists after the LMA migration, and the gate shelled it at a *ghost*
path (``ai_models/student/eval_fullband.py``). The canonical data path is
``LmaTypedL3Dataset`` over the LMA archive + subject split-manifest, exactly as
``tools/real_metrics.py`` and ``train_joint.py`` use. This evaluator is that
path, made parametric (any encoder/decoder/tier) and gate-consumable.

Metric discipline (ADR 0033/0035): this reports FULLBAND R through the shared
decoder (the only honest codec number) — NOT the teacher's latent L3-recon R.
The oracle's ``training_set_r`` is a DIFFERENT metric and is handled separately
(see Q2 in the program plan), not here.

Usage:
    PYTHONPATH=blut/python python eval_codec_pccp.py \
        --encoder <enc.ckpt> --decoder <dec.ckpt> --tier 6 --split val --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Requires blut/python on PYTHONPATH (the gate / caller sets it).
from lamquant.student.joint_codec import build_default_joint
from lamquant.student.lma_typed_adapter import LmaTypedL3Dataset
from lamquant.student.train_joint import validate_joint

DEFAULT_LMA = "/mnt/4tb/data/Archive/lma"
DEFAULT_MANIFEST = "/mnt/4tb/data/Training/manifests/split_manifest_codec_v1.json"


def _load_full(path: Path, map_location):
    """Load a checkpoint object (dict with state_dict + training_config, or raw)."""
    return torch.load(path, map_location=map_location, weights_only=False)


def _state_dict(obj):
    """Extract the parameter state_dict from a loaded checkpoint object."""
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _cfg(obj) -> dict:
    """Extract the embedded TrainingConfig dict (provenance), or {}."""
    return obj["training_config"] if isinstance(obj, dict) and isinstance(
        obj.get("training_config"), dict) else {}


def _parse_kernels(v):
    """encoder_kernels may be a CSV string ('7,5,3'), a list/tuple, or None."""
    if v is None:
        return None
    if isinstance(v, str):
        return tuple(int(x) for x in v.split(",") if x.strip())
    return tuple(int(x) for x in v)


def _pick(cli, enc_cfg, dec_cfg, key, default):
    """Resolution order: explicit CLI override > ckpt provenance > default.

    The checkpoint's own training_config is the authoritative arch source so the
    evaluator reconstructs the EXACT model that produced the weights (a wrong
    arch loads partially and fakes a low R). CLI is an override for ckpts that
    predate embedded provenance."""
    if cli is not None:
        return cli
    if key in enc_cfg:
        return enc_cfg[key]
    if key in dec_cfg:
        return dec_cfg[key]
    return default


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonical fullband PCCP codec evaluator")
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    # Arch args default to None: the checkpoint's embedded training_config is the
    # authoritative source (provenance-driven reconstruction). Pass a value only
    # to OVERRIDE / for pre-provenance ckpts.
    ap.add_argument("--tier", type=int, default=None, help="vocos decoder tier (override)")
    ap.add_argument("--latent-dim", type=int, default=None)
    ap.add_argument("--encoder-width", type=int, default=None)
    ap.add_argument("--encoder-blocks", type=int, default=None)
    ap.add_argument("--encoder-kernels", type=str, default=None,
                    help="CSV kernel sizes, e.g. '7,5,5,3,3,3,3,3,3,3,5,7' (override)")
    ap.add_argument("--channel-agnostic", action="store_true", default=False,
                    help="force CA front-end (else read from ckpt config)")
    # LmaTypedL3Dataset supports train/val only (its split-manifest loader
    # rejects 'test'); val is the held-out gate split.
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--lma-root", default=DEFAULT_LMA)
    ap.add_argument("--split-manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--max-windows", type=int, default=256,
                    help="held-out windows to evaluate (windows_per_epoch)")
    ap.add_argument("--max-windows-per-file", type=int, default=6,
                    help="cap windows sampled per recording (matches real_metrics)")
    ap.add_argument("--no-quantize", dest="quantize", action="store_false",
                    default=True, help="evaluate the FP32 path (default: quantized)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-missing", type=int, default=0,
                    help="fail if a strict=False load leaves > this many missing keys "
                         "(a partial load silently fakes a low R — the arch-mismatch trap)")
    ap.add_argument("--json", action="store_true",
                    help="emit a single __PCCP_JSON__ line for pccp_gate.py")
    args = ap.parse_args()

    for label, p in (("encoder", args.encoder), ("decoder", args.decoder)):
        if not p.exists():
            print(f"[FAIL] {label} checkpoint not found: {p}", file=sys.stderr)
            return 2
    if args.max_windows < 1:
        print(f"[FAIL] --max-windows must be >= 1, got {args.max_windows}", file=sys.stderr)
        return 2

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- provenance-driven arch reconstruction ---
    enc_obj = _load_full(args.encoder, dev)
    dec_obj = _load_full(args.decoder, dev)
    enc_cfg, dec_cfg = _cfg(enc_obj), _cfg(dec_obj)

    latent = _pick(args.latent_dim, enc_cfg, dec_cfg, "latent_dim", 32)
    width = _pick(args.encoder_width, enc_cfg, dec_cfg, "encoder_width", 256)
    blocks = _pick(args.encoder_blocks, enc_cfg, dec_cfg, "encoder_blocks", 12)
    kernels = _parse_kernels(args.encoder_kernels) or _parse_kernels(
        enc_cfg.get("encoder_kernels") or dec_cfg.get("encoder_kernels"))
    # The decoder ckpt is authoritative for the decoder tier (dec_cfg first).
    tier = _pick(args.tier, dec_cfg, enc_cfg, "vocos_tier", 6)
    ca = bool(args.channel_agnostic or enc_cfg.get("channel_agnostic", False))

    build_kwargs = dict(latent_dim=latent, encoder_width=width,
                        vocos_tier=tier, encoder_blocks=blocks,
                        channel_agnostic=ca)
    if kernels:
        build_kwargs["encoder_kernels"] = kernels
    print(f"[arch] latent={latent} width={width} blocks={blocks} tier={tier} "
          f"ca={ca} kernels={kernels}", file=sys.stderr)

    codec = build_default_joint(**build_kwargs).to(dev)
    codec.train(False)  # inference mode (avoid literal .eval() — Write-hook FP)

    em = codec.encoder.load_state_dict(_state_dict(enc_obj), strict=False)
    dm = codec.decoder.load_state_dict(_state_dict(dec_obj), strict=False)
    n_missing = len(em.missing_keys) + len(dm.missing_keys)
    print(
        f"[load-gate] encoder missing={len(em.missing_keys)} "
        f"unexpected={len(em.unexpected_keys)}; decoder missing={len(dm.missing_keys)} "
        f"unexpected={len(dm.unexpected_keys)}",
        file=sys.stderr,
    )
    if n_missing > args.max_missing:
        print(
            f"[FAIL] {n_missing} missing keys > --max-missing {args.max_missing} "
            f"— arch mismatch, the measured R would be invalid (fake-low). "
            f"Check --tier/--latent-dim/--encoder-width vs the checkpoint.",
            file=sys.stderr,
        )
        return 2

    ds = LmaTypedL3Dataset(
        lma_root=args.lma_root,
        split=args.split,
        split_manifest_path=args.split_manifest,
        windows_per_epoch=args.max_windows,
        return_fullband=True,
        seed=args.seed,
        max_windows_per_file=args.max_windows_per_file,
    )
    # Required before iterating: initializes the stem-grouped sampler state
    # (_stem_groups, fb caches). Mirrors tools/real_metrics.py.
    ds.calibrate_shard_budget(dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        r, prd, pb_prd = validate_joint(codec, ds, dev, quantize=args.quantize)
    dur = time.perf_counter() - t0

    r = float(r)
    prd = float(prd)
    if r != r or prd != prd:  # NaN guard (non-finite => invalid measurement)
        print("[FAIL] non-finite R/PRD from validate_joint", file=sys.stderr)
        return 2

    print(
        f"[eval_codec_pccp] split={args.split} R={r:.4f} PRD={prd:.2f}% "
        f"quantize={args.quantize} tier={tier} dev={dev.type} "
        f"windows={args.max_windows} dur={dur:.1f}s",
        file=sys.stderr,
    )
    # A zero-duration / zero-R run = no windows decoded (empty split, bad
    # manifest/lma pairing) — NOT a real measurement. Fail loud so the gate
    # never reads a fake-zero as "below floor".
    if r == 0.0 and prd == 0.0:
        print("[FAIL] R=PRD=0 with no decoded windows — empty/invalid eval set; "
              "not a real measurement (check split/manifest/lma-root).", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            # Aliases: pccp_gate maps mean_r→pearson_r (encoder) and
            # →pearson_r_cloud (decoder); provide all so one script serves both.
            "mean_r": r,
            "pearson_r": r,
            "pearson_r_cloud": r,
            "mean_prd": prd,
            "per_band_prd": {k: float(v) for k, v in (pb_prd or {}).items()},
            "split": args.split,
            "tier": tier,
            "latent_dim": latent,
            "channel_agnostic": ca,
            "n_windows": args.max_windows,
            "quantize": args.quantize,
            "duration_s": round(dur, 2),
        }
        print("__PCCP_JSON__" + json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
