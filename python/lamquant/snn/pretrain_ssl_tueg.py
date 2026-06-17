#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# pretrain_ssl_tueg.py — masked-reconstruction self-supervised pretrain of the
# MambaSNN BACKBONE on the unlabeled LMA corpus (TUEG-scale L3 windows).
#
# BUILD #1 of the ADR-0027 SNN 4-state CR-controller campaign.
#
# Why: the oracle finding says CRITICAL / INTERESTING are TEMPORAL tiers that
# per-timestep oracles fail — the SSM has to learn temporal structure to beat
# them. Supervised 4-state labels are scarce (seizures ~3% of timesteps), so
# we first teach the backbone temporal structure WITHOUT labels: mask random
# spans of the [21,313] L3 input, pass through the backbone, and reconstruct
# the masked L3 from the backbone's own latent. MSE on masked positions only
# (BERT-style masked modeling, FEMBA/LaBraM-style for EEG). The resulting
# backbone-init checkpoint is loaded by train_4state_controller.py via
# --init-backbone (strict=False) so the supervised run starts from a backbone
# that already encodes temporal dynamics.
#
# Reuses (imports, never reimplements):
#   * MambaSNN backbone   (lamquant_neural.models.mamba_ssm_minimal)
#   * LmaDataset          (lamquant.snn.lma_dataset) — train split, labels ignored
#   * WSDScheduler        (lamquant.ingredients.schedules.wsd)
#   * ESOAP               (lamquant.student.esoap) — optional, --optimizer esoap
#
# The reconstruction is built on the SHARED backbone path only: spatial_mix →
# ssm_blocks → readout → activity_logits [B,8,T]. A SEPARATE linear recon head
# (NOT part of MambaSNN) maps those 8 latent channels back to the 21 L3
# channels. Only the MambaSNN state_dict (a strict subset of its keys) is
# saved, so the controller can load it with strict=False and a clean key
# overlap. The recon head is discarded — it never enters the controller.
#
# CLI mirrors the trainer: --lma-root / --split-manifest / --epochs / --out.
# A 1-epoch SMOKE (tiny --max-windows-per-file) proves it trains and that the
# saved keys match MambaSNN's state_dict keys (prints the key-overlap).
#
# Programming-Bible style: contract assertions, no silent fallback, typed.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

LOG = logging.getLogger("lamquant.snn.pretrain_ssl_tueg")

# ---------------------------------------------------------------------------
# Path plumbing — mirror train_4state_controller.py so cross-area imports
# resolve identically whether launched directly or via BLUT.
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("snn", "student", "dataset", "common"):
    _p = os.path.join(ROOT_DIR, "lamquant", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# The MambaSNN backbone lives in the sibling LamQuant-Neural package; the
# trainer relies on it already being importable (installed editable). Add the
# repo root defensively so a bare checkout still resolves it.
_NEURAL_ROOT = "/mnt/4tb/LamQuant/LamQuant-Neural"
if os.path.isdir(_NEURAL_ROOT) and _NEURAL_ROOT not in sys.path:
    sys.path.insert(0, _NEURAL_ROOT)

from lamquant_neural.models.mamba_ssm_minimal import MambaSNN, clamp_ssm_params  # noqa: E402
from lamquant.snn.lma_dataset import LmaDataset  # noqa: E402

# Geometry — L3 latent dims (preprocess_subband_single output). Must match the
# dataset contract: __getitem__ -> (l3 [21,313] float32, labels [8,313] int64).
L3_CHANNELS = 21
L3_T = 313


# ===========================================================================
# Span masking — BERT/FEMBA-style contiguous-span masking over the time axis.
# ===========================================================================

def make_span_mask(B: int, T: int, mask_frac: float, mean_span: int,
                   generator: torch.Generator,
                   device: torch.device) -> torch.Tensor:
    """Boolean [B, T] mask, ``True`` where the timestep is MASKED (to predict).

    Contiguous spans (mean length ``mean_span``) are masked until ~``mask_frac``
    of the T timesteps are covered, independently per batch row. Span masking
    (vs i.i.d. per-timestep) forces the model to use temporal context rather
    than interpolating a single dropped frame — the regime the oracle finding
    says the event tiers (CRITICAL/INTERESTING) live in.

    Each row is guaranteed at least one masked timestep so the MSE always has
    a denominator (no silent zero-loss batch).

    Args:
        B, T: batch and time dims.
        mask_frac: target fraction of timesteps to mask, in (0, 1).
        mean_span: average contiguous span length (>= 1).
        generator: torch.Generator for reproducible masking.
        device: device for the returned mask.

    Returns:
        ``[B, T]`` bool tensor, True == masked.
    """
    assert isinstance(B, int) and B > 0, f"B must be positive int, got {B!r}"
    assert isinstance(T, int) and T > 0, f"T must be positive int, got {T!r}"
    assert 0.0 < mask_frac < 1.0, f"mask_frac must be in (0,1), got {mask_frac}"
    assert isinstance(mean_span, int) and mean_span >= 1, \
        f"mean_span must be int >= 1, got {mean_span!r}"

    target_masked = max(1, int(round(mask_frac * T)))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        n_masked = 0
        # Cap the number of span placements so a pathological RNG draw can't
        # loop forever (each span adds >= 1 masked step, so 4*T placements is
        # a generous ceiling that the target-coverage break hits well before).
        for _ in range(4 * T):
            if n_masked >= target_masked:
                break
            # Span length: at least 1, centred on mean_span (uniform 1..2*mean-1
            # has expectation mean_span; cheap and bounded).
            hi = 2 * mean_span - 1 if mean_span > 1 else 1
            span = int(torch.randint(1, hi + 1, (1,), generator=generator,
                                     device=device).item())
            start = int(torch.randint(0, T, (1,), generator=generator,
                                      device=device).item())
            end = min(start + span, T)
            newly = (~mask[b, start:end]).sum().item()
            mask[b, start:end] = True
            n_masked += int(newly)
        if not bool(mask[b].any()):
            # Guarantee at least one masked step per row.
            j = int(torch.randint(0, T, (1,), generator=generator,
                                   device=device).item())
            mask[b, j] = True
    assert mask.shape == (B, T) and mask.dtype == torch.bool
    assert bool(mask.any()), "span mask produced an all-False mask"
    return mask


# ===========================================================================
# SSL wrapper — backbone forward on masked input + linear reconstruction head.
# ===========================================================================

class SSLReconstructor(nn.Module):
    """MambaSNN backbone + a tiny linear L3 reconstruction head.

    The backbone is the ONLY part whose weights are saved (it is exactly the
    object the controller instantiates). The recon head is auxiliary: it maps
    the backbone's 8 group-logit channels back to the 21 L3 channels so the
    MSE can be taken in L3 space. It is thrown away after pretrain.

    Forward path:
        x_masked [B,21,T]  -- mask-zeroed L3 --
          -> backbone(x_masked) -> activity_logits [B,8,T]
          -> recon_head (8 -> 21, per timestep) -> recon [B,21,T]

    We reconstruct the ORIGINAL (unmasked) L3 at the masked positions only.
    """

    def __init__(self, backbone: MambaSNN, num_groups: int = MambaSNN.NUM_GROUPS,
                 l3_channels: int = L3_CHANNELS):
        super().__init__()
        assert isinstance(backbone, MambaSNN), \
            f"backbone must be MambaSNN, got {type(backbone).__name__}"
        self.backbone = backbone
        # Linear per-timestep map: 8 latent group channels -> 21 L3 channels.
        # Applied as a Conv1d(kernel=1) so it operates over [B, 8, T] directly.
        self.recon_head = nn.Conv1d(num_groups, l3_channels, kernel_size=1)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """x_masked: [B,21,T] -> recon [B,21,T]."""
        assert x_masked.dim() == 3 and x_masked.shape[1] == L3_CHANNELS, \
            f"x_masked must be [B,21,T], got {tuple(x_masked.shape)}"
        activity_logits, _spike_rate, _seizure = self.backbone(x_masked)
        # use_subband=True keeps stride 1, so T_out == T (no pooling). Assert it
        # so a future stride change surfaces here instead of silently
        # misaligning the recon target.
        assert activity_logits.shape[-1] == x_masked.shape[-1], (
            "backbone changed the time dim (stride pooling?) — SSL recon "
            f"expects T_out==T_in, got {activity_logits.shape[-1]} vs "
            f"{x_masked.shape[-1]}")
        recon = self.recon_head(activity_logits)  # [B,21,T]
        assert recon.shape == x_masked.shape
        return recon


def masked_recon_loss(recon: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """MSE between ``recon`` and ``target`` over MASKED timesteps only.

    Args:
        recon:  [B,21,T] reconstructed L3.
        target: [B,21,T] original (unmasked) L3.
        mask:   [B,T] bool, True == masked (the positions we score).

    Returns:
        scalar MSE averaged over (masked timesteps x 21 channels).
    """
    assert recon.shape == target.shape and recon.dim() == 3
    assert mask.shape == (recon.shape[0], recon.shape[2]), \
        f"mask must be [B,T], got {tuple(mask.shape)} for recon {tuple(recon.shape)}"
    m = mask.unsqueeze(1).to(recon.dtype)        # [B,1,T]
    sq = (recon - target).pow(2) * m              # zero on unmasked
    denom = m.sum() * recon.shape[1]              # masked steps x channels
    # denom > 0 is guaranteed by make_span_mask (>=1 masked step per row).
    assert denom.item() > 0, "no masked positions — empty SSL loss"
    return sq.sum() / denom


# ===========================================================================
# Train one epoch.
# ===========================================================================

def train_epoch(ssl: SSLReconstructor, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                mask_frac: float, mean_span: int, grad_clip: float,
                seed: int, epoch: int,
                max_steps: Optional[int] = None) -> Tuple[float, int, int]:
    """Returns (avg_masked_mse, n_steps, nan_skips)."""
    ssl.train()
    total = 0.0
    n_steps = 0
    nan_skips = 0
    # Reproducible per-epoch masking (distinct each epoch, deterministic given
    # the seed). CPU generator: make_span_mask builds the mask on CPU-cheap
    # scalar draws then moves to device.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + epoch)

    for l3, _labels in loader:
        # Labels are IGNORED — this is unsupervised reconstruction.
        l3 = l3.to(device, non_blocking=True)
        assert l3.dim() == 3 and l3.shape[1] == L3_CHANNELS, \
            f"expected l3 [B,21,T], got {tuple(l3.shape)}"
        B, _C, T = l3.shape

        mask = make_span_mask(B, T, mask_frac, mean_span, gen, torch.device("cpu"))
        mask = mask.to(device)
        # Zero the masked timesteps across all 21 channels (BERT [MASK] == 0
        # for a normalized L3; the backbone never sees the masked values).
        x_masked = l3.clone()
        x_masked[mask.unsqueeze(1).expand(-1, L3_CHANNELS, -1)] = 0.0

        optimizer.zero_grad(set_to_none=True)
        recon = ssl(x_masked)
        loss = masked_recon_loss(recon, l3, mask)

        if not torch.isfinite(loss):
            nan_skips += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ssl.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            clamp_ssm_params(ssl.backbone)   # SSM float32-safe band (B1)
        n_steps += 1
        total += float(loss.item())

        if max_steps is not None and n_steps >= max_steps:
            break

    return total / max(n_steps, 1), n_steps, nan_skips


# ===========================================================================
# LMA root expansion — identical semantics to train_4state_controller.py.
# ===========================================================================

def expand_lma_roots(roots) -> list[Path]:
    lma_paths: list[Path] = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix == ".lma":
            lma_paths.append(r)
            continue
        if not r.is_dir():
            raise FileNotFoundError(f"--lma-root not found: {r}")
        found = sorted(r.glob("*/*.lma")) or sorted(r.glob("*.lma"))
        if not found:
            raise RuntimeError(f"no .lma archives under {r}")
        lma_paths.extend(found)
    seen: set = set()
    return [p for p in lma_paths if not (str(p) in seen or seen.add(str(p)))]


# ===========================================================================
# Backbone save — strict subset of MambaSNN.state_dict(), atomic write.
# ===========================================================================

def save_backbone(backbone: MambaSNN, out_path: str, meta: dict) -> dict:
    """Save the SSL-trained MambaSNN parameters + a metadata block, atomically.

    The SSL objective exercises ONLY the shared backbone path
    (``spatial_mix → ssm_blocks → readout`` → activity_logits → recon). The
    ``seizure_head`` is NOT part of the reconstruction graph, so it carries no
    learned signal — and its key SHAPE depends on the controller's
    ``SNN_SEIZURE_HEAD`` env (single Linear vs MLP). We therefore DROP every
    ``seizure_head.*`` key from the saved state_dict so the controller's
    ``load_state_dict(payload["backbone"], strict=False)`` reports ZERO
    unexpected keys regardless of which seizure head the controller builds, and
    keeps the controller's own (bias-initialised) seizure head untouched. The
    only ``missing`` keys on the controller side are then its 4-state head and
    its seizure head — exactly the parts SSL never trained.
    """
    sd = {k: v.detach().to("cpu", copy=True) for k, v in
          backbone.state_dict().items()
          if not k.startswith("seizure_head.")}
    payload = {
        "backbone": sd,
        "backbone_keys": sorted(sd.keys()),
        "ssl_meta": meta,
        "format": "lamquant-snn-ssl-backbone-v1",
    }
    tmp = f"{out_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    return payload["ssl_meta"]


# ===========================================================================
# Main.
# ===========================================================================

def main() -> None:
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Masked-reconstruction SSL pretrain of the MambaSNN "
                    "backbone on the unlabeled LMA corpus.")
    # nargs="+" (the convention shared by train_4state_controller + snn_metrics):
    # pass ALL roots after ONE flag, space-separated — `--lma-root A B C`.
    # NOT repeated `--lma-root A --lma-root B` (nargs keeps only the LAST flag's
    # values), which silently collapses to one corpus.
    p.add_argument("--lma-root", type=Path, nargs="+", required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--split", default="train", choices=["train", "val"],
                   help="LMA split to pretrain on (default train; labels ignored)")
    p.add_argument("--out", type=Path, required=True,
                   help="output backbone-init checkpoint (.pt)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=40)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-windows-per-file", type=int, default=5,
                   help="LMA window-selection cap (tiny value = SMOKE)")
    p.add_argument("--mask-frac", type=float, default=0.30,
                   help="fraction of timesteps masked per window (BERT-style)")
    p.add_argument("--mean-span", type=int, default=10,
                   help="mean contiguous masked-span length (timesteps)")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--warmup-frac", type=float, default=0.10)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "esoap"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-steps-per-epoch", type=int, default=None,
                   help="cap steps/epoch (SMOKE: set tiny to finish fast)")
    p.add_argument("--smoke", action="store_true",
                   help="1-epoch tiny run: forces epochs=1, "
                        "max-windows-per-file=1, max-steps-per-epoch=3, "
                        "num-workers=0 — proves it trains + key overlap")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_windows_per_file = 1
        args.max_steps_per_epoch = args.max_steps_per_epoch or 3
        if args.num_workers is None:
            args.num_workers = 0

    # Force the load-compatible seizure head so the saved backbone state_dict
    # matches the run-3..11 checkpoint key shape (single Linear seizure head),
    # per the build directive. Set BEFORE constructing MambaSNN (the head kind
    # is read from this env in __init__).
    os.environ.setdefault("SNN_SEIZURE_HEAD", "linear")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    import random as _random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[ssl] device={device} optimizer={args.optimizer} "
          f"SNN_SEIZURE_HEAD={os.environ.get('SNN_SEIZURE_HEAD')}")
    print(f"[ssl] mask_frac={args.mask_frac} mean_span={args.mean_span} "
          f"epochs={args.epochs} smoke={args.smoke}")

    # ---- Backbone (subband path, stride 1 -> T_out == T == 313). ----
    backbone = MambaSNN(in_channels=21, d_model=args.d_model,
                        d_state=args.d_state, n_layers=args.n_layers,
                        use_subband=True).to(device)
    ssl = SSLReconstructor(backbone).to(device)
    n_bb = sum(v.numel() for v in backbone.state_dict().values())
    n_head = sum(p.numel() for p in ssl.recon_head.parameters())
    print(f"[ssl] backbone params={n_bb:,} (saved) + recon_head={n_head:,} "
          f"(discarded)")

    # ---- Data: train split, labels ignored. ----
    lma_paths = expand_lma_roots(args.lma_root)
    print(f"[ssl] {len(lma_paths)} .lma archive(s)")
    ds = LmaDataset(lma_paths=lma_paths, split=args.split,
                    split_manifest_path=args.split_manifest,
                    max_windows_per_file=args.max_windows_per_file)
    print(f"[ssl] {args.split}={len(ds)} windows")

    _default_workers = 4 if os.environ.get("L3_CACHE_DIR") else 2
    num_workers = args.num_workers if args.num_workers is not None else \
        int(os.environ.get("LMA_NUM_WORKERS", str(_default_workers)))
    _dl_kwargs = {}
    if num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = int(
            os.environ.get("LMA_PREFETCH_FACTOR", "4"))
    pin = device.type == "cuda" and num_workers > 0
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin,
                        drop_last=False, **_dl_kwargs)

    # ---- Optimizer (ADR 0050/0051 ingredient registry). The ESOAP suffix
    #      routing now lives in one place (ingredients/optimizers/_specs.py)
    #      instead of being copy-pasted here and in train_4state_controller. ----
    named = list(ssl.named_parameters())
    from lamquant.ingredients import build_ingredient
    optimizer = build_ingredient(
        "optimizer", args.optimizer,
        {"lr": args.lr, "weight_decay": args.weight_decay, "betas": (0.9, 0.95)},
        named_params=named)
    print(f"[ssl] optimizer: {args.optimizer} (ingredient registry)")

    # ---- Schedule: WSD warmup -> stable -> short cosine decay tail. ----
    from lamquant.ingredients.schedules.wsd import WSDScheduler
    scheduler = WSDScheduler(optimizer, total_epochs=args.epochs,
                             peak_lr=args.lr, warmup_frac=args.warmup_frac,
                             decay_frac=0.10, min_lr=args.lr_min,
                             warmup_kind="cosine")
    print(f"[ssl] schedule: cosine-warmup -> WSD -> cosine decay "
          f"(warmup={scheduler.warmup_epochs}ep)")

    out_path = str(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    train_start = time.time()
    best_mse = float("inf")
    print(f"[ssl] pretraining {args.epochs} epoch(s) x {len(loader)} batches "
          f"(bs={args.batch_size})")

    for epoch in range(args.epochs):
        ep_start = time.time()
        avg_mse, n_steps, nan_skips = train_epoch(
            ssl, loader, optimizer, device, args.mask_frac, args.mean_span,
            args.grad_clip, args.seed, epoch,
            max_steps=args.max_steps_per_epoch)
        scheduler.step()

        improved = ""
        if avg_mse < best_mse and n_steps > 0:
            best_mse = avg_mse
            improved = " *BEST*"
            meta = save_backbone(backbone, out_path, {
                "epoch": epoch + 1,
                "masked_mse": avg_mse,
                "mask_frac": args.mask_frac,
                "mean_span": args.mean_span,
                "split_manifest": str(args.split_manifest),
                "lma_root": [str(x) for x in args.lma_root],
                "seizure_head_kind": os.environ.get("SNN_SEIZURE_HEAD"),
                "d_model": args.d_model, "d_state": args.d_state,
                "n_layers": args.n_layers,
            })

        ep_sec = time.time() - ep_start
        gpu_mb = (torch.cuda.memory_allocated() / 1e6
                  if torch.cuda.is_available() else 0)
        print(f"E{epoch+1:3d}/{args.epochs} masked_mse={avg_mse:.6f} "
              f"steps={n_steps} skips={nan_skips} lr={scheduler.get_last_lr()[0]:.2e} "
              f"{ep_sec:.0f}s GPU={gpu_mb:.0f}M{improved}")

    # Always ensure a checkpoint exists (e.g. a degenerate 0-step epoch never
    # triggered the *BEST* save). No silent skip.
    if not os.path.exists(out_path):
        save_backbone(backbone, out_path, {
            "epoch": args.epochs, "masked_mse": best_mse,
            "note": "final-save (no per-epoch improvement)",
            "seizure_head_kind": os.environ.get("SNN_SEIZURE_HEAD"),
        })

    total_h = (time.time() - train_start) / 3600
    print(f"\n[ssl] done in {total_h:.3f}h. best masked_mse={best_mse:.6f}. "
          f"saved backbone -> {out_path}")

    # ---- Verify saved keys == MambaSNN.state_dict() keys (key overlap). ----
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    saved_keys = set(payload["backbone"].keys())
    ref = MambaSNN(in_channels=21, d_model=args.d_model, d_state=args.d_state,
                   n_layers=args.n_layers, use_subband=True)
    ref_keys = set(ref.state_dict().keys())
    # The controller's actual strict=False load: zero unexpected keys is the
    # contract (the controller keeps its own seizure + 4-state head).
    missing, unexpected = ref.load_state_dict(payload["backbone"], strict=False)
    overlap = saved_keys & ref_keys
    only_saved = saved_keys - ref_keys
    only_ref = ref_keys - saved_keys
    print(f"[ssl] KEY OVERLAP: saved={len(saved_keys)} ref={len(ref_keys)} "
          f"overlap={len(overlap)} only_saved={len(only_saved)} "
          f"only_ref={len(only_ref)}")
    print(f"[ssl] strict=False load: unexpected={unexpected} "
          f"missing={sorted(missing)}")
    if only_ref:
        print(f"[ssl]   only_ref (controller keeps its init for these — "
              f"seizure_head expected): {sorted(only_ref)}")
    # The saved set MUST be a subset of MambaSNN's keys, AND the load must
    # report zero unexpected keys, for a clean strict=False --init-backbone.
    assert not only_saved, (
        "saved backbone contains keys not in MambaSNN.state_dict() — the "
        "recon head leaked into the checkpoint")
    assert unexpected == [], (
        f"strict=False load reported unexpected keys {unexpected} — saved "
        "backbone is not a clean subset of the controller's MambaSNN")
    print(f"[ssl] init-recipe: in the 4-state trainer, after building the "
          f"MambaSNN backbone (SNN_SEIZURE_HEAD=linear):\n"
          f"        ck = torch.load('{out_path}', map_location=device, "
          f"weights_only=False)\n"
          f"        missing, unexpected = model.load_state_dict("
          f"ck['backbone'], strict=False)\n"
          f"        # expect unexpected == [] and missing == the head/"
          f"seizure-MLP keys only")


if __name__ == "__main__":
    main()
