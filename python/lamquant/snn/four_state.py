# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# four_state.py — 4-state "needs-precision" CR-controller target derivation.
#
# The LamQuant SNN's ACTUAL job is NOT seizure detection. It is a per-
# latent-timestep compression-tier controller: each EEG latent timestep is
# mapped to one of four SNAC compression tiers
#
#     QUIET=0       → FSQ level 2   → CR ~525:1   (max compression)
#     BASELINE=1    → FSQ level 3   → CR ~134:1
#     INTERESTING=2 → FSQ level 4   → CR ~82:1
#     CRITICAL=3    → FSQ level 5   → CR ~63:1    (max fidelity)
#
# Seizure is ONE trigger of CRITICAL — the seizure-safety tier — not the
# objective. The target distribution (ADR 0007 / project_adaptive_snac_cr.md)
# is approximately:
#
#     QUIET 50%  /  BASELINE 35%  /  INTERESTING 12%  /  CRITICAL 3%
#
# This module derives that 4-state target from the existing 3-class activity
# labels ({0=quiet, 1=active, 2=seizure} per spatial group per latent
# timestep) plus the L3 signal energy, and calibrates the QUIET/BASELINE
# split threshold from the data.
#
# Programming-Bible style: contract assertions, no silent fallback, typed.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # avoid importing the dataset at module load (heavy deps)
    from lamquant.snn.lma_dataset import LmaDataset

# ----------------------------------------------------------------------
# State / level / CR tables — single source of truth for this module.
# ----------------------------------------------------------------------

# State index → human name.
STATE_NAMES = ("QUIET", "BASELINE", "INTERESTING", "CRITICAL")
NUM_STATES = 4

# State index → FSQ level. MUST match heads.LEVEL_TABLE_4 = [2, 3, 4, 5].
LEVEL_TABLE_4 = (2, 3, 4, 5)

# State index → compression ratio (X:1). From ADR 0007 / adaptive SNAC memory.
# Lower state (more compression) = higher CR.
CR_TABLE_4 = (525.0, 134.0, 82.0, 63.0)

# Target marginal distribution over the 4 states (ADR 0007).
TARGET_DIST_4 = (0.50, 0.35, 0.12, 0.03)


# ----------------------------------------------------------------------
# Per-timestep L3 RMS — the QUIET vs BASELINE discriminator.
# ----------------------------------------------------------------------

def _l3_rms_pooled(l3: np.ndarray, target_T: int) -> np.ndarray:
    """Per-timestep RMS of L3 across the 21 channels, pooled to ``target_T``.

    Args:
        l3: ``[21, T_l3]`` float L3 subband signal.
        target_T: the label time resolution to align RMS to.

    Returns:
        ``[target_T]`` float64 RMS, one value per (pooled) latent timestep.

    Resolution handling: L3 is computed per latent timestep too (T_l3 == 313
    in the dataset, same as the label T), but in the general case T_l3 may
    differ from target_T. We compute the per-timestep RMS at the native L3
    resolution, then average-pool / linearly-interpolate to ``target_T`` so
    each label timestep gets a single energy scalar at its own resolution.
    """
    assert isinstance(l3, np.ndarray), f"l3 must be ndarray, got {type(l3).__name__}"
    assert l3.ndim == 2, f"l3 must be 2-D [C, T_l3], got shape {l3.shape}"
    assert l3.shape[0] == 21, f"l3 must have 21 channels, got {l3.shape[0]}"
    assert isinstance(target_T, int) and target_T > 0, \
        f"target_T must be positive int, got {target_T!r}"

    # RMS across the 21 channels per native L3 timestep → [T_l3].
    l3f = l3.astype(np.float64, copy=False)
    rms_native = np.sqrt(np.mean(l3f * l3f, axis=0))  # [T_l3]
    T_l3 = rms_native.shape[0]
    assert T_l3 > 0, "l3 has zero time dimension"

    if T_l3 == target_T:
        return rms_native

    # Pool/interp to target_T. Use torch adaptive avg pool for the downsample
    # case and linear interpolate for the upsample case — both keep the
    # energy envelope, no silent truncation.
    t = torch.from_numpy(rms_native).view(1, 1, T_l3)
    if T_l3 > target_T:
        out = F.adaptive_avg_pool1d(t, target_T)
    else:
        out = F.interpolate(t, size=target_T, mode="linear", align_corners=False)
    out_np = out.view(target_T).numpy()
    assert out_np.shape == (target_T,), \
        f"pooled RMS shape {out_np.shape} != ({target_T},)"
    return out_np


# ----------------------------------------------------------------------
# Deliverable 1 — pure 4-state target derivation.
# ----------------------------------------------------------------------

def derive_4state_target(labels_3class: np.ndarray,
                         l3: np.ndarray,
                         quiet_rms_threshold: float) -> np.ndarray:
    """Map per-timestep 3-class activity labels + L3 energy → 4-state target.

    Logic per latent timestep ``t`` (operating at the LABEL resolution ``T``):

      * ``max3 = max over the 8 spatial groups of labels_3class[:, t]``
      * ``max3 == 2`` (any group seizure)  → CRITICAL = 3
      * ``max3 == 1`` (any group active)   → INTERESTING = 2
      * ``max3 == 0`` (all groups quiet)   → split by L3 energy:
            BASELINE = 1 if rms[t] >  quiet_rms_threshold
            QUIET    = 0 if rms[t] <= quiet_rms_threshold

    Seizure is only ONE route into CRITICAL — it is the seizure-safety tier,
    not the training objective.

    Args:
        labels_3class: ``[8, T]`` int in {0, 1, 2} — 3-class label per spatial
            group per latent timestep.
        l3: ``[21, T_l3]`` float L3 subband signal (energy source for the
            QUIET/BASELINE split). Pooled to ``T`` internally.
        quiet_rms_threshold: RMS cut separating QUIET from BASELINE among the
            quiet (max3==0) timesteps.

    Returns:
        ``[T]`` int64 array in {0, 1, 2, 3}.
    """
    assert isinstance(labels_3class, np.ndarray), \
        f"labels_3class must be ndarray, got {type(labels_3class).__name__}"
    assert labels_3class.ndim == 2 and labels_3class.shape[0] == 8, \
        f"labels_3class must be [8, T], got shape {labels_3class.shape}"
    assert isinstance(l3, np.ndarray) and l3.ndim == 2 and l3.shape[0] == 21, \
        f"l3 must be [21, T_l3], got shape {getattr(l3, 'shape', None)}"
    lbl = labels_3class.astype(np.int64, copy=False)
    assert lbl.min() >= 0 and lbl.max() <= 2, \
        f"labels_3class out of range — expected {{0,1,2}}, got min={lbl.min()} max={lbl.max()}"
    assert np.isfinite(quiet_rms_threshold), \
        f"quiet_rms_threshold must be finite, got {quiet_rms_threshold!r}"

    T = lbl.shape[1]
    max3 = lbl.max(axis=0)  # [T] in {0,1,2}

    # RMS aligned to the LABEL resolution T (handles T vs T_l3 mismatch).
    rms = _l3_rms_pooled(l3, T)  # [T]

    target = np.zeros(T, dtype=np.int64)               # default QUIET=0
    quiet_mask = (max3 == 0)
    # Among quiet timesteps, energetic ones become BASELINE=1.
    target[quiet_mask & (rms > quiet_rms_threshold)] = 1
    target[max3 == 1] = 2                               # INTERESTING
    target[max3 == 2] = 3                               # CRITICAL

    assert target.shape == (T,) and target.dtype == np.int64
    assert target.min() >= 0 and target.max() <= 3
    return target


def apply_energy_failsafe(states: np.ndarray,
                          l3: np.ndarray,
                          hi_rms_threshold: float) -> np.ndarray:
    """Deterministic high-amplitude fail-safe override (ADR 0029 addendum §3).

    Deploy-time STRUCTURAL guarantee: any timestep whose pooled L3 RMS exceeds
    ``hi_rms_threshold`` is FORCED to CRITICAL (max tier / max FSQ level / max
    bits), regardless of the learned controller's prediction. It binds on the
    HARD argmax decision the deployed codec uses, so the guarantee cannot leak
    through a soft policy. Over-firing only OVER-codes (bounded extra bits);
    it can never under-code — the asymmetric-cost-safe direction.

    SCOPE (read carefully): structural ONLY for the HIGH-AMPLITUDE critical
    subset. CRITICAL content is seizure-annotated, and many seizures are
    LOW-amplitude electrographic — those carry no deterministic deploy-time
    signal and are NOT caught here. Covering them is the job of the learned
    conservative seizure-suspicion gate, whose sensitivity is DATA-BOUND (~0.70
    frontier). This override is a supplement to that gate, not a replacement,
    and must NOT be marketed as a full seizure guarantee.

    Args:
        states: ``[T]`` int array in {0,1,2,3} — the controller's per-timestep
            tier decision (hard argmax).
        l3: ``[21, T_l3]`` float L3 subband signal (energy source). Pooled to
            the state resolution ``T`` internally.
        hi_rms_threshold: RMS above which a timestep is forced to CRITICAL. A
            CONSERVATIVE high percentile (fire on the loud stuff).

    Returns:
        ``[T]`` int64 — ``states`` with high-RMS timesteps raised to CRITICAL.
        Never lowers a tier.
    """
    states = np.asarray(states)
    assert states.ndim == 1, f"states must be [T], got {states.shape}"
    assert isinstance(l3, np.ndarray) and l3.ndim == 2 and l3.shape[0] == 21, \
        f"l3 must be [21, T_l3], got shape {getattr(l3, 'shape', None)}"
    assert np.isfinite(hi_rms_threshold), \
        f"hi_rms_threshold must be finite, got {hi_rms_threshold!r}"
    T = states.shape[0]
    rms = _l3_rms_pooled(l3, T)                      # [T], aligned to states
    out = states.astype(np.int64, copy=True)
    out[rms > hi_rms_threshold] = len(STATE_NAMES) - 1   # force CRITICAL (max tier)
    return out


# ----------------------------------------------------------------------
# Deliverable 1 (cont.) — calibrate the QUIET/BASELINE threshold.
# ----------------------------------------------------------------------

# Fraction of QUIET (max3==0) timesteps that should become BASELINE so that,
# given the target QUIET 50% / BASELINE 35% marginals, the threshold lands the
# split correctly: BASELINE share among quiet = 0.35 / (0.50 + 0.35) ≈ 0.412.
BASELINE_FRACTION_OF_QUIET = TARGET_DIST_4[1] / (TARGET_DIST_4[0] + TARGET_DIST_4[1])


def calibrate_quiet_threshold(dataset: "LmaDataset",
                              n_windows: int = 2000,
                              baseline_frac_of_quiet: float = BASELINE_FRACTION_OF_QUIET,
                              seed: int = 1337) -> float:
    """Pick the RMS threshold that splits QUIET vs BASELINE at the target ratio.

    Samples up to ``n_windows`` windows from ``dataset``, collects the
    per-timestep L3 RMS of QUIET-only timesteps (``max3 == 0``), and returns
    the percentile that makes ``baseline_frac_of_quiet`` (~41.2%) of those
    quiet timesteps fall ABOVE the threshold (i.e. become BASELINE). With the
    target QUIET 50% / BASELINE 35% marginals this yields the right overall
    split: among all timesteps the quiet-class is ~85%, 41.2% of which → 35%
    BASELINE and 58.8% → 50% QUIET.

    Args:
        dataset: an ``LmaDataset`` yielding ``(l3 [21,313], labels [8,313])``.
        n_windows: max number of windows to sample.
        baseline_frac_of_quiet: target fraction of quiet timesteps → BASELINE.
        seed: RNG seed for the window subsample (reproducible calibration).

    Returns:
        The RMS threshold (float). ``rms > threshold`` ⇒ BASELINE.
    """
    assert hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"), \
        "dataset must be a torch Dataset (len + getitem)"
    assert isinstance(n_windows, int) and n_windows > 0, \
        f"n_windows must be positive int, got {n_windows!r}"
    assert 0.0 < baseline_frac_of_quiet < 1.0, \
        f"baseline_frac_of_quiet must be in (0,1), got {baseline_frac_of_quiet!r}"

    n_total = len(dataset)
    assert n_total > 0, "dataset is empty — cannot calibrate"
    n_sample = min(n_windows, n_total)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n_total, size=n_sample, replace=False)

    quiet_rms_chunks: list[np.ndarray] = []
    n_quiet = n_seen = 0
    for i in idxs:
        l3_t, labels_t = dataset[int(i)]
        l3 = l3_t.numpy() if isinstance(l3_t, torch.Tensor) else np.asarray(l3_t)
        labels = labels_t.numpy() if isinstance(labels_t, torch.Tensor) else np.asarray(labels_t)
        assert l3.shape[0] == 21 and labels.shape[0] == 8, \
            f"unexpected window shapes l3={l3.shape} labels={labels.shape}"
        n_seen += 1
        max3 = labels.max(axis=0)  # [T]
        rms = _l3_rms_pooled(l3, max3.shape[0])
        q = rms[max3 == 0]
        if q.size:
            quiet_rms_chunks.append(q)
            n_quiet += q.size

    assert quiet_rms_chunks, \
        "no QUIET (max3==0) timesteps found in the sampled windows — cannot " \
        "calibrate the QUIET/BASELINE threshold"
    quiet_rms = np.concatenate(quiet_rms_chunks)

    # We want the top `baseline_frac_of_quiet` of quiet RMS to be BASELINE,
    # so the threshold is the (1 - frac) percentile of the quiet RMS dist.
    pct = 100.0 * (1.0 - baseline_frac_of_quiet)
    threshold = float(np.percentile(quiet_rms, pct))
    assert np.isfinite(threshold), f"computed threshold is non-finite: {threshold}"
    return threshold


# ----------------------------------------------------------------------
# Distribution report — used by the __main__ smoke and the trainer.
# ----------------------------------------------------------------------

def state_distribution(dataset: "LmaDataset",
                       quiet_rms_threshold: float,
                       n_windows: int = 2000,
                       seed: int = 1337) -> np.ndarray:
    """Empirical 4-state marginal distribution over sampled windows.

    Returns a length-4 float array summing to 1.0 (fractions of timesteps in
    QUIET / BASELINE / INTERESTING / CRITICAL).
    """
    n_total = len(dataset)
    assert n_total > 0, "dataset is empty"
    n_sample = min(n_windows, n_total)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n_total, size=n_sample, replace=False)

    counts = np.zeros(NUM_STATES, dtype=np.int64)
    for i in idxs:
        l3_t, labels_t = dataset[int(i)]
        l3 = l3_t.numpy() if isinstance(l3_t, torch.Tensor) else np.asarray(l3_t)
        labels = labels_t.numpy() if isinstance(labels_t, torch.Tensor) else np.asarray(labels_t)
        tgt = derive_4state_target(labels, l3, quiet_rms_threshold)
        binc = np.bincount(tgt, minlength=NUM_STATES)
        counts += binc[:NUM_STATES]

    total = int(counts.sum())
    assert total > 0, "no timesteps counted"
    return counts.astype(np.float64) / total


# ----------------------------------------------------------------------
# __main__ smoke — calibrate on split_manifest_v9 val and report.
# ----------------------------------------------------------------------

def _main() -> None:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="Calibrate the QUIET/BASELINE RMS threshold and report the "
                    "resulting 4-state distribution.")
    p.add_argument("--split-manifest", type=Path,
                   default=Path("/mnt/4tb/data/Training/manifests/split_manifest_v9.json"))
    p.add_argument("--lma-root", type=Path, nargs="+", default=[
        Path("/mnt/4tb/data/Training/lma"),
        Path("/mnt/4tb/data/Archive/lma/physionet/chbmit.lma"),
        Path("/mnt/4tb/data/Archive/lma/physionet/eegmmidb.lma"),
        Path("/mnt/4tb/data/Archive/lma/physionet/mental_arithmetic.lma"),
        Path("/mnt/4tb/data/Archive/lma/tuh/tuep_v3.1.0.lma"),
    ])
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--n-windows", type=int, default=2000)
    p.add_argument("--max-windows-per-file", type=int, default=5)
    args = p.parse_args()

    from lamquant.snn.lma_dataset import LmaDataset

    # Expand roots to explicit .lma paths (mirror the trainer's logic).
    lma_paths: list[Path] = []
    for r in args.lma_root:
        r = Path(r)
        if r.is_file() and r.suffix == ".lma":
            lma_paths.append(r)
        elif r.is_dir():
            found = sorted(r.glob("*/*.lma")) or sorted(r.glob("*.lma"))
            lma_paths.extend(found)
    seen: set = set()
    lma_paths = [q for q in lma_paths if not (str(q) in seen or seen.add(str(q)))]
    print(f"[four_state] {len(lma_paths)} .lma archive(s); split={args.split}")

    ds = LmaDataset(lma_paths=lma_paths, split=args.split,
                    split_manifest_path=args.split_manifest,
                    max_windows_per_file=args.max_windows_per_file)
    print(f"[four_state] dataset windows: {len(ds)}")

    thr = calibrate_quiet_threshold(ds, n_windows=args.n_windows)
    print(f"[four_state] calibrated quiet_rms_threshold = {thr:.6g}")

    dist = state_distribution(ds, thr, n_windows=args.n_windows)
    print("[four_state] 4-state distribution (achieved vs target):")
    for k in range(NUM_STATES):
        print(f"    {STATE_NAMES[k]:<11s} {dist[k]*100:6.2f}%   "
              f"(target {TARGET_DIST_4[k]*100:5.1f}%)   "
              f"L={LEVEL_TABLE_4[k]}  CR={CR_TABLE_4[k]:.0f}:1")
    implied_cr = float(sum(dist[k] * CR_TABLE_4[k] for k in range(NUM_STATES)))
    print(f"[four_state] label-implied mean CR = {implied_cr:.1f}:1")


if __name__ == "__main__":
    _main()
