"""lma_dataset.py — LMA-direct training dataset for SNN.

Reads per-recording `.lma` archives via the Rust PyO3 extension:
  - `lamquant_core.lma_read_entry(lma_path, '<stem>.lml')` → container bytes
  - `lamquant_core.container_read_bytes(bytes)` → (signal int64, metadata)

then runs the SAME preprocessing pipeline as `convert_lml`
(`ai_models/dataset_sim/preprocess.py`) so per-window L3 matches the
existing Q31+L3 pipeline bit-exact:
  digital → microvolts → 21ch select → resample 250 Hz → highpass 0.5 Hz
  → Q31 normalize → de-quantize → `preprocess_subband_single`.

Contract matches `SubbandActivityDataset` so the training loop is a
drop-in swap (`(signal_l3 [21,313] float32, labels [8,L3_T] int64)`).

Worker-local cache: a tiny LRU (default 2 entries) of fully-processed
**float32 signals** per LMA, so multiple windows from the same LMA
re-use the decode + preprocess work. L3 itself is not cached — it's
fast given the cached signal.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, Sampler

# Lazy imports inside __init__ to keep `import lma_dataset` light.

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

LOG = logging.getLogger(__name__)

# Shape constants — must match SubbandActivityDataset
WINDOW_SAMPLES = 2500          # 10 s @ 250 Hz
TARGET_SR = 250.0              # Hz
TARGET_CHANNELS = 21
L3_T = 313                     # preprocess_subband_single output time dim
LABEL_PER_WINDOW = 312         # 2500 // 8 (stride-8 SNN)
Q31_HEADROOM = 0.72            # convert_lml default
HIGHPASS_HZ = 0.5

# Worker-local cache cap. We cache the L3 stack per LMA, not the raw
# signal — L3 is ~3 MB per LMA (1500 windows × 21 × 313 × 4 = 39 MB
# worst case for 3.6 hr TUEG) so cap 8 fits comfortably in ~100-300 MB
# per worker. Computing L3 for all windows of an LMA in one shot
# amortizes the LML decode + DWT cost across every window we'll need
# from that LMA, vs paying the DWT per window with a signal cache.
L3_CACHE_CAP = 8


# ----------------------------------------------------------------------
# Preprocessing helpers — mirror convert_lml exactly
# ----------------------------------------------------------------------

def _lazy_imports():
    """Load heavy deps once per worker."""
    global _lamquant_core, _channel_resolver, _lml_digital_to_float
    global _preprocess_subband_single, _butter_sos, _resample_poly
    global _CHANNEL_PRESETS
    if "_lamquant_core" in globals():
        return
    import lamquant_core as _lamquant_core
    from lamquant.dataset.preprocess import _lml_digital_to_float as _ldtf
    from lamquant.dataset.preprocess import CHANNEL_PRESETS as _CP
    from lamquant_codec import channel_resolver as _cr
    from lamquant.student.subband_preprocess import preprocess_subband_single as _psbs
    from scipy.signal import butter, resample_poly
    _channel_resolver = _cr
    _lml_digital_to_float = _ldtf
    _preprocess_subband_single = _psbs
    _butter_sos = butter
    _resample_poly = resample_poly
    _CHANNEL_PRESETS = _CP


def _highpass_sos():
    """Compile + cache the highpass biquad filter (rate-independent shape).

    Cached at module level so each worker pays the compile cost once.
    """
    if not hasattr(_highpass_sos, "_cache"):
        from scipy.signal import butter
        _highpass_sos._cache = butter(2, HIGHPASS_HZ, btype="high",
                                       fs=TARGET_SR, output="sos")
    return _highpass_sos._cache


def _decode_and_preprocess(lma_path: str, stem: str,
                           lml_internal: Optional[str] = None) -> Optional[np.ndarray]:
    """Delegate to canonical ``lamquant_codec.training.decode_lma_signal``.

    G3 (2026-05-18): the old 70-line decode lived here as a duplicate of
    the canonical codec-local version. It carried the legacy
    ``container_read_bytes -> np.asarray(int64) -> np.float64`` chain
    that peaked at ~18 GB on an 8 hr 27-ch TUEG file. Now we forward to
    the canonical path which uses ``container_read_phys_f32`` (Rust f32
    decode + per-channel calibration in one pass, ~12 GB peak).

    Behaviour + return contract unchanged: ``Optional[np.ndarray]`` of
    shape ``[21, T_resampled] float32``, same Q31 round-trip,
    bit-exact L3 at the output of ``preprocess_subband_single``.
    """
    from lamquant_codec.training.lma_dataset import (
        decode_lma_signal as _canonical_decode,
    )
    return _canonical_decode(lma_path, stem, lml_entry_name=lml_internal)


# ----------------------------------------------------------------------
# Worker-local LRU on L3 stacks (per-process; DataLoader workers each
# have their own). We cache the L3 stack [n_windows, 21, 313] instead
# of the raw signal: one decode + one DWT pass per LMA, then every
# window from that LMA is a cheap array slice. Transient None results
# (decode failure) are NOT cached — V4 Pro 2026-05-16 cache-poisoning
# fix.
# ----------------------------------------------------------------------
import collections as _collections

_L3_CACHE: "_collections.OrderedDict[Tuple[str, str], np.ndarray]" = \
    _collections.OrderedDict()

# Per-worker label-NPZ cache (O3). __getitem__ used to call
# `lamquant_core.lma_read_entry(lma, label.npz)` PER WINDOW, even when
# LmaGroupedSampler hands 5 consecutive windows from the same LMA.
# Cache the decompressed activity array per LMA and serve those 4
# follow-up windows from RAM (each activity array is ~few KB, so cap=8
# fits comfortably in memory). Key: (lma_path, label_npz_name).
LABEL_CACHE_CAP = 8
_LABEL_CACHE: "_collections.OrderedDict[Tuple[str, str], np.ndarray]" = \
    _collections.OrderedDict()


def _label_cache_dir() -> Optional[Path]:
    """Optional disk-staged label cache (W1 perf workaround).

    ``lma_read_entry`` on per-dataset LMAs costs ~150 ms per call
    because the Rust side re-parses the manifest each time. For a 64 K
    train-set init that adds up to >2.5 h. When ``LMA_LABEL_CACHE_DIR``
    points at a pre-staged dir of ``<stem>_labels.npz`` (see
    ``scripts/w1_stage_labels_v2.py``), both ``__init__`` and
    ``__getitem__`` read directly from disk (<1 ms each) and skip the
    LMA round-trip entirely.
    """
    d = os.environ.get("LMA_LABEL_CACHE_DIR",
                       "/mnt/4tb/data/Training/labels")
    return Path(d) if d else None


def _fadvise_hint(path: Path) -> None:
    """Tell the kernel a file is about to be read sequentially in
    full (POSIX_FADV_SEQUENTIAL + POSIX_FADV_WILLNEED). Triggers
    aggressive readahead so the upcoming `np.load(mmap_mode='r')`
    + window slice both hit RAM. Cheap (one open + two syscalls +
    close) and silently no-ops on platforms / FSes that don't
    implement posix_fadvise (most do; tmpfs / FUSE may not).
    """
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        finally:
            os.close(fd)
    except OSError:
        pass  # FS doesn't support fadvise — silent fallback


def _detail_bands_cfg() -> Tuple[str, ...]:
    """Detail subband channels to append to L3, from the SNN_DETAIL_BANDS env.

    Empty (default) -> L3 only, behaviour identical to before. Set to e.g.
    "l3_detail" (the 15.6-31.25 Hz LVFA band) or
    "l3_detail,l2_detail,l1_detail" (full >15 Hz reconstruction basis) to
    append those bands as extra input channels, pooled to the L3 313 grid.
    Used by the oracle ceiling probe and the L3+details deployable arm.
    """
    raw = os.environ.get("SNN_DETAIL_BANDS", "").strip()
    return tuple(b for b in (s.strip() for s in raw.split(",")) if b)


def _stack_detail_bands(l3: np.ndarray, subs: list) -> np.ndarray:
    """Append env-selected detail subbands to L3 as extra channels.

    Args:
        l3: ``[21, 313]`` level-3 DWT approximation (<=15.6 Hz).
        subs: list of 21 per-channel dicts with detail bands ('l3_detail'
            15.6-31.25, 'l2_detail' 31.25-62.5, 'l1_detail' 62.5-125 Hz),
            as returned by ``preprocess_subband_single``.

    Returns:
        ``[21*(1+k), 313]`` float32, k = number of selected bands. With no
        bands selected (default) returns the bare ``[21, 313]`` L3 — identical
        to the prior behaviour, so existing callers are unaffected.
    """
    l3 = np.asarray(l3, dtype=np.float32)
    bands = _detail_bands_cfg()
    if not bands:
        return l3
    T = l3.shape[1]
    out = [l3]
    for band in bands:
        mat = np.empty((len(subs), T), dtype=np.float32)
        for c, d in enumerate(subs):
            v = np.asarray(d[band], dtype=np.float32)
            if v.shape[0] == T:
                mat[c] = v
            else:
                # resample the band coefficients onto the 313 grid (linear)
                mat[c] = np.interp(
                    np.linspace(0.0, 1.0, T, dtype=np.float64),
                    np.linspace(0.0, 1.0, v.shape[0], dtype=np.float64),
                    v.astype(np.float64),
                ).astype(np.float32)
        out.append(mat)
    return np.concatenate(out, axis=0)


def _compute_l3_stack(lma_path: str, stem: str,
                      lml_internal: Optional[str] = None) -> Optional[np.ndarray]:
    """Decode + preprocess signal then run preprocess_subband_single
    on every 2500-sample window in one pass. Returns L3 stack shape
    [n_windows, 21, 313] float32, or None on decode failure.

    Per-dataset LMA layout: pass ``lml_internal`` to point at the LML
    entry path inside the archive (e.g. ``edf/000/<subj>/.../<stem>.lml``).
    When omitted, falls back to per-stem ``<stem>.lml`` for legacy archives.
    """
    _lazy_imports()
    signal = _decode_and_preprocess(lma_path, stem, lml_internal=lml_internal)
    if signal is None:
        return None
    T = signal.shape[1]
    n_full = T // WINDOW_SAMPLES
    if n_full == 0:
        # Recording shorter than one window — pad and emit one window.
        window = np.zeros((TARGET_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
        window[:, :T] = signal[:, :T]
        l3, _, subs = _preprocess_subband_single(window, order=8, autocorr_len=256)
        return np.expand_dims(_stack_detail_bands(l3, subs), 0)
    l3_list = []
    for w in range(n_full):
        s = w * WINDOW_SAMPLES
        e = s + WINDOW_SAMPLES
        window = signal[:, s:e].astype(np.float32)
        l3, _, subs = _preprocess_subband_single(window, order=8, autocorr_len=256)
        l3_list.append(_stack_detail_bands(l3, subs))
    return np.stack(l3_list, axis=0)


def _l3_cache_dir() -> Optional[Path]:
    """Optional on-disk L3 cache root. Opt-in via env var.

    When `L3_CACHE_DIR` is set, `_cached_l3_stack` stores the L3 stack
    per LMA as `<dir>/<stem>.npy` after first compute, and mmap-loads
    on subsequent epochs. Cuts per-LMA fetch from ~3 s (full decode +
    resample + highpass + DWT) to ~1-5 ms (mmap open + index). Typical
    cache size: ~125 GB (float16) or ~250 GB (float32) for the full
    71 K LMA corpus.

    Unset → in-memory LRU only (legacy behaviour).
    """
    d = os.environ.get("L3_CACHE_DIR")
    if not d:
        return None
    return Path(d)


def _l3_cache_dtype() -> np.dtype:
    """Storage dtype for on-disk L3 cache (default float16).

    Override via `L3_CACHE_DTYPE` env var. Supported values:
      - `float16` (default): halves disk footprint vs float32.
        Noise floor ~1e-3 of input scale. **Safe for SNN** (INT8 /
        W2A8 quantization post-train dominates the noise floor) and
        for QAT phases.
      - `float32`: full precision. **Use for fp32 warmup phases**
        in encoder / teacher / decoder training where gradient SNR
        matters before QAT kicks in.

    If a cache file exists at a DIFFERENT dtype than the current
    setting, load returns whatever dtype is on disk; the cast back to
    float32 happens in `__getitem__`. To switch dtypes, wipe the
    cache dir + rebuild via `scripts/prewarm_l3_cache.py`.
    """
    raw = os.environ.get("L3_CACHE_DTYPE", "float16").lower().strip()
    if raw in ("float16", "f16", "half", "fp16"):
        return np.dtype(np.float16)
    if raw in ("float32", "f32", "single", "fp32"):
        return np.dtype(np.float32)
    LOG.warning("L3_CACHE_DTYPE=%r not recognized; falling back to float16. "
                "Valid: float16, float32.", raw)
    return np.dtype(np.float16)


def _cached_l3_stack(lma_path: str, stem: str,
                     lml_internal: Optional[str] = None) -> Optional[np.ndarray]:
    """Two-tier cache around `_compute_l3_stack`:

      1. In-memory LRU (`_L3_CACHE`, cap L3_CACHE_CAP per worker) —
         hot in-epoch repeats.
      2. Optional on-disk `.npy` per LMA when `L3_CACHE_DIR` is set —
         survives across epochs + workers. mmap-loaded so each
         `__getitem__` reads only the requested window slice from
         disk page cache. Skips the full LMA decode + resample +
         highpass + DWT chain (5-40 s) entirely on hit.

    Only non-None results are cached at either tier (V4 Pro 2026-05-16
    poisoning fix).
    """
    key = (lma_path, stem)
    cached = _L3_CACHE.get(key)
    if cached is not None:
        _L3_CACHE.move_to_end(key)
        return cached

    # Disk-tier check.
    disk_dir = _l3_cache_dir()
    disk_path: Optional[Path] = None
    if disk_dir is not None:
        disk_path = disk_dir / f"{stem}.npy"
        if disk_path.exists():
            try:
                # O4 (reverted 2026-05-19) — posix_fadvise hint here
                # regressed wall time by 15-29% in bench O3_O4. Likely
                # cause: 4 extra syscalls per cache hit × 64 K LMAs
                # outweighs any readahead win on already-page-cached
                # files. Definition kept for opt-in via env if a future
                # cold-cache workload would benefit.
                # mmap_mode='r' keeps the file paged in lazily —
                # `__getitem__` only touches the bytes of the
                # requested window slice.
                stack = np.load(disk_path, mmap_mode="r")
                # Promote to in-memory LRU so subsequent windows in
                # the same epoch hit RAM, not disk.
                _L3_CACHE[key] = stack
                if len(_L3_CACHE) > L3_CACHE_CAP:
                    _L3_CACHE.popitem(last=False)
                return stack
            except Exception as e:
                LOG.warning("L3 disk-cache load failed for %s: %s — recomputing",
                            stem, e)

    # Cache miss at both tiers — full decode.
    result = _compute_l3_stack(lma_path, stem, lml_internal=lml_internal)
    if result is None:
        return None

    # Persist to disk for next epoch / next worker / next run.
    # Atomic write: tmp + rename so a crash mid-save doesn't leave
    # a torn file (matches the LMA migration's delete-as-you-go
    # discipline).
    if disk_path is not None:
        try:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            # np.save auto-appends `.npy` when the filename doesn't
            # end in it, so a tmp like `<x>.npy.tmp` ends up written
            # as `<x>.npy.tmp.npy` and the rename target vanishes.
            # Sandwich `.tmp` between the stem and the `.npy` suffix.
            tmp = disk_path.with_name(disk_path.stem + ".tmp.npy")
            # Storage dtype controlled by L3_CACHE_DTYPE env var
            # (default float16, halves disk footprint vs float32).
            # See `_l3_cache_dtype` docstring for guidance.
            np.save(tmp, result.astype(_l3_cache_dtype()))
            tmp.replace(disk_path)
        except Exception as e:
            LOG.warning("L3 disk-cache save failed for %s: %s — in-memory only",
                        stem, e)

    _L3_CACHE[key] = result
    if len(_L3_CACHE) > L3_CACHE_CAP:
        _L3_CACHE.popitem(last=False)
    return result


# ----------------------------------------------------------------------
# Window selection policy
# ----------------------------------------------------------------------

MAX_WINDOWS_PER_FILE = 5
MAX_SEIZURE_WINDOWS_PER_FILE = 10
MIN_BACKGROUND_PER_FILE = 1


def select_windows(activity_labels: np.ndarray,
                   max_windows: int = MAX_WINDOWS_PER_FILE,
                   max_seizure_windows: int = MAX_SEIZURE_WINDOWS_PER_FILE,
                   min_background: int = MIN_BACKGROUND_PER_FILE) -> List[int]:
    """Deterministic per-LMA window selection.

    Algorithm:
      1. Find all windows whose label slice (LABEL_PER_WINDOW=312 cols, no
         overlap) contains seizure class 2.
      2. Keep first `max_seizure_windows` (cap status-epilepticus dominance).
      3. Fill remaining budget with evenly-spaced background windows.
      4. `min_background` only applies when picked_seizures == 0 — ensures
         at least one window is selected for pure-background recordings.
         When seizures saturate `max_seizure_windows`, no background is
         added beyond `max_windows - picked_seizures` (which may be zero
         or negative; clamped at zero).
      5. Return sorted list of window indices.

    Edge cases:
      - 0 seizure windows → up to `max_windows` evenly-spaced background
        (at least `min_background`).
      - All seizure → up to `max_seizure_windows` seizures, 0 background.
      - 0 windows at all → returns [0] (degenerate; caller must check).
    """
    if activity_labels.size == 0:
        return [0]

    n_total = max(1, activity_labels.shape[1] // LABEL_PER_WINDOW)

    seizure_wins: List[int] = []
    for wi in range(n_total):
        lbl_start = wi * LABEL_PER_WINDOW
        # Use LABEL_PER_WINDOW (no overlap) for seizure detection so a
        # seizure in window N doesn't flag window N-1 / N+1 as seizure too.
        lbl_end = min(lbl_start + LABEL_PER_WINDOW, activity_labels.shape[1])
        if lbl_end > lbl_start and np.any(activity_labels[:, lbl_start:lbl_end] == 2):
            seizure_wins.append(wi)

    picked_seizures = seizure_wins[:max_seizure_windows]
    bg_candidates = [i for i in range(n_total) if i not in set(picked_seizures)]

    # Background budget: fill up to max_windows total. min_background floor
    # ONLY when no seizures picked (avoid forcing a bg when seizures
    # already saturate the recording).
    bg_budget = max(0, max_windows - len(picked_seizures))
    if not picked_seizures and bg_candidates:
        bg_budget = max(bg_budget, min_background)
    bg_budget = min(bg_budget, len(bg_candidates))

    if bg_budget > 0:
        bg_idx = np.linspace(0, len(bg_candidates) - 1, bg_budget, dtype=int)
        picked_bg = [bg_candidates[i] for i in bg_idx]
    else:
        picked_bg = []

    selected = sorted(set(picked_seizures) | set(picked_bg))
    return selected if selected else [0]


# ----------------------------------------------------------------------
# Split manifest reader
# ----------------------------------------------------------------------

def load_split_manifest(path: Path, split: Literal["train", "val"]) -> Tuple[List[str], Dict[str, str]]:
    """Return (stems_for_split, subject_id_by_stem).

    Manifest schema:
      {
        "subjects": {subject_id: "train"|"val", ...},
        "stems_by_subject": {subject_id: [stem, stem, ...], ...},
        ...
      }
    """
    if not path.exists():
        raise FileNotFoundError(f"split manifest not found: {path}")
    meta = json.loads(path.read_text())
    subjects = meta["subjects"]
    stems_by_subject = meta.get("stems_by_subject", {})
    stems: List[str] = []
    subject_by_stem: Dict[str, str] = {}
    for sid, assigned_split in subjects.items():
        if assigned_split != split:
            continue
        for stem in stems_by_subject.get(sid, []):
            stems.append(stem)
            subject_by_stem[stem] = sid
    return stems, subject_by_stem


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class LmaDataset(Dataset):
    """LMA-direct training dataset for SNN.

    Random window access via worker-local LRU on the decoded+preprocessed
    signal. Same `(signal, labels)` contract as `SubbandActivityDataset`.

    Refuses to construct if the split manifest is missing OR if a
    holdout subject appears in the train list (rule 30: hostile-caller).
    """

    def __init__(self,
                 lma_paths: Optional[Sequence[Path]] = None,
                 split: Literal["train", "val"] = "train",
                 split_manifest_path: Optional[Path] = None,
                 lma_dir: Optional[Path] = None,
                 lma_root: Optional[Path] = None,  # back-compat alias
                 max_windows_per_file: int = MAX_WINDOWS_PER_FILE,
                 max_seizure_windows_per_file: int = MAX_SEIZURE_WINDOWS_PER_FILE,
                 min_background_per_file: int = MIN_BACKGROUND_PER_FILE,
                 seq_windows: int = 1,
                 require_meta_subject_match: bool = True):  # noqa: arg unused (back-compat)
        """LMA-direct training dataset (Phase M per-dataset layout).

        Accepts EITHER an explicit list of per-dataset LMAs OR a
        directory of them (auto-globs ``*.lma`` one level deep, e.g.
        ``Archive/lma/{tuh,physionet}/*.lma``).

        Old per-stem signature (``lma_root: dir/<stem>.lma``) is
        preserved via the same kwarg but treated as ``lma_dir``.

        Args:
            lma_paths: explicit list of ``.lma`` archive paths.
            lma_dir: directory containing ``<corpus>.lma`` files
                (or ``<source>/<corpus>.lma`` nested one level).
            lma_root: legacy alias for ``lma_dir``.
            split: 'train' or 'val'.
            split_manifest_path: split manifest JSON.
            max_*_per_file / min_background_per_file: seizure-aware
                window selection knobs.
            require_meta_subject_match: NO-OP under per-dataset layout
                (no per-stem meta.json present); kept for back-compat.
        """
        if split_manifest_path is None:
            raise ValueError("split_manifest_path is required")
        split_manifest_path = Path(split_manifest_path)
        if split not in ("train", "val"):
            raise ValueError(
                f"split must be 'train' or 'val', got {split!r}"
            )

        self.seq_windows = int(seq_windows)
        if self.seq_windows < 1:
            raise ValueError(f"seq_windows must be >= 1, got {seq_windows}")

        # Resolve LMA paths.
        if lma_paths is None and (lma_dir is not None or lma_root is not None):
            base = Path(lma_dir if lma_dir is not None else lma_root)
            if not base.exists():
                raise FileNotFoundError(f"lma_dir not found: {base}")
            # Try one-level-deep glob first (Archive/lma/<source>/*.lma);
            # fall back to flat (legacy).
            lma_paths = sorted(base.glob("*/*.lma"))
            if not lma_paths:
                lma_paths = sorted(base.glob("*.lma"))
            if not lma_paths:
                raise RuntimeError(
                    f"no .lma archives found under {base} (looked one and "
                    f"two levels deep)"
                )
        if not lma_paths:
            raise ValueError(
                "must supply either lma_paths=[...] or lma_dir=<path>"
            )
        lma_paths = [Path(p) for p in lma_paths]
        for p in lma_paths:
            if not p.exists():
                raise FileNotFoundError(f"lma path not found: {p}")

        # Load split assignments + subject-bleed check.
        split_stems, subject_by_stem = load_split_manifest(split_manifest_path, split)
        other_split = "val" if split == "train" else "train"
        other_stems, _ = load_split_manifest(split_manifest_path, other_split)
        overlap = set(split_stems) & set(other_stems)
        if overlap:
            raise RuntimeError(
                f"split manifest is corrupt: {len(overlap)} stems in both "
                f"train and val (first 5: {sorted(overlap)[:5]})"
            )

        # Build per-stem entry index across all LMAs (one-shot via `lml ls`).
        # Maps stem -> {"lma": <path>, "lml": <internal>, "labels": <internal>}.
        from lamquant_codec.training import build_lma_entry_index
        LOG.info("[LmaDS:%s] indexing %d per-dataset LMAs ...",
                 split, len(lma_paths))
        entry_idx = build_lma_entry_index([str(p) for p in lma_paths])
        LOG.info("[LmaDS:%s] entry index built: %d stems", split, len(entry_idx))

        # Build window index per stem in our split.
        # Index tuple: (lma_path, stem, win_idx, lml_internal, label_internal)
        self.index: List[Tuple[Path, str, int, str, str]] = []
        # Per-window seizure flag, index-aligned with self.index (B2/B3
        # seizure-balanced sampler). True iff this window's label slice
        # contains class 2 (seizure). Built in the same pass that already
        # scans `activity`, so it costs nothing extra.
        self.seizure_flags: List[bool] = []
        n_seen = n_missing_lma = n_no_labels = 0
        n_seizure_windows = 0

        # Group split stems by their LMA so we open each archive once and
        # iterate its entries linearly. Without this, lma_read_entry on a
        # 700 GB TUEG LMA re-parses the manifest per call (~300 ms each
        # → 4 h for the train init alone).
        from collections import defaultdict as _dd
        import lamquant_core as _lc
        # Labels may live INSIDE the archive (M2 per-dataset packing) OR in
        # the disk-staged cache (_label_cache_dir, default
        # /mnt/4tb/data/Training/labels). Plain `lml encode` archives bundle
        # the source + sibling annotations but NOT the activity_labels NPZ —
        # those are training metadata, not part of the lossless recording. So
        # a stem with no in-archive label entry is still usable when its
        # <stem>_labels.npz exists in the disk cache; both __getitem__ and
        # iter_labels_only already prefer that cache and only fall back to
        # lma_read_entry. The sentinel label_internal below is never read via
        # lma_read_entry on the happy path (cached.exists() short-circuits).
        _idx_label_cache = _label_cache_dir()
        by_lma: dict = _dd(list)
        for stem in split_stems:
            info = entry_idx.get(stem)
            if info is None:
                n_missing_lma += 1
                continue
            label_internal = info["labels"]
            if label_internal is None:
                if (_idx_label_cache is not None
                        and (_idx_label_cache / f"{stem}_labels.npz").exists()):
                    label_internal = f"__diskcache__/{stem}_labels.npz"
                else:
                    n_no_labels += 1
                    continue
            by_lma[info["lma"]].append((stem, info["lml"], label_internal))

        label_cache = _label_cache_dir()
        for lma_str, items in by_lma.items():
            lma_path = Path(lma_str)
            for stem, lml_internal, label_internal in items:
                # Prefer disk-staged label NPZ over lma_read_entry round-trip.
                cached = (label_cache / f"{stem}_labels.npz") if label_cache else None
                try:
                    if cached is not None and cached.exists():
                        with np.load(cached, allow_pickle=True) as ld:
                            activity = np.asarray(ld["activity_labels"])
                    else:
                        label_bytes = _lc.lma_read_entry(lma_str, label_internal)
                        with np.load(io.BytesIO(label_bytes), allow_pickle=True) as ld:
                            activity = np.asarray(ld["activity_labels"])
                except Exception as e:
                    LOG.warning("labels unreadable for %s: %s", stem, e)
                    continue

                selected = select_windows(
                    activity,
                    max_windows=max_windows_per_file,
                    max_seizure_windows=max_seizure_windows_per_file,
                    min_background=min_background_per_file,
                )
                if self.seq_windows > 1:
                    # Cross-window sequence slots: K CONSECUTIVE windows so the
                    # SSM scan carries state across the 10 s window boundaries
                    # (ADR-0027 temporal-context lever). win_idx stores the START
                    # window; __getitem__ concatenates [start, start+K).
                    K = self.seq_windows
                    n_win = activity.shape[1] // LABEL_PER_WINDOW
                    if n_win < K:
                        n_seen += 1
                        continue
                    stride = max(1, K // 2)              # 50% overlap for coverage
                    sz_slots, bg_slots = [], []
                    for w in range(0, n_win - K + 1, stride):
                        s = w * LABEL_PER_WINDOW
                        # span of K windows in label space; activity is 3-class
                        # (0 bg / 1 active / 2 SEIZURE), matching derive_4state_target
                        # max3==2 -> CRITICAL and the per-window path below.
                        e = min((w + K) * LABEL_PER_WINDOW, activity.shape[1])
                        is_sz = bool(e > s and np.any(activity[:, s:e] == 2))
                        (sz_slots if is_sz else bg_slots).append(w)
                    bg_budget = max(min_background_per_file,
                                    max_windows_per_file // K)
                    if len(bg_slots) > bg_budget:
                        keep = np.linspace(0, len(bg_slots) - 1, bg_budget, dtype=int)
                        bg_slots = [bg_slots[i] for i in keep]
                    for w in sorted(set(sz_slots) | set(bg_slots)):
                        is_sz = w in set(sz_slots)
                        if is_sz:
                            n_seizure_windows += 1
                        self.index.append((lma_path, stem, w, lml_internal, label_internal))
                        self.seizure_flags.append(is_sz)
                    n_seen += 1
                    continue
                for wi in selected:
                    lbl_start = wi * LABEL_PER_WINDOW
                    lbl_end = min(lbl_start + L3_T, activity.shape[1])
                    is_sz = bool(
                        lbl_end > lbl_start
                        and np.any(activity[:, lbl_start:lbl_end] == 2)
                    )
                    if is_sz:
                        n_seizure_windows += 1
                    self.index.append((lma_path, stem, wi, lml_internal, label_internal))
                    self.seizure_flags.append(is_sz)
                n_seen += 1

        self._max_windows_per_file = max_windows_per_file
        LOG.info("[LmaDS:%s] %d stems indexed, %d windows total (%d seizure); "
                 "skipped %d missing_lma, %d no_labels",
                 split, n_seen, len(self.index), n_seizure_windows,
                 n_missing_lma, n_no_labels)

        if not self.index:
            raise RuntimeError(
                f"LmaDataset(split={split!r}) produced 0 windows — check "
                f"lma_paths={lma_paths!r}, manifest={split_manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spawn workers don't carry parent's globals; `_compute_l3_stack`
        # calls `_lazy_imports()` but a CACHE HIT skips it, so the label
        # path below would see `_lamquant_core` undefined.
        _lazy_imports()
        lma_path, stem, win_idx, lml_internal, label_internal = self.index[idx]

        # L3 stack cached per LMA: one decode + DWT pass per LMA, then
        # every window for that LMA is a cheap slice.  Cache key is the
        # stem (corpus-unique under the per-dataset layout), so existing
        # on-disk `<stem>.npy` files in Training/l3_cache/ stay valid.
        l3_stack = _cached_l3_stack(str(lma_path), stem, lml_internal=lml_internal)
        K = self.seq_windows
        ch = l3_stack.shape[1] if l3_stack is not None else TARGET_CHANNELS
        if K == 1:
            if l3_stack is None or win_idx >= l3_stack.shape[0]:
                l3 = np.zeros((ch, L3_T), dtype=np.float32)
            else:
                l3 = np.asarray(l3_stack[win_idx], dtype=np.float32)
        else:
            # Cross-window: concatenate K CONSECUTIVE windows -> [ch, K*L3_T].
            # win_idx is the START window; the SSM scans the whole span so its
            # state carries across the 10 s boundaries.
            l3 = np.zeros((ch, L3_T * K), dtype=np.float32)
            if l3_stack is not None:
                for k in range(K):
                    wk = win_idx + k
                    if wk < l3_stack.shape[0]:
                        l3[:, k * L3_T:(k + 1) * L3_T] = np.asarray(
                            l3_stack[wk], dtype=np.float32)

        # Read labels NPZ for this window. With LmaGroupedSampler the
        # same `(lma_path, label_internal)` shows up 5x in a row; cache
        # the activity array so 4 of those 5 hits skip the
        # lma_read_entry + NPZ decompress (~5x fewer disk hits).
        lbl_key = (str(lma_path), label_internal)
        activity = _LABEL_CACHE.get(lbl_key)
        if activity is not None:
            _LABEL_CACHE.move_to_end(lbl_key)
        else:
            label_cache = _label_cache_dir()
            cached = (label_cache / f"{stem}_labels.npz") if label_cache else None
            try:
                if cached is not None and cached.exists():
                    with np.load(cached, allow_pickle=True) as ld:
                        activity = np.asarray(ld["activity_labels"])
                else:
                    label_bytes = _lamquant_core.lma_read_entry(str(lma_path), label_internal)
                    with np.load(io.BytesIO(label_bytes), allow_pickle=True) as ld:
                        activity = np.asarray(ld["activity_labels"])
            except Exception as e:
                LOG.warning("labels load failed at __getitem__ for %s: %s", stem, e)
                activity = np.zeros((8, L3_T), dtype=np.int64)
            _LABEL_CACHE[lbl_key] = activity
            if len(_LABEL_CACHE) > LABEL_CACHE_CAP:
                _LABEL_CACHE.popitem(last=False)

        labels_window = np.zeros((8, L3_T * K), dtype=np.int64)
        for k in range(K):
            wk = win_idx + k
            lbl_start = wk * LABEL_PER_WINDOW
            lbl_end = min(lbl_start + L3_T, activity.shape[1])
            o = k * L3_T
            if lbl_end > lbl_start:
                lbl_len = lbl_end - lbl_start
                labels_window[:, o:o + lbl_len] = activity[:, lbl_start:lbl_end].astype(np.int64)
                if lbl_len < L3_T:
                    labels_window[:, o + lbl_len:o + L3_T] = \
                        labels_window[:, o + lbl_len - 1:o + lbl_len]

        return torch.from_numpy(l3), torch.from_numpy(labels_window)


def iter_labels_only(dataset: "LmaDataset"):
    """Stream per-window label arrays without touching the L3 cache.

    Built for the trainer's pos_weight scan, which only needs the
    label tensor (not the L3 signal). Going through `__getitem__`
    forces a full L3 stack load per window — the slow path. This
    helper groups index entries by `(lma_path, label_npz_name)`,
    reads each label NPZ exactly once via `lamquant_core.lma_read_entry`,
    then yields the per-window slice for every index that points into it.

    Yields:
        labels_window: np.ndarray shape (8, L3_T) dtype int64 —
                       identical to the second element of __getitem__.
    """
    _lazy_imports()
    label_cache = _label_cache_dir()
    # Group: same (lma_path, label_internal) → load NPZ once.
    by_lma: Dict[tuple, List[Tuple[int, int, str]]] = defaultdict(list)
    for idx, entry in enumerate(dataset.index):
        # Tuple shape: (lma_path, stem, win_idx, lml_internal, label_internal)
        lma_path = entry[0]
        stem = entry[1]
        win_idx = entry[2]
        label_internal = entry[4] if len(entry) >= 5 else entry[3]
        by_lma[(str(lma_path), label_internal)].append((idx, win_idx, stem))

    for (lma_path, label_internal), entries in by_lma.items():
        # Prefer staged disk cache; fall back to LMA read.
        stem_for_cache = entries[0][2]
        cached = (label_cache / f"{stem_for_cache}_labels.npz") if label_cache else None
        try:
            if cached is not None and cached.exists():
                with np.load(cached, allow_pickle=True) as ld:
                    activity = np.asarray(ld["activity_labels"])
            else:
                label_bytes = _lamquant_core.lma_read_entry(lma_path, label_internal)
                with np.load(io.BytesIO(label_bytes), allow_pickle=True) as ld:
                    activity = np.asarray(ld["activity_labels"])
        except Exception as e:
            LOG.warning("labels load failed in iter_labels_only for %s: %s",
                        Path(lma_path).stem, e)
            activity = np.zeros((8, L3_T), dtype=np.int64)

        for _idx, win_idx, _stem in entries:
            lbl_start = win_idx * LABEL_PER_WINDOW
            lbl_end = min(lbl_start + L3_T, activity.shape[1])
            labels_window = np.zeros((8, L3_T), dtype=np.int64)
            if lbl_end > lbl_start:
                lbl_len = lbl_end - lbl_start
                labels_window[:, :lbl_len] = activity[:, lbl_start:lbl_end].astype(np.int64)
                if lbl_len < L3_T:
                    labels_window[:, lbl_len:] = labels_window[:, lbl_len - 1:lbl_len]
            yield labels_window


class LmaGroupedSampler(Sampler[int]):
    """Yield dataset indices grouped by LMA.

    DataLoader with `shuffle=True` scatters windows across all 64K
    LMAs, defeating the per-worker L3 cache (cap 8 vs 64K LMAs =
    near-zero hit rate). With this sampler, every window from one LMA
    is yielded contiguously: first window is a cache miss (decode +
    DWT), remaining ~4 windows hit the cache. ~5× throughput.

    Shuffle is applied at the LMA-group level (which LMA next), not
    within-group (windows stay sorted by win_idx — irrelevant for the
    SNN since each window is an independent sample).

    Args:
        dataset: an `LmaDataset` instance.
        shuffle: shuffle the order of LMAs each epoch.
        seed: base seed; per-epoch shuffle uses `seed + epoch`.
    """

    def __init__(self, dataset: "LmaDataset", shuffle: bool = True,
                 seed: int = 42):
        if not isinstance(dataset, LmaDataset):
            raise TypeError(
                f"LmaGroupedSampler requires LmaDataset, got "
                f"{type(dataset).__name__}"
            )
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group dataset indices by lma_path (preserves win_idx order within group)
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, entry in enumerate(dataset.index):
            lma_path = entry[0]
            groups[str(lma_path)].append(idx)
        self.groups: List[List[int]] = list(groups.values())
        self._total = sum(len(g) for g in self.groups)

    def set_epoch(self, epoch: int) -> None:
        """PyTorch DistributedSampler convention. Call before each epoch
        to vary the shuffle seed across epochs."""
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            order = rng.permutation(len(self.groups))
        else:
            order = range(len(self.groups))
        for gi in order:
            yield from self.groups[gi]

    def __len__(self) -> int:
        return self._total


class SeizureBalancedSampler(Sampler[int]):
    """Interleave seizure-bearing and background windows to a target fraction.

    B2 + B3 (run-2 2026-05-29). The SNN never got the clinical-style balanced
    sampler. With the natural ~18% seizure-window rate (and `max_windows`
    capping background per file), most batches in Run #1 carried few or zero
    seizure windows, so the gated seizure loss vanished and the optimizer
    drifted into the QUIET collapse. This sampler guarantees every batch hits
    a target seizure-window fraction.

    Curriculum (B3): the target fraction starts at `start_frac` (default 0.5)
    for the warmup phase and anneals linearly toward `natural_frac` (~0.18)
    over `anneal_epochs`, so val specificity stays calibrated to the real
    operating distribution once the seizure signal is consolidated. Call
    `set_epoch(e)` before each epoch (the trainer does).

    Seizure windows are drawn WITH replacement when they would otherwise run
    out (the rare class), so a high target fraction does not truncate the
    epoch. Length is fixed to the dataset size so step count per epoch is
    stable for the LR schedule.

    LMA-grouping (cache locality) is sacrificed for balance per the plan —
    use the on-disk L3 cache (`L3_CACHE_DIR`) to offset the lost group
    locality.
    """

    def __init__(self, dataset: "LmaDataset", start_frac: float = 0.5,
                 natural_frac: float = 0.18, anneal_epochs: int = 20,
                 seed: int = 42):
        if not isinstance(dataset, LmaDataset):
            raise TypeError(
                f"SeizureBalancedSampler requires LmaDataset, got "
                f"{type(dataset).__name__}"
            )
        flags = getattr(dataset, "seizure_flags", None)
        if flags is None or len(flags) != len(dataset):
            raise ValueError(
                "LmaDataset.seizure_flags missing or misaligned — rebuild "
                "the dataset index (expected one flag per window)"
            )
        self.dataset = dataset
        self.start_frac = float(np.clip(start_frac, 0.0, 1.0))
        self.natural_frac = float(np.clip(natural_frac, 0.0, 1.0))
        self.anneal_epochs = max(0, int(anneal_epochs))
        self.seed = seed
        self.epoch = 0

        flags_arr = np.asarray(flags, dtype=bool)
        self.sz_idx = np.nonzero(flags_arr)[0]
        self.bg_idx = np.nonzero(~flags_arr)[0]
        self._total = len(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def current_frac(self) -> float:
        """Target seizure-window fraction for the current epoch (B3 anneal)."""
        if self.anneal_epochs <= 0 or self.epoch >= self.anneal_epochs:
            return self.natural_frac
        t = self.epoch / float(self.anneal_epochs)
        return self.start_frac + t * (self.natural_frac - self.start_frac)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        frac = self.current_frac()
        n = self._total
        n_sz = int(round(frac * n))
        n_bg = n - n_sz

        # Degenerate splits: if one class is empty, fall back to sampling the
        # other (defensive — a val split with zero seizures should never use
        # this sampler, but never raise mid-epoch).
        if len(self.sz_idx) == 0:
            n_sz, n_bg = 0, n
        if len(self.bg_idx) == 0:
            n_sz, n_bg = n, 0

        picks = []
        if n_sz > 0:
            replace_sz = n_sz > len(self.sz_idx)
            picks.append(rng.choice(self.sz_idx, size=n_sz, replace=replace_sz))
        if n_bg > 0:
            replace_bg = n_bg > len(self.bg_idx)
            picks.append(rng.choice(self.bg_idx, size=n_bg, replace=replace_bg))
        order = np.concatenate(picks) if picks else np.arange(n)
        rng.shuffle(order)
        yield from (int(i) for i in order)

    def __len__(self) -> int:
        return self._total
