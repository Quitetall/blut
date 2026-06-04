"""lma_typed_adapter.py — neural-side bridge: LMA-direct → typed batches.

Why this exists
---------------
``train_joint.py`` (the TNN / joint encoder+decoder trainer) drives its
loop through the *typed streaming* contract that ``lamquant.oracle.
streaming_dataset.PrecomputedL3Dataset`` implements:

    train_ds.calibrate_shard_budget(device)
    for batch in train_ds.prefetch_typed_batches(batch_size, device, sampler):
        batch.assert_no_leakage(Split.TRAIN)   # iterates batch.splits
        x_l3 = batch.l3_approx                  # [B, 21, 313] on device
        ... batch.fullband_target ...           # [B, 21, 2500] or None
        ... batch.has_seizure ...               # List[bool], per-sample
    # validation loop additionally reads batch.clinical_categories

When ``train_joint`` is launched with ``--lma-root`` + ``--split-manifest``
it instead constructs the canonical ``lamquant_codec.training.LmaL3Dataset``,
which is a plain map-style ``Dataset`` (``__getitem__`` → ``(l3, l3, mask)``).
It has **none** of those streaming methods, so the loop hits
``AttributeError: 'LmaL3Dataset' object has no attribute
'calibrate_shard_budget'`` before epoch 1. It also yields a *dummy zero
mask*, so even if the surface matched there would be no real seizure
labels for the multi-task seizure head.

This adapter is the neural-side fix. It does NOT touch the canonical
codec (``lamquant_codec`` is the Lossless single-source-of-truth, governed
by standard SDLC). It wraps the *seizure-aware* ``lamquant.snn.lma_dataset.
LmaDataset`` — which already carries real per-window activity labels +
subject IDs from the split manifest — and re-implements the typed
streaming surface ``train_joint`` consumes, assembling ``TrainingBatch``
instances with:

  - ``l3_approx``        real L3 windows, [B, 21, 313], float32 on device
  - ``fullband_target``  raw [B, 21, 2500] window when return_fullband=True,
                         else None (decoded via the canonical
                         ``decode_lma_signal`` helper)
  - ``splits``           every entry == the fixed split this dataset holds
                         (so ``assert_no_leakage`` is a true safety net)
  - ``has_seizure``      real bool per window (activity class 2 present in
                         the window's label slice)
  - ``patient_ids``      subject id from the split manifest (provenance)
  - ``datasets``         corpus tag parsed from the stem (provenance)
  - ``clinical_categories`` 'seizure' when has_seizure else 'normal'
                         (coarse — see NOTE below)

See ADR 0017 (BLUT canonical trainer + LMA-direct).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as _MapDatasetBase

from lamquant.snn.lma_annotations import LMA_ANNOTATION_SENTINEL

# MOVE-B (2026-05-29): now at blut/python/lamquant/student/. Put the
# lamquant package root (blut/python) + the common DTO dir on sys.path
# so `from data_types import ...` and `from lamquant.snn.lma_dataset
# import ...` resolve regardless of how the trainer was launched.
_LAMQUANT = Path(__file__).resolve().parents[1]            # lamquant/
_REPO = _LAMQUANT.parent                                   # blut/python (pkg root)
for _p in (str(_REPO), str(_LAMQUANT), str(_LAMQUANT / 'common')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shape constants — keep in lockstep with snn.lma_dataset / canonical codec.
TARGET_CHANNELS = 21
L3_T = 313
WINDOW_SAMPLES = 2500          # 10 s @ 250 Hz
LABEL_PER_WINDOW = 312         # 2500 // 8 (stride-8 label grid)
SEIZURE_CLASS = 2              # activity_labels value meaning "seizure"

# Distinct sentinel for the fullband-signal LRU: decode_lma_signal legitimately
# returns None (decode failure) and we cache that too, so we cannot use None or
# .get(key) to mean "absent" — a cached-None must hit, not re-decode.
_CACHE_MISS = object()


class _TypedWindowMapDataset(_MapDatasetBase):
    """Map-style view over one epoch's window indices, for DataLoader workers.

    Used only when ``num_workers > 0``. Each fork-worker process inherits the
    adapter's wrapped ``LmaDataset`` + its index/caches via copy-on-write and
    decodes independently; the on-disk L3 cache (``L3_CACHE_DIR``) is shared
    through the filesystem. ``__getitem__`` returns the exact per-window row the
    synchronous path builds, so the assembled ``TrainingBatch`` is identical
    regardless of worker count — only the decode is parallelised.
    """

    def __init__(self, adapter: "LmaTypedL3Dataset", epoch_idx: Sequence[int]):
        self._a = adapter
        self._idx = epoch_idx

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: int):
        return self._a._window_row(int(self._idx[i]))


def _identity_collate(rows):
    """Keep the per-window rows as a plain list. Workers already produced the
    (l3, fb, has_seizure, pid, dataset) tuples; the main process does the single
    stack + pinned H2D copy per batch (in ``prefetch_typed_batches._emit``).
    torch's default ``pin_memory`` still recurses this list/tuple structure and
    pins the contained tensors."""
    return rows


def _import_training_batch():
    """Import TrainingBatch + Split with the same dual-path fallback the
    legacy streaming dataset uses (package import, then bare-script)."""
    try:
        from lamquant.common.data_types import Split, TrainingBatch
    except ImportError:
        from data_types import Split, TrainingBatch  # type: ignore
    return TrainingBatch, Split


def _dataset_tag_from_stem(stem: str) -> str:
    """Best-effort corpus tag for provenance.

    Stems look like ``aaaaaaac_s001_t000`` (TUH-family) — there is no
    embedded corpus name, so we return a stable coarse tag. This is
    provenance metadata only; it does NOT gate the loss or sampling.
    Refine here if a stem→corpus map becomes available.
    """
    return "lma"


class LmaTypedL3Dataset:
    """Typed-batch streaming adapter over the seizure-aware ``LmaDataset``.

    Drop-in for the ``PrecomputedL3Dataset`` *typed* surface that
    ``train_joint.py`` consumes. Construct one per split.

    Parameters
    ----------
    lma_root / lma_dir : directory of ``<stem>.lma`` (or ``<src>/<corpus>.lma``)
        archives. Passed straight to the wrapped ``LmaDataset``.
    split : 'train' | 'val'
    split_manifest_path : split manifest JSON (subject→split + stems).
    windows_per_epoch : how many windows one epoch yields. The wrapped
        ``LmaDataset`` enumerates a deterministic, seizure-aware window
        index; we sample (with replacement when the index is smaller)
        ``windows_per_epoch`` indices from it per epoch.
    return_fullband : when True, decode + attach the raw [21, 2500] window
        as ``fullband_target`` (needed by Tier 3+ iSTFT decoders). Off by
        default; the L3-only path is the firmware-TNN default.
    seed : RNG seed for per-epoch index sampling.
    **lma_kwargs : forwarded to ``LmaDataset`` (e.g. max_windows_per_file).
    """

    def __init__(
        self,
        *,
        lma_root: Optional[Path] = None,
        lma_dir: Optional[Path] = None,
        split: str = "train",
        split_manifest_path: Optional[Path] = None,
        windows_per_epoch: int = 50_000,
        return_fullband: bool = False,
        seed: int = 0,
        num_workers: int = 0,
        **lma_kwargs,
    ):
        from snn.lma_dataset import LmaDataset

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if split_manifest_path is None:
            raise ValueError("split_manifest_path is required")

        self.split = split
        self.windows_per_epoch = int(windows_per_epoch)
        self._return_fullband = bool(return_fullband)
        self._rng = np.random.default_rng(seed)
        # Parallel decode workers. The per-window decode (canonical Rust
        # decode_lma_signal + numpy/scipy L3 DWT) is CPU-bound and was the
        # GPU-starvation bottleneck (util ~30%). >0 routes prefetch through a
        # torch DataLoader with that many fork-worker processes, each with its
        # own in-process caches + the shared on-disk L3 cache, overlapping
        # decode with GPU compute. Env LMA_NUM_WORKERS overrides; default 0
        # keeps the legacy synchronous path (no behaviour change unless asked).
        _envw = os.environ.get("LMA_NUM_WORKERS")
        self._num_workers = int(_envw) if _envw not in (None, "") else int(num_workers)

        # The seizure-aware dataset owns: decode, L3, label NPZ, subject map,
        # and seizure-aware per-file window selection. We delegate all of it.
        self._base = LmaDataset(
            lma_dir=Path(lma_dir) if lma_dir is not None else None,
            lma_root=Path(lma_root) if lma_root is not None else None,
            split=split,
            split_manifest_path=Path(split_manifest_path),
            **lma_kwargs,
        )

        # Resolve subject id per stem once (provenance for patient_ids).
        from snn.lma_dataset import load_split_manifest
        _, self._subject_by_stem = load_split_manifest(
            Path(split_manifest_path), split
        )

        # Pre-compute per-base-index seizure flag from the wrapped dataset's
        # window index. self._base.index entries are
        #   (lma_path, stem, win_idx, lml_internal, label_internal)
        # We read each window's label slice once at construction so the hot
        # loop never re-touches the label NPZ just to set has_seizure.
        self._n_base = len(self._base)
        self._win_has_seizure: List[bool] = self._precompute_seizure_flags()

        # Disable clinical_sampling in train_joint: the wrapped dataset
        # already does seizure-aware window selection, so a second weighted
        # sampler would double-count. train_joint guards on
        # `train_ds._win_clinical_category is not None`.
        self._win_clinical_category = None

        # Streaming guards / state mirrored from PrecomputedL3Dataset.
        self._win_dataset = ["present"]   # non-None sentinel: typed path is enabled
        self._shard_max = None

    # ---- construction helpers ------------------------------------------

    def _precompute_seizure_flags(self) -> List[bool]:
        """Per-base-window seizure flag, read from the label cache/NPZ once.

        Uses the same selection-aligned slice the wrapped dataset uses:
        window ``wi`` covers label columns ``[wi*312, wi*312+313)``. A
        window is "seizure" if any label in that slice equals class 2.

        On any read failure we default the window to False (no seizure) and
        log — a missing label must never silently *invent* a positive.
        """
        import io
        import logging

        log = logging.getLogger(__name__)
        flags: List[bool] = [False] * self._n_base

        # Group base indices by (lma_path, label_internal) so each label
        # NPZ is decoded at most once.
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for i in range(self._n_base):
            lma_path, stem, win_idx, _lml, label_internal = self._base.index[i]
            groups[(str(lma_path), stem, label_internal)].append((i, win_idx))

        # Reuse the wrapped module's disk-label-cache resolver.
        from snn.lma_dataset import _label_cache_dir
        label_cache = _label_cache_dir()
        import lamquant_core as _lc

        for (lma_str, stem, label_internal), items in groups.items():
            activity = None
            cached = (label_cache / f"{stem}_labels.npz") if label_cache else None
            try:
                if label_internal == LMA_ANNOTATION_SENTINEL:
                    # On-the-fly sentinel: not a real archive entry. Derive the
                    # seizure flag from the LMA's bundled annotation (None ->
                    # background-only recording -> all windows stay False).
                    from lamquant.snn.lma_annotations import lma_activity_labels
                    activity = lma_activity_labels(lma_str, stem)
                    if activity is None:
                        continue  # no annotation -> no seizures -> leave False
                    activity = np.asarray(activity)
                elif cached is not None and cached.exists():
                    with np.load(cached, allow_pickle=True) as ld:
                        activity = np.asarray(ld["activity_labels"])
                else:
                    label_bytes = _lc.lma_read_entry(lma_str, label_internal)
                    with np.load(io.BytesIO(label_bytes), allow_pickle=True) as ld:
                        activity = np.asarray(ld["activity_labels"])
            except Exception as e:  # noqa: BLE001 — provenance, must not crash init
                log.warning("seizure-flag labels unreadable for %s: %s", stem, e)
                continue
            n_cols = activity.shape[1]
            for base_i, win_idx in items:
                lo = win_idx * LABEL_PER_WINDOW
                hi = min(lo + L3_T, n_cols)
                if hi > lo and np.any(activity[:, lo:hi] == SEIZURE_CLASS):
                    flags[base_i] = True
        return flags

    # ---- typed streaming surface that train_joint consumes -------------

    def __len__(self) -> int:
        return self.windows_per_epoch

    def calibrate_shard_budget(self, device) -> None:
        """API-compatible no-op-ish calibration.

        ``PrecomputedL3Dataset`` holds the whole epoch in one RAM tensor and
        shards the GPU transfer. The LMA-direct path decodes lazily per
        window via the wrapped dataset, so there is no monolithic transfer
        to size — we just record the device. Kept as a method so the
        trainer's ``train_ds.calibrate_shard_budget(device)`` call resolves.
        """
        self._device = torch.device(device) if isinstance(device, str) else device
        # No VRAM-sized shard: per-window lazy decode means peak GPU memory
        # is one batch, already bounded by batch_size.
        self._shard_max = self.windows_per_epoch

        # Per-(lma,stem) decoded fullband-signal LRU. decode_lma_signal decodes
        # the ENTIRE recording (a TUEG file can be hours -> seconds per call);
        # without this, every fetched window re-decoded the whole recording
        # (7+ h/epoch on TUEG). With stem-grouped sampling (below) a stem's
        # windows arrive consecutively, so a tiny LRU collapses N per-window
        # decodes into 1 per stem. Signals are large (~hundreds of MB for long
        # recordings) so the cap is deliberately small.
        from collections import OrderedDict as _OrderedDict
        self._fb_sig_cache: "_OrderedDict" = _OrderedDict()
        self._fb_sig_cache_cap = 3
        # Cross-epoch DISK cache for the decoded fullband signal (the loss
        # TARGET). decode_lma_signal re-decodes the lossless recording every
        # epoch — the residual dataload bottleneck after the L3 input cache
        # (E1 2026-06-04: GPU ~10% even with L3 cached). Persisting it makes
        # epochs 2+ a mmap slice instead of a full decode. Enabled by
        # FB_CACHE_DIR; fp16 (FB_CACHE_DTYPE) halves disk; FB_CACHE_MAX_GB
        # bounds growth (writes stop past budget -> long-tail stems still
        # decode; in-mem LRU still serves) to avoid filling a near-full disk.
        _fbd = os.environ.get("FB_CACHE_DIR", "").strip()
        self._fb_disk_dir = _fbd or None
        self._fb_disk_dtype = np.float16 if os.environ.get(
            "FB_CACHE_DTYPE", "float16").strip().lower() in ("float16", "fp16", "half") else np.float32
        self._fb_disk_budget = int(float(os.environ.get("FB_CACHE_MAX_GB", "60")) * 1e9)
        self._fb_disk_bytes = 0
        self._stem_groups = None   # lazily built grouped index (sampler)

    def _fb_disk_load(self, stem: str):
        """mmap the disk-cached decoded fullband signal for `stem`, or None on
        miss / no cache dir / load error (caller decodes + saves)."""
        if self._fb_disk_dir is None:
            return None
        p = os.path.join(self._fb_disk_dir, f"{stem}__fb.npy")
        if os.path.exists(p):
            try:
                return np.load(p, mmap_mode="r")
            except Exception:
                return None
        return None

    def _fb_disk_save(self, stem: str, signal) -> None:
        """Persist a decoded fullband signal (best-effort, atomic, budget-capped).
        Never raises — the disk cache is an optimization, not a correctness path."""
        if self._fb_disk_dir is None or signal is None:
            return
        if self._fb_disk_bytes >= self._fb_disk_budget:
            return  # budget exhausted -> stop writing (near-full disk guard)
        try:
            os.makedirs(self._fb_disk_dir, exist_ok=True)
            p = os.path.join(self._fb_disk_dir, f"{stem}__fb.npy")
            if os.path.exists(p):
                return
            arr = np.asarray(signal, dtype=self._fb_disk_dtype)
            tmp = p + ".tmp.npy"           # np.save appends .npy; sandwich .tmp
            np.save(tmp, arr)
            os.replace(tmp, p)
            self._fb_disk_bytes += arr.nbytes
        except Exception:
            pass

    def _sample_epoch_indices(
        self, n_total: int, sampler: Optional[Sequence[int]]
    ) -> List[int]:
        """Pick n_total base-window indices for this epoch.

        ``sampler`` is accepted for surface-compatibility with
        ``prefetch_typed_batches(sampler=...)`` but is normally None on the
        LMA path (we set ``_win_clinical_category=None`` so train_joint does
        not build a ClinicalWeightedSampler). When provided, indices are
        drawn from it and clamped into range.
        """
        if sampler is not None:
            it = iter(sampler)
            idx = [int(next(it)) for _ in range(n_total)]
            return [min(max(i, 0), self._n_base - 1) for i in idx]
        # Stem-grouped epoch: shuffle stems, emit each stem's contiguous base
        # indices together. The base index is built stem-by-stem so a stem's
        # windows are already contiguous; grouping preserves that locality so
        # the per-stem fullband-signal LRU in _fetch_window hits — 1 full
        # recording decode per stem instead of one per window (the dominant
        # cost on long TUEG recordings). Random PER-WINDOW draws over a huge
        # index gave every fetch a fresh stem -> 0% cache hit -> 7 h/epoch.
        if self._stem_groups is None:
            from collections import OrderedDict as _OD
            groups: "_OD" = _OD()
            for i in range(self._n_base):
                e = self._base.index[i]
                groups.setdefault((str(e[0]), e[1]), []).append(i)
            self._stem_groups = list(groups.values())
        n_groups = len(self._stem_groups)
        order = self._rng.permutation(n_groups) if n_groups else []
        out: List[int] = []
        gi = 0
        while len(out) < n_total and n_groups:
            out.extend(self._stem_groups[int(order[gi % n_groups])])
            gi += 1
        return out[:n_total]

    def _fetch_window(self, base_idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (l3 [21,313] float32 cpu, fullband [21,2500] or None).

        L3 comes straight from the wrapped seizure-aware dataset (which uses
        the canonical decode + preprocess). Fullband, when requested, is the
        raw decoded window from the SAME archive via the canonical
        ``decode_lma_signal`` (no second preprocessing pipeline).
        """
        l3, _labels = self._base[base_idx]           # (torch[21,313], torch[8,L3_T])
        if not isinstance(l3, torch.Tensor):
            l3 = torch.from_numpy(np.asarray(l3, dtype=np.float32))
        l3 = l3.float()

        if not self._return_fullband:
            return l3, None

        lma_path, stem, win_idx, _lml, _lbl = self._base.index[base_idx]
        # Per-(lma,stem) LRU: decode_lma_signal decodes the WHOLE recording, so
        # without caching every window re-decoded its (often multi-hour) parent
        # -> 7+ h/epoch on TUEG. Stem-grouped sampling delivers a stem's windows
        # consecutively, so this tiny cache collapses them into one decode.
        cache_key = (str(lma_path), stem, _lml)
        signal = self._fb_sig_cache.get(cache_key, _CACHE_MISS)
        if signal is _CACHE_MISS:
            signal = self._fb_disk_load(stem)   # cross-epoch disk tier (mmap slice)
            if signal is None:
                from lamquant_codec.training import decode_lma_signal
                # Propagate the resolved internal entry (e.g. 'S001/S001R01.edf'
                # for `lml archive` corpora). Without it decode_lma_signal
                # defaults to the legacy '<stem>.lml' name, absent in per-corpus
                # archives -> signal None -> fullband_target None.
                signal = decode_lma_signal(str(lma_path), stem, lml_entry_name=_lml)
                self._fb_disk_save(stem, signal)   # persist decode for next epoch
            self._fb_sig_cache[cache_key] = signal
            if len(self._fb_sig_cache) > self._fb_sig_cache_cap:
                self._fb_sig_cache.popitem(last=False)
        else:
            self._fb_sig_cache.move_to_end(cache_key)
        if signal is None:
            fb = torch.zeros(TARGET_CHANNELS, WINDOW_SAMPLES, dtype=torch.float32)
            return l3, fb
        start = win_idx * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        if end > signal.shape[1]:
            window = np.zeros((TARGET_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
            avail = max(0, signal.shape[1] - start)
            if avail > 0:
                window[:, :avail] = signal[:, start:start + avail]
        else:
            window = np.asarray(signal[:, start:end], dtype=np.float32)
        fb = torch.from_numpy(np.ascontiguousarray(window))
        return l3, fb

    def _window_row(self, bi: int):
        """One window's row: (l3[21,313], fb[21,2500] or None, has_seizure,
        patient_id, dataset_tag). Shared by the synchronous and worker-pool
        prefetch paths so the assembled batch is identical either way."""
        l3, fb = self._fetch_window(bi)
        _lma_path, stem, _wi, _lml, _lbl = self._base.index[bi]
        return (
            l3,
            fb if self._return_fullband else None,
            bool(self._win_has_seizure[bi]),
            self._subject_by_stem.get(stem, stem),
            _dataset_tag_from_stem(stem),
        )

    def prefetch_typed_batches(self, batch_size, device, sampler=None):
        """Yield ``TrainingBatch`` instances, l3_approx already on device.

        Mirrors ``PrecomputedL3Dataset.prefetch_typed_batches`` semantics:
          - drops the trailing partial batch (n_total floors to batch_size),
          - l3_approx is [B,21,313] float32 on ``device``,
          - fullband_target is [B,21,2500] float32 on ``device`` or None,
          - splits is all-``self.split`` so assert_no_leakage is meaningful,
          - has_seizure is the real per-window flag,
          - provenance arrays (datasets, patient_ids) are populated.

        When ``self._num_workers > 0`` the per-window decode is parallelised
        across that many DataLoader fork-workers (decode is CPU-bound and was
        starving the GPU); the assembled ``TrainingBatch`` is byte-identical to
        the synchronous path — only the decode is overlapped with GPU compute.
        """
        TrainingBatch, _Split = _import_training_batch()
        dev = torch.device(device) if isinstance(device, str) else device

        n_total = (self.windows_per_epoch // batch_size) * batch_size
        epoch_idx = self._sample_epoch_indices(n_total, sampler)

        def _emit(rows):
            l3_batch = torch.stack([r[0] for r in rows], dim=0).to(dev, non_blocking=True)
            fb_batch = (
                torch.stack([r[1] for r in rows], dim=0).to(dev, non_blocking=True)
                if self._return_fullband else None
            )
            has_sz = [bool(r[2]) for r in rows]
            n = len(rows)
            return TrainingBatch(
                l3_approx=l3_batch,
                fullband_target=fb_batch,
                datasets=[r[4] for r in rows],
                patient_ids=[r[3] for r in rows],
                splits=[self.split] * n,
                has_seizure=has_sz,
                event_types=[""] * n,
                clinical_categories=["seizure" if s else "normal" for s in has_sz],
            )

        if self._num_workers and self._num_workers > 0:
            # Parallel decode path. shuffle=False: epoch_idx is already
            # stem-grouped by _sample_epoch_indices, so contiguous worker chunks
            # keep the per-stem decode/L3 cache hits. drop_last matches the
            # n_total flooring above.
            from torch.utils.data import DataLoader
            loader = DataLoader(
                _TypedWindowMapDataset(self, epoch_idx),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self._num_workers,
                collate_fn=_identity_collate,
                pin_memory=(dev.type == "cuda"),
                prefetch_factor=4,
                persistent_workers=False,
            )
            for rows in loader:
                yield _emit(rows)
            return

        for start in range(0, n_total, batch_size):
            rows = [self._window_row(bi) for bi in epoch_idx[start:start + batch_size]]
            yield _emit(rows)
