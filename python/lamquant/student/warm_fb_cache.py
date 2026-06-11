#!/usr/bin/env python3
"""Warm the per-window fullband disk cache the joint-codec trainer reads.

Never-OOM Phase 2 (cached upstream DAG stage). The LMA-direct trainer
(``train_joint.py`` → ``LmaTypedL3Dataset``) reads each fullband target
window from a per-window disk cache FIRST (``FB_CACHE_DIR/*__fbw.npy``,
mmap-loaded) and only on a MISS decodes the WHOLE multi-hour recording
in-process into an LRU (cap 1) — the decode that starves the GPU and, when
the LRU fills, drove the epoch-2 OOM. After every used window is on disk the
in-RAM signal LRU stays empty, so the per-worker resident set collapses to
CoW-fork + an mmap page (reclaimable), not a whole recording.

This script pre-populates that EXACT cache by driving the SAME adapter the
trainer uses — it does NOT reimplement the cache-key derivation (``_fb_win_path``
hashes ``sha1(f"{lma_path}\\x00{lml}")[:10]_{stem}_w{win_idx}``). Reusing the
adapter object is the load-bearing correctness property: any divergence in the
key (a different ``lma_root`` spelling, a different ``lml_entry_name``) would
make the warm cache silently never hit and the trainer re-decode from scratch.

It calls ``cache_paths.apply_env()`` FIRST — identically to ``train_joint.py``
— so ``FB_CACHE_DIR`` resolves to the same ``<LAMQUANT_DATA_ROOT>/Training/
fb_cache`` the trainer will read. Both halves (train + val) are warmed with the
trainer's seeds, though the cache key is window-identity (seed-independent), so
a complete warm covers every base window regardless of epoch sampling.

Idempotent: ``_fb_win_save`` skips a window already on disk, so a re-run only
fills gaps. Deterministic: the codec decode is lossless + the fp16 cast is the
trainer's, so a warm-written window is byte-identical to what a train worker
would have written.

Usage:
    python warm_fb_cache.py --lma-root <dir> --split-manifest <json> \\
        [--splits train val] [--seed 42] [--max-windows N] [--min-free-gb 40]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time


def _eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def _free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return float("inf")


def warm_split(
    lma_root: str,
    split: str,
    split_manifest_path: str,
    seed: int,
    max_windows: int | None,
) -> tuple[int, int]:
    """Warm every (or the first ``max_windows``) base window of one split.

    Returns (windows_processed, windows_in_split). Drives the trainer's
    adapter so the disk-cache keys match exactly.
    """
    # Bare import (matches train_joint.py): this script lives in student/ next
    # to lma_typed_adapter.py, so student/ is sys.path[0] when it runs.
    from lma_typed_adapter import LmaTypedL3Dataset

    ds = LmaTypedL3Dataset(
        lma_root=lma_root,
        split=split,
        split_manifest_path=split_manifest_path,
        # Irrelevant for warming (the cache key is per-window-identity, not
        # per-epoch-sample); pass a placeholder so the constructor is happy.
        windows_per_epoch=1,
        return_fullband=True,
        seed=seed,
    )
    n_base = int(ds._n_base)
    total = n_base if max_windows is None else min(n_base, int(max_windows))
    _eprint(
        f"[warm_fb_cache] split={split} base_windows={n_base} "
        f"warming={total} seed={seed}"
    )
    if total == 0:
        return 0, n_base

    # The base index is built stem-by-stem (contiguous per stem), so iterating
    # 0..n_base hits the adapter's in-proc per-stem LRU — one whole-recording
    # decode per stem, not one per window (the dominant cost on long TUEG
    # recordings). _fetch_window persists each [21,2500] window via _fb_win_save.
    log_every = max(1, total // 100)  # ~1% granularity
    t0 = time.time()
    for i in range(total):
        try:
            ds._fetch_window(i)
        except Exception as e:  # noqa: BLE001 — never abort the whole warm on one bad window
            _eprint(f"[warm_fb_cache] window {i} failed (skipping): {e}")
        if (i + 1) % log_every == 0 or (i + 1) == total:
            # tqdm-shaped "<done>/<total> [" so the BLUT runner's progress
            # parser (parse_tqdm_progress) forwards it as a StageStep.
            rate = (i + 1) / max(1e-6, time.time() - t0)
            _eprint(f"{i + 1}/{total} [warm_fb_cache split={split} {rate:.0f} win/s]")
    return total, n_base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Warm the per-window fullband disk cache.")
    ap.add_argument("--lma-root", required=True, help="Root of the LMA corpus.")
    ap.add_argument(
        "--split-manifest", required=True, help="Subject-grouped split manifest JSON."
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to warm (default: train val — both the trainer reads).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; val uses seed+1 (mirrors train_joint).",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap windows warmed PER SPLIT (default: all base windows).",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Refuse to start if the cache disk has less than this free "
        "(default: FB_CACHE_MIN_FREE_GB env or 40). Prevents a silent partial "
        "warm that would leave the trainer re-decoding (and re-OOM-prone).",
    )
    args = ap.parse_args(argv)

    if not os.path.isdir(args.lma_root):
        _eprint(f"[warm_fb_cache] FATAL: lma_root not a dir: {args.lma_root}")
        return 2
    if not os.path.isfile(args.split_manifest):
        _eprint(f"[warm_fb_cache] FATAL: split_manifest not a file: {args.split_manifest}")
        return 2

    # Resolve the canonical cache dirs from the ONE root (LAMQUANT_DATA_ROOT or
    # the canonical default) — IDENTICALLY to train_joint.py:732, so the warm
    # FB_CACHE_DIR is the exact dir the trainer reads. apply_env force-overwrites
    # FB_CACHE_DIR/L3_CACHE_DIR/MEMMAP_DIR so a stale value can't survive.
    try:
        from lamquant.common.cache_paths import apply_env
        layout = apply_env()
    except Exception as e:  # noqa: BLE001
        _eprint(f"[warm_fb_cache] FATAL: cache_paths.apply_env() failed: {e}")
        return 2
    fb_dir = layout.fb_cache_dir
    os.makedirs(fb_dir, exist_ok=True)
    _eprint(f"[warm_fb_cache] data_root={layout.data_root} FB_CACHE_DIR={fb_dir}")

    # Disk preflight: a near-full disk makes _fb_win_save's statvfs guard stop
    # writing mid-run, producing a SILENT partial cache. Fail loudly up front.
    min_free = (
        args.min_free_gb
        if args.min_free_gb is not None
        else float(os.environ.get("FB_CACHE_MIN_FREE_GB", "40"))
    )
    free = _free_gb(fb_dir)
    if free < min_free:
        _eprint(
            f"[warm_fb_cache] FATAL: only {free:.1f} GB free at {fb_dir}, "
            f"need >= {min_free:.1f} GB to warm without a silent partial cache. "
            f"Free disk or lower --min-free-gb if you accept a partial warm."
        )
        return 3

    splits = list(dict.fromkeys(args.splits))  # de-dup, preserve order
    for s in splits:
        if s not in ("train", "val"):
            _eprint(f"[warm_fb_cache] FATAL: split must be train|val, got {s!r}")
            return 2

    grand_total = 0
    t0 = time.time()
    for s in splits:
        seed = args.seed if s == "train" else args.seed + 1
        try:
            done, n_base = warm_split(
                args.lma_root, s, args.split_manifest, seed, args.max_windows
            )
        except Exception as e:  # noqa: BLE001 — a split that can't even construct is fatal
            _eprint(f"[warm_fb_cache] FATAL: split {s} failed to warm: {e}")
            return 4
        grand_total += done
        # Re-check free disk after each split; if the guard tripped, the cache
        # is partial — surface it as a failure, not a silent success.
        free_after = _free_gb(fb_dir)
        if free_after < min_free * 0.5:
            _eprint(
                f"[warm_fb_cache] FATAL: free disk fell to {free_after:.1f} GB during "
                f"split {s} — the cache is likely PARTIAL (the trainer would "
                f"re-decode + risk OOM). Free disk and re-run."
            )
            return 3

    elapsed = time.time() - t0
    n_files = 0
    try:
        n_files = sum(1 for f in os.scandir(fb_dir) if f.name.endswith("__fbw.npy"))
    except OSError:
        pass
    _eprint(
        f"[warm_fb_cache] DONE: warmed {grand_total} windows across {splits} in "
        f"{elapsed:.0f}s; {n_files} __fbw.npy files at {fb_dir} "
        f"({_free_gb(fb_dir):.0f} GB free)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
