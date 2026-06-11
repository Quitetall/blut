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

WINDOW-SELECTION CONTRACT (cache-hit correctness): the per-window cache key
embeds ``win_idx``, which the dataset's ``select_windows`` derives from
``max_windows_per_file``. This script does NOT pass that knob, so it warms the
SAME windows the trainer reads BY DEFAULT (both fall through to the dataset's
``MAX_WINDOWS_PER_FILE``). If you OVERRIDE ``--max-windows-per-file`` on
``train_joint`` you MUST warm with the same value (or the warmed window set will
not match what the trainer fetches → a partial cache → re-decode + re-OOM under
the warm-tightened broker footprint).

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
        # FAIL-CLOSED: an unknowable free-space state must trip the safety guard
        # (refuse to warm), never bypass it. Returning inf would let a genuinely
        # full / inaccessible disk silently produce a partial cache.
        return 0.0


# Per-split warm dataset, built ONCE in the parent and INHERITED by the
# fork-pool workers (copy-on-write). So the (potentially large) per-recording
# index is constructed a single time, not once per worker, while each worker's
# in-process signal LRU stays PRIVATE — decode-once-per-stem still holds within
# a worker's contiguous slice. Module-global because a fork worker reads the
# parent's globals; an explicit arg can't cross the fork boundary cheaply.
_WARM_DS = None


def _build_warm_ds(lma_root: str, split: str, split_manifest_path: str, seed: int):
    """Construct + calibrate the trainer's adapter for one split (the warm
    dataset). Reusing the adapter is what guarantees byte-identical cache keys."""
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
    # CRITICAL: the disk-cache attributes (_fb_disk_dir / _fb_sig_cache /
    # _fb_disk_dtype / _fb_min_free) are initialized in calibrate_shard_budget,
    # NOT __init__ — the trainer calls it post-construction (train_joint.py).
    # Without it, _fetch_window → _fb_win_path raises AttributeError on
    # self._fb_disk_dir and the warm silently writes NOTHING. "cpu" because the
    # decode is CPU-bound; the device is only recorded, not used for the decode.
    ds.calibrate_shard_budget("cpu")
    return ds


def _warm_range(rng: tuple[int, int]) -> tuple[int, int]:
    """Warm a contiguous ``[start, end)`` base-index slice using the
    fork-inherited ``_WARM_DS``. Returns (processed, failed). Runs in a worker
    process (parallel path) — kept top-level + dependency-free so the fork pool
    can dispatch it by name."""
    if _WARM_DS is None:
        # The fork-inherited global is unset — would only happen if the pool
        # somehow used `spawn` (the child re-imports the module fresh). Fail
        # LOUDLY rather than silently warming nothing.
        raise RuntimeError("_warm_range: _WARM_DS not inherited (non-fork start method?)")
    start, end = rng
    failed = 0
    for i in range(start, end):
        # `except Exception` does NOT catch KeyboardInterrupt (BaseException) —
        # Ctrl-C kills the worker + the pool raises in the parent.
        try:
            _WARM_DS._fetch_window(i)
        except Exception as e:  # noqa: BLE001 — one bad window must not abort the slice
            failed += 1
            # Surface the first few per worker (a systemic setup error repeats);
            # mirrors the serial path so parallel mode is debuggable too.
            if failed <= 3:
                _eprint(f"[warm_fb_cache] worker window {i} failed: {e!r}")
    return (end - start), failed


def _warm_serial(split: str, total: int, n_base: int) -> tuple[int, int, int]:
    """Single-process warm of ``range(total)`` (the debug / tiny-split path)."""
    log_every = max(1, total // 100)  # ~1% granularity
    failed = 0
    t0 = time.time()
    for i in range(total):
        try:
            _WARM_DS._fetch_window(i)
        except Exception as e:  # noqa: BLE001 — one bad window must not abort the warm
            failed += 1
            # Log the FIRST few in full (a systemic setup error — e.g. a missing
            # adapter attribute — would repeat; surface it, don't bury it).
            if failed <= 5:
                _eprint(f"[warm_fb_cache] window {i} failed: {e!r}")
        if (i + 1) % log_every == 0 or (i + 1) == total:
            # tqdm-shaped "<done>/<total> [" so the BLUT runner's progress
            # parser (parse_tqdm_progress) forwards it as a StageStep.
            rate = (i + 1) / max(1e-6, time.time() - t0)
            _eprint(f"{i + 1}/{total} [warm_fb_cache split={split} {rate:.0f} win/s]")
    if failed:
        _eprint(f"[warm_fb_cache] split={split}: {failed}/{total} windows FAILED")
    return total, n_base, failed


def warm_split(
    lma_root: str,
    split: str,
    split_manifest_path: str,
    seed: int,
    max_windows: int | None,
    workers: int,
) -> tuple[int, int, int]:
    """Warm every (or the first ``max_windows``) base window of one split.

    Returns (windows_processed, windows_in_split, windows_failed). Drives the
    trainer's adapter so the disk-cache keys match exactly. ``workers`` > 1
    decodes contiguous slices in parallel fork workers (the warm is CPU-bound
    serial decode — single-process was ~20 h for the full corpus).
    """
    global _WARM_DS
    _WARM_DS = _build_warm_ds(lma_root, split, split_manifest_path, seed)
    n_base = int(_WARM_DS._n_base)
    total = n_base if max_windows is None else min(n_base, int(max_windows))
    _eprint(
        f"[warm_fb_cache] split={split} base_windows={n_base} "
        f"warming={total} seed={seed} workers={workers}"
    )
    if total == 0:
        return 0, n_base, 0

    nproc = max(1, int(workers))
    import multiprocessing as mp

    # The CoW-inherited `_WARM_DS` only works under FORK (a `spawn` child
    # re-imports the module with `_WARM_DS=None`), so the parallel path requires
    # fork to be available. fork is available on Linux + macOS (deprecated but
    # works for this pure-rust/numpy decode — we request it explicitly via
    # get_context below regardless of the platform default); only a genuinely
    # fork-less platform (e.g. Windows) lacks it.
    fork_ok = "fork" in mp.get_all_start_methods()
    # Serial for: a small split (fork + per-worker index-share overhead isn't
    # worth it under ~512 windows), an explicit single worker, or no fork.
    if nproc <= 1 or total < 512 or not fork_ok:
        if not fork_ok and nproc > 1 and total >= 512:
            _eprint("[warm_fb_cache] fork start method unavailable — warming serially")
        return _warm_serial(split, total, n_base)

    # Parallel: contiguous chunks. The base index is stem-contiguous, so a
    # contiguous slice ≈ whole stems → the per-stem in-proc LRU still amortises
    # the whole-recording decode within a worker; only the few stems straddling
    # a chunk boundary decode in two workers (minor). FORK IS SAFE here: the
    # decode is CPU/rust and no CUDA context is initialised (calibrate uses
    # device="cpu"), so the inherited interpreter state is benign. Workers
    # inherit `_WARM_DS` (built above) via copy-on-write — one index build total.
    # Integer-division bounds → gap-free, deterministic partition (no rounding).
    bounds = [total * k // nproc for k in range(nproc + 1)]
    chunks = [(bounds[k], bounds[k + 1]) for k in range(nproc) if bounds[k + 1] > bounds[k]]
    _eprint(
        f"[warm_fb_cache] split={split}: {total} windows across {len(chunks)} fork workers"
    )
    done = 0
    failed = 0
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(len(chunks)) as pool:
        for cdone, cfailed in pool.imap_unordered(_warm_range, chunks):
            done += cdone
            failed += cfailed
            rate = done / max(1e-6, time.time() - t0)
            _eprint(
                f"{done}/{total} [warm_fb_cache split={split} x{len(chunks)} {rate:.0f} win/s]"
            )
    if failed:
        _eprint(f"[warm_fb_cache] split={split}: {failed}/{total} windows FAILED")
    return total, n_base, failed


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
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel decode workers (the warm is CPU-bound serial decode; "
        "single-process was ~20 h for the full corpus). Default: WARM_FB_WORKERS "
        "env, else min(6, cpu//2). 1 = serial. Each worker holds ~one recording "
        "in RAM, so size against the stage's memory budget.",
    )
    args = ap.parse_args(argv)

    # Resolve worker count: explicit flag > WARM_FB_WORKERS env > min(6, cpu//2).
    # Each fork worker holds ~one recording in RAM, so the result is hard-capped
    # at the cpu count (a stale/typo'd env must not fork-bomb the box).
    cpu = os.cpu_count() or 4
    if args.workers is not None:
        workers = max(1, args.workers)
    else:
        env_w = os.environ.get("WARM_FB_WORKERS", "").strip()
        if env_w:
            try:
                workers = max(1, int(env_w))
            except ValueError:
                _eprint(f"[warm_fb_cache] WARM_FB_WORKERS={env_w!r} not an int — using default")
                workers = max(1, min(6, cpu // 2))
        else:
            workers = max(1, min(6, cpu // 2))
    workers = min(workers, cpu)  # fork-bomb guard (never more workers than cores)

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
    grand_failed = 0
    t0 = time.time()
    for s in splits:
        seed = args.seed if s == "train" else args.seed + 1
        try:
            done, n_base, failed = warm_split(
                args.lma_root, s, args.split_manifest, seed, args.max_windows, workers
            )
        except Exception as e:  # noqa: BLE001 — a split that can't even construct is fatal
            _eprint(f"[warm_fb_cache] FATAL: split {s} failed to warm: {e!r}")
            return 4
        grand_total += done
        grand_failed += failed
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

    # Catch a SILENT systemic failure: if we attempted windows but NOTHING
    # landed on disk, the warm achieved nothing (e.g. a setup error skipped
    # every window) — the trainer would re-decode + risk OOM under the
    # warm-tightened footprint. Fail loudly rather than report a false success.
    if grand_total > 0 and n_files == 0:
        _eprint(
            f"[warm_fb_cache] FATAL: attempted {grand_total} windows but 0 "
            f"__fbw.npy files exist at {fb_dir} — the warm wrote NOTHING "
            f"({grand_failed} window failures). The trainer would re-decode + "
            f"risk OOM. Aborting (do NOT report success)."
        )
        return 5
    # A high failure fraction means a partial cache — warn loudly (the trainer
    # self-heals via FB_SIG_CACHE_CAP=1 + the broker, but the operator should know).
    if grand_total > 0 and grand_failed * 100 > grand_total * 5:
        _eprint(
            f"[warm_fb_cache] WARNING: {grand_failed}/{grand_total} windows failed "
            f"(> 5%) — the cache is PARTIAL; the trainer will decode the gaps."
        )
    _eprint(
        f"[warm_fb_cache] DONE: warmed {grand_total - grand_failed}/{grand_total} "
        f"windows across {splits} in {elapsed:.0f}s; {n_files} __fbw.npy files at "
        f"{fb_dir} ({_free_gb(fb_dir):.0f} GB free)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
