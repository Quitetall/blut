#!/usr/bin/env python3
"""precompute_band_std.py — frozen GLOBAL per-band input std for the codec.

The 0b half of the allocation fix (the loss half landed as
``per_band_relative_loss`` / ADR 0049). The encoder now ingests the full
residual (``--detail-bands all`` -> ``in_ch=168``), but the detail bands
enter at their RAW 1/f amplitudes: l3_detail / l2_detail / l1_detail are
~10x smaller than the L3 approximation, so a unit-variance-ish encoder stem
sees them as near-noise and the optimizer never learns to allocate capacity
to the >15 Hz content. ``_stack_detail_bands`` (lma_dataset.py) feeds those
coefficients with NO normalization.

This script measures, ONCE over the training corpus, the **global** standard
deviation of each detail band (one scalar per band, pooled over every channel,
every sample, every window of every recording in the train split). The result
is written as a small JSON that ``_stack_detail_bands`` loads and uses to
divide each band group by its frozen std at stack time.

Why GLOBAL (corpus-frozen), NOT per-window
------------------------------------------
Per-window / per-recording normalization would divide out the absolute
amplitude that is clinically meaningful (a high-amplitude gamma burst and a
quiet background epoch would be rescaled to the same variance, destroying the
very signal the detail bands carry). A single frozen scalar per band is an
affine rescale of the whole corpus: it equalizes the *cross-band* dynamic
range the encoder stem sees (so D1/D2/D3 are no longer 10x below L3) while
preserving every *within-band, cross-window* amplitude relationship. It is
also deployment-safe — the same three scalars travel with the model as a
frozen config/buffer (see ``_stack_detail_bands`` + the band-std note), so the
on-device encoder applies the identical rescale.

The std is computed with a streaming (Welford-free, sum-of-squares) reducer so
the full corpus never has to be resident: we accumulate ``count`` and
``sum(x^2)`` per band and take ``sqrt(sumsq / count)`` at the end. The bands
are zero-mean by construction (DWT detail coefficients of a highpassed,
LPC-residualized signal), so we use the RMS about zero as the std — this is the
exact scale ``_stack_detail_bands`` divides by, and avoids a two-pass mean.

Reuse
-----
This drives the SAME decode + preprocess path the trainer uses, so the measured
coefficients are bit-identical to what ``_stack_detail_bands`` will normalize:

  - ``LmaDataset`` (lma_dataset.py) for the train-split stem index (honors the
    split manifest, so val/test recordings never leak into the std).
  - ``_decode_and_preprocess`` -> canonical ``decode_lma_signal`` for the
    decode (same Q31 round-trip as ``_compute_l3_stack``).
  - ``_preprocess_subband_single`` for the per-window subband dicts (same
    ``order=8, autocorr_len=256`` as ``_compute_l3_stack``).

Usage
-----
    PYTHONPATH=/mnt/4tb/LamQuant/blut/python \
    python -u -m lamquant.snn.precompute_band_std \
        --lma-root /mnt/4tb/data/Archive/lma/tuh /mnt/4tb/data/Archive/lma/physionet \
        --split-manifest /mnt/4tb/data/Training/manifests/split_v11.json \
        --out blut/python/lamquant/snn/band_std.json \
        --max-stems 4000 --workers 8

``--max-stems`` subsamples the train stems (evenly spaced) for a fast estimate;
the global std of a band is extremely stable across a few thousand recordings
(1/f scale is a corpus property, not a per-record one), so a few-K subsample
matches the full-corpus number to <1 %. Omit it for the exact full-corpus run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Resolve the lamquant package root (blut/python) so `lamquant.snn.*` and the
# `common/` DTO dir import regardless of how the script is launched — mirrors
# lma_dataset.py's _REPO bootstrap.
_LAMQUANT = Path(__file__).resolve().parents[1]            # lamquant/
_REPO = _LAMQUANT.parent                                   # blut/python
for _p in (str(_REPO), str(_LAMQUANT), str(_LAMQUANT / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lamquant.snn.lma_dataset import (              # noqa: E402
    LmaDataset,
    WINDOW_SAMPLES,
    TARGET_CHANNELS,
    _decode_and_preprocess,
    _lazy_imports,
)

# The three detail bands (l3_approx is L3 itself -> never normalized, it sets
# the reference scale). Order is fixed + matches _stack_detail_bands / ADR 0049.
DETAIL_BANDS: Tuple[str, ...] = ("l3_detail", "l2_detail", "l1_detail")

# Schema version of the band_std JSON. Bump if the band set / decode pipeline
# changes so a stale file is rejected loudly rather than silently mis-scaling.
BAND_STD_SCHEMA = 1


def _band_sumsq_for_stem(lma_path: str, stem: str,
                         lml_internal: Optional[str]) -> Optional[Dict[str, Tuple[float, int]]]:
    """Stream one recording -> per-band (sum-of-squares, count).

    Decodes + preprocesses every full 2500-sample window of the recording
    (same path as ``_compute_l3_stack``) and accumulates ``sum(coeff^2)`` and
    the coefficient ``count`` per detail band, pooled over all 21 channels and
    all windows. Returns None on decode failure (the stem is simply skipped —
    a few unreadable recordings do not move a corpus-global scalar).
    """
    _lazy_imports()
    # Imported lazily so the heavy DSP primitive loads once per worker.
    from lamquant.student.subband_preprocess import preprocess_subband_single

    signal = _decode_and_preprocess(lma_path, stem, lml_internal=lml_internal)
    if signal is None:
        return None
    T = signal.shape[1]
    n_full = T // WINDOW_SAMPLES
    if n_full == 0:
        return None  # sub-window recordings contribute nothing reliable

    acc: Dict[str, Tuple[float, int]] = {b: (0.0, 0) for b in DETAIL_BANDS}
    acc["l3_approx"] = (0.0, 0)   # L3 reference std: the TARGET scale detail is
                                  # rescaled TO (consumer divides by std/l3_ref),
                                  # so we must measure it; L3 is never itself rescaled.
    for w in range(n_full):
        s = w * WINDOW_SAMPLES
        window = signal[:, s:s + WINDOW_SAMPLES].astype(np.float32)
        _l3, _coeffs, subs = preprocess_subband_single(
            window, order=8, autocorr_len=256)
        l3f = np.asarray(_l3, dtype=np.float64)
        ssq_l3, cnt_l3 = acc["l3_approx"]
        acc["l3_approx"] = (ssq_l3 + float(np.dot(l3f.ravel(), l3f.ravel())),
                            cnt_l3 + l3f.size)
        for band in DETAIL_BANDS:
            ssq, cnt = acc[band]
            for d in subs:                       # one dict per channel
                v = np.asarray(d[band], dtype=np.float64)
                ssq += float(np.dot(v, v))       # sum of squares about zero
                cnt += v.size
            acc[band] = (ssq, cnt)
    return acc


def _select_stems(ds: LmaDataset, max_stems: Optional[int]
                  ) -> List[Tuple[str, str, Optional[str]]]:
    """Unique (lma_path, stem, lml_internal) over the dataset's window index.

    ``ds.index`` carries one entry PER WINDOW; we collapse to per-recording
    (the decode is per-recording) and, when ``max_stems`` is set, take an
    evenly-spaced subsample so the estimate spans the corpus rather than its
    first N alphabetical stems.
    """
    seen: Dict[str, Tuple[str, str, Optional[str]]] = {}
    for entry in ds.index:
        lma_path, stem, _win, lml_internal = (
            str(entry[0]), entry[1], entry[2], entry[3])
        if stem not in seen:
            seen[stem] = (lma_path, stem, lml_internal)
    stems = list(seen.values())
    if max_stems is not None and 0 < max_stems < len(stems):
        idx = np.linspace(0, len(stems) - 1, max_stems, dtype=int)
        stems = [stems[i] for i in idx]
    return stems


def compute_band_std(lma_root: List[str], split_manifest: str,
                     max_stems: Optional[int] = None,
                     workers: int = 8) -> Dict[str, float]:
    """Return {band: global_std} over the TRAIN split of the manifest.

    Builds the train-split ``LmaDataset`` (so val/test never leak), streams
    each recording through the decode+preprocess path, and reduces to one std
    per band, plus l3_approx — the reference scale each detail band is rescaled
    TO (consumer: `inv_scale = l3_ref / band_std`). L3 itself is never rescaled.
    """
    # Build train-split dataset purely for its honest, manifest-filtered stem
    # index. We never call __getitem__ (no L3 cache, no label decode) — we
    # only read ds.index, so SNN_DETAIL_BANDS / cache env vars are irrelevant.
    ds = LmaDataset(
        lma_dir=Path(lma_root[0]) if len(lma_root) == 1 else None,
        lma_paths=None if len(lma_root) == 1 else _glob_roots(lma_root),
        split="train",
        split_manifest_path=Path(split_manifest),
        derive_labels_from_lma=True,
    )
    stems = _select_stems(ds, max_stems)
    print(f"[band-std] {len(stems)} unique train recordings "
          f"(max_stems={max_stems}); reducing with {workers} workers")

    # Reduce includes l3_approx (the reference scale) alongside the detail bands.
    reduce_bands = DETAIL_BANDS + ("l3_approx",)
    totals: Dict[str, Tuple[float, int]] = {b: (0.0, 0) for b in reduce_bands}
    n_ok = n_fail = 0
    t0 = time.time()

    def _merge(part: Optional[Dict[str, Tuple[float, int]]]) -> None:
        nonlocal n_ok, n_fail
        if part is None:
            n_fail += 1
            return
        n_ok += 1
        for b in reduce_bands:
            tssq, tcnt = totals[b]
            pssq, pcnt = part[b]
            totals[b] = (tssq + pssq, tcnt + pcnt)

    if workers <= 1:
        for i, (lp, st, li) in enumerate(stems):
            _merge(_band_sumsq_for_stem(lp, st, li))
            if (i + 1) % 200 == 0:
                print(f"[band-std]   {i + 1}/{len(stems)} "
                      f"({n_ok} ok, {n_fail} fail, {time.time() - t0:.0f}s)")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_band_sumsq_for_stem, lp, st, li): st
                    for (lp, st, li) in stems}
            done = 0
            for fut in as_completed(futs):
                try:
                    _merge(fut.result())
                except Exception as e:        # one bad worker never aborts the run
                    n_fail += 1
                    print(f"[band-std]   worker failed for "
                          f"{futs[fut]}: {e}", file=sys.stderr)
                done += 1
                if done % 200 == 0:
                    print(f"[band-std]   {done}/{len(stems)} "
                          f"({n_ok} ok, {n_fail} fail, {time.time() - t0:.0f}s)")

    if n_ok == 0:
        raise RuntimeError("band-std: every recording failed to decode — "
                           "check --lma-root / --split-manifest")

    out: Dict[str, float] = {}
    for b in reduce_bands:
        ssq, cnt = totals[b]
        if cnt == 0:
            raise RuntimeError(f"band-std: zero coefficients for {b}")
        std = math.sqrt(ssq / cnt)
        if not math.isfinite(std) or std <= 0.0:
            raise RuntimeError(f"band-std: non-positive std for {b}: {std}")
        out[b] = std
    print(f"[band-std] DONE: {n_ok} ok / {n_fail} fail in "
          f"{time.time() - t0:.0f}s  ->  {out}")
    return out


def _glob_roots(roots: List[str]) -> List[Path]:
    """One/two-level-deep ``*.lma`` glob over multiple roots (mirrors
    LmaDataset's own resolution so ``--lma-root a b c`` works like the trainer's
    multi-root path)."""
    paths: List[Path] = []
    for r in roots:
        base = Path(r)
        found = sorted(base.glob("*/*.lma")) or sorted(base.glob("*.lma"))
        if not found:
            raise RuntimeError(f"no .lma archives under {base}")
        paths.extend(found)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lma-root", nargs="+", required=True,
                    help="one or more dirs of per-corpus/per-recording .lma "
                         "(space-separated; nargs='+', NOT repeated flags)")
    ap.add_argument("--split-manifest", required=True,
                    help="split manifest JSON (train split is the std source)")
    ap.add_argument("--out", required=True,
                    help="output JSON path (e.g. "
                         "blut/python/lamquant/snn/band_std.json)")
    ap.add_argument("--max-stems", type=int, default=None,
                    help="evenly-spaced subsample of train recordings for a "
                         "fast estimate (omit for exact full-corpus)")
    ap.add_argument("--workers", type=int, default=8,
                    help="process pool size for the decode+DWT reduce")
    args = ap.parse_args()

    band_std = compute_band_std(
        lma_root=args.lma_root,
        split_manifest=args.split_manifest,
        max_stems=args.max_stems,
        workers=args.workers,
    )

    payload = {
        "schema": BAND_STD_SCHEMA,
        # Frozen scalars consumed by _stack_detail_bands: the three detail stds
        # plus l3_approx, the reference scale detail is rescaled TO (L3 itself is
        # never rescaled — the consumer only divides detail by std/l3_ref).
        "band_std": band_std,
        "provenance": {
            "lma_root": args.lma_root,
            "split_manifest": args.split_manifest,
            "split": "train",
            "max_stems": args.max_stems,
            "window_samples": WINDOW_SAMPLES,
            "channels": TARGET_CHANNELS,
            "lpc_order": 8,
            "autocorr_len": 256,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[band-std] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
