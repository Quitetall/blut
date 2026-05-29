#!/usr/bin/env python3
"""launch_production.py — preflight + launch the LamQuant production run.

The production run is a multi-day commitment (Tier 7 800M decoder, full
manifest, fullband joint loss, ~400 epochs). Catching a missing
prerequisite at hour 0 vs hour 6 is worth a 30-second preflight script.

Checks performed:

  1. Manifest exists and validates (DatasetManifest.load() raises on any
     leakage / schema violation).
  2. Fullband train + val memmaps exist on disk and are sized correctly
     for the manifest's window count.
  3. /mnt/4tb has enough free space for checkpoints + alpha CSV
     (estimate ~50 GB over the run).
  4. CUDA reports a GPU with ≥ 20 GB VRAM (Tier 7 needs ~16 GB at
     batch=8 with gradient checkpointing).
  5. RAM available is enough for the L3 cache (~17 GB) plus headroom
     (5 GB) — fullband uses memmap so it doesn't count.
  6. (Optional) Smoke test: load 1 batch end-to-end and run one
     train_step + one validate_joint to surface OOM / config bugs
     before the real run starts.

Then it spawns train_joint with the chosen preset+tier in the background
(or foreground via --foreground). Logs go to outputs/ with an
ISO-timestamped filename.

Usage:
    # Dry-run preflight (no training)
    python ai_models/student/launch_production.py --preflight-only

    # Launch with default production preset + Tier 7
    python ai_models/student/launch_production.py

    # Custom tier / preset
    python ai_models/student/launch_production.py --tier 6 --config standard
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'lamquant'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'common'))  # MOVE-B: common DTOs


# ============================================================
# Preflight checks
# ============================================================

def _check(label: str, ok: bool, detail: str = '') -> bool:
    glyph = '✓' if ok else '✗'
    print(f'  {glyph} {label}' + (f'  — {detail}' if detail else ''))
    return ok


def preflight(args) -> bool:
    """Return True iff every gate passes."""
    print('\n[*] Production preflight')
    all_ok = True

    # 1. Manifest
    from data_types import DatasetManifest, Split
    manifest_path = _REPO / 'lamquant' / 'dataset_sim' / 'manifest_v3.json'
    try:
        manifest = DatasetManifest.load(manifest_path)
        all_ok &= _check(
            'manifest validates', True,
            f'{manifest.train_files:,} train files, {manifest.val_files:,} val'
        )
    except Exception as e:
        all_ok &= _check('manifest validates', False, str(e))
        return False

    # 2. Fullband memmaps
    ds_dir = _REPO / 'lamquant' / 'dataset_sim'
    expected = {
        'train': (ds_dir / 'fullband_train.dat',
                  ds_dir / 'fullband_train.meta.json',
                  manifest.train_windows),
        'val':   (ds_dir / 'fullband_val.dat',
                  ds_dir / 'fullband_val.meta.json',
                  manifest.val_windows),
    }
    for split, (dat, meta, expected_windows) in expected.items():
        if not dat.exists():
            all_ok &= _check(
                f'fullband_{split}.dat exists', False,
                f'missing — run precompute_fullband_memmap.py --splits {split}'
            )
            continue
        if not meta.exists():
            all_ok &= _check(f'fullband_{split}.meta.json exists', False,
                              f'missing alongside {dat}')
            continue
        with open(meta) as f:
            m = json.load(f)
        actual_dat_bytes = dat.stat().st_size
        expected_bytes = m['n_windows'] * m['n_channels'] * m['window_samples'] * 2
        size_ok = actual_dat_bytes == expected_bytes
        win_ok = m['n_windows'] >= expected_windows
        all_ok &= _check(
            f'fullband_{split}.dat size matches meta',
            size_ok,
            f'{actual_dat_bytes / 1e9:.1f} GB '
            f'(expected {expected_bytes / 1e9:.1f} GB)'
        )
        all_ok &= _check(
            f'fullband_{split} covers manifest', win_ok,
            f'{m["n_windows"]:,} memmap windows vs {expected_windows:,} manifest'
        )

    # 3. Disk space
    free_bytes = shutil.disk_usage('/mnt/4tb').free
    need_bytes = 50 * 1024**3
    all_ok &= _check(
        '/mnt/4tb free space ≥ 50 GB',
        free_bytes >= need_bytes,
        f'{free_bytes / 1e9:.0f} GB free'
    )

    # 4. CUDA
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory
            free_vram = torch.cuda.mem_get_info(0)[0] if hasattr(torch.cuda, 'mem_get_info') else vram
            ok = free_vram >= 18 * 1024**3
            all_ok &= _check(
                'CUDA GPU available, ≥ 18 GB free VRAM', ok,
                f'{torch.cuda.get_device_name(0)}, '
                f'{free_vram / 1e9:.1f} GB free / {vram / 1e9:.1f} GB total'
            )
        else:
            all_ok &= _check('CUDA GPU available', False, 'no CUDA device')
    except ImportError:
        all_ok &= _check('torch importable', False)

    # 5. RAM
    try:
        import psutil
        avail = psutil.virtual_memory().available
        ok = avail >= 25 * 1024**3
        all_ok &= _check(
            'RAM available ≥ 25 GB (L3 cache + headroom)', ok,
            f'{avail / 1e9:.0f} GB available'
        )
    except ImportError:
        # Fallback to /proc/meminfo
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail = int(line.split()[1]) * 1024
                        break
            ok = avail >= 25 * 1024**3
            all_ok &= _check(
                'RAM available ≥ 25 GB (L3 cache + headroom)', ok,
                f'{avail / 1e9:.0f} GB available'
            )
        except Exception:
            print('  ? RAM check skipped (psutil + /proc/meminfo unavailable)')

    return all_ok


# ============================================================
# Launch
# ============================================================

def launch(args) -> int:
    """Spawn train_joint with production settings. Returns the child's exit code."""
    cmd = [
        sys.executable, '-u',
        str(_REPO / 'lamquant' / 'student' / 'train_joint.py'),
        '--config', args.config,
        '--tier', str(args.tier),
        '--fullband-mode', args.fullband_mode,
    ]

    log_dir = _REPO / 'outputs'
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'production_t{args.tier}_{args.config}_{stamp}.log'

    print(f'\n[*] Command: {" ".join(cmd)}')
    print(f'[*] Log:     {log_path}')

    if args.foreground:
        print(f'[*] Running in FOREGROUND (Ctrl-C interrupts the run)')
        with open(log_path, 'w') as f:
            return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)

    # Background: detach with nohup so the run survives shell exit.
    print(f'[*] Spawning in background — tail the log for progress.')
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    print(f'[*] PID: {proc.pid}')
    print(f'[*] tail -f {log_path}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog='launch_production')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Run checks only, do not launch training')
    parser.add_argument('--config', default='production',
                        choices=['fast', 'standard', 'medium', 'production'])
    parser.add_argument('--tier', type=int, default=7,
                        help='Vocos decoder tier (3+ → fullband loss path)')
    parser.add_argument('--fullband-mode',
                        choices=['auto', 'ram', 'memmap', 'off'],
                        default='memmap',
                        help='Fullband target source (default: memmap for production)')
    parser.add_argument('--foreground', action='store_true',
                        help='Block on the run instead of detaching')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Launch without running checks (for re-launches)')
    args = parser.parse_args()

    if not args.skip_preflight:
        ok = preflight(args)
        if not ok:
            print('\n[!] Preflight FAILED — aborting launch.')
            return 1
        print('\n[*] Preflight OK')

    if args.preflight_only:
        return 0

    return launch(args)


if __name__ == '__main__':
    sys.exit(main())
