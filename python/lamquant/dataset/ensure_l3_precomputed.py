#!/usr/bin/env python3
"""
Ensure L3 precomputation is complete before training.

This script is called by training_cockpit and training scripts to:
1. Check if L3 arrays are present in all Q31 NPZ files
2. Run precompute_l3_fast.py if any are missing
3. Provide a clean status report

Usage:
    python ensure_l3_precomputed.py [--force] [--workers N]

Options:
    --force         Force re-precomputation even if L3 arrays exist
    --workers N     Number of parallel workers (default: 8)
"""

import os
import sys
import glob
import argparse
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm

def check_l3_status(q31_dir):
    """
    Check which NPZ files are missing L3 precomputation.

    Returns:
        (missing_files, total_files) where missing_files is a list of paths
    """
    npz_files = sorted(glob.glob(os.path.join(q31_dir, '*.npz')))

    if not npz_files:
        return [], 0

    missing = []
    print(f"[*] Scanning {len(npz_files)} NPZ files for L3 arrays...")
    for npz_path in tqdm(npz_files, desc="Checking L3 status"):
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if 'l3' not in data:
                    missing.append(npz_path)
        except Exception as e:
            print(f"[!] Error reading {npz_path}: {e}")
            missing.append(npz_path)

    return missing, len(npz_files)


def run_precomputation(q31_dir, workers=8):
    """Run precompute_l3_fast.py."""
    script_path = os.path.join(os.path.dirname(__file__), '../student/precompute_l3_fast.py')

    if not os.path.exists(script_path):
        print(f"[!] Precomputation script not found: {script_path}")
        return False

    print(f"[*] Running L3 precomputation with {workers} workers...")
    try:
        result = subprocess.run(
            ['python', script_path, '--input', q31_dir, '--workers', str(workers)],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[!] Precomputation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[!] Failed to run precomputation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Ensure L3 precomputation is complete"
    )
    parser.add_argument(
        '--q31-dir',
        default='ai_models/dataset_sim/q31_events',
        help='Directory containing Q31 NPZ files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-precomputation even if L3 arrays exist'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers for precomputation'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    q31_dir = args.q31_dir
    if not os.path.exists(q31_dir):
        print(f"[!] Q31 directory not found: {q31_dir}")
        return False

    # Check status
    if args.force:
        missing_files = sorted(glob.glob(os.path.join(q31_dir, '*.npz')))
        total_files = len(missing_files)
        print(f"[*] --force flag set: Will re-precompute all {total_files} files")
    else:
        missing_files, total_files = check_l3_status(q31_dir)

    if not missing_files:
        print(f"\n✓ L3 precomputation complete: all {total_files} files have L3 arrays")
        return True

    print(f"\n[*] Found {len(missing_files)}/{total_files} files missing L3 arrays")

    # Run precomputation
    success = run_precomputation(q31_dir, args.workers)

    if success:
        # Re-check status
        missing_after, _ = check_l3_status(q31_dir)
        if missing_after:
            print(f"[!] After precomputation, {len(missing_after)} files still missing L3")
            return False
        print(f"\n✓ L3 precomputation successful: all files ready for training")
        return True
    else:
        print("\n[!] L3 precomputation failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
