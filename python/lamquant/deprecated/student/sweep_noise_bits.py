#!/usr/bin/env python3
"""Noise-bits ablation sweep.

Trains with 0-10 LSBs masked in the L3 input. Same data, same model,
same hyperparameters — only the noise floor changes.

Hypothesis: 1-6 bits masked → faster convergence (model capacity goes
to signal, not ADC thermal noise). 7+ bits → information loss begins
to hurt quality.

Uses the experiment runner's sweep() for auto-logging and leaderboard.

Usage:
    python sweep_noise_bits.py                  # full 0-10 sweep
    python sweep_noise_bits.py --max-bits 6     # quick 0-6 sweep
    python sweep_noise_bits.py --preset medium  # longer training per point
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'lamquant'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'student'))

from lamquant.common.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Noise-bits ablation sweep')
    parser.add_argument('--max-bits', type=int, default=10,
                        help='Maximum noise bits to test (default: 10)')
    parser.add_argument('--preset', default='fast',
                        help='Training preset (default: fast)')
    parser.add_argument('--tier', type=int, default=3,
                        help='Decoder tier (default: 3)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Random seeds (default: 0)')
    args = parser.parse_args()

    runner = ExperimentRunner()
    bits_range = list(range(0, args.max_bits + 1))

    print(f'\n{"="*60}')
    print(f'  Noise-Bits Ablation: {len(bits_range)} levels × {len(args.seeds)} seeds')
    print(f'  Preset: {args.preset} | Tier: {args.tier}')
    print(f'  Bits: {bits_range}')
    print(f'{"="*60}\n')

    results = runner.sweep(
        tier=args.tier,
        preset=args.preset,
        seeds=args.seeds,
        grid={'train_noise_bits': bits_range},
    )

    # Summary table
    print(f'\n{"="*60}')
    print(f'  NOISE-BITS ABLATION RESULTS')
    print(f'{"="*60}')
    print(f'  {"bits":>4s}  {"R":>8s}  {"PRD":>8s}  {"loss":>10s}  {"epochs":>6s}')
    print(f'  {"─"*4}  {"─"*8}  {"─"*8}  {"─"*10}  {"─"*6}')
    for r in results:
        if r is None:
            continue
        nb = r.get('config', {}).get('train_noise_bits', '?')
        R = r.get('best_val_r', 0)
        prd = r.get('best_val_prd', 0)
        loss = r.get('best_val_loss', 0)
        ep = r.get('best_epoch', 0)
        print(f'  {nb:>4}  {R:8.4f}  {prd:8.2f}  {loss:10.6f}  {ep:6d}')

    print(f'\nResults logged to outputs/experiment_log.jsonl')
    print(f'View leaderboard: python -m lamquant.common.experiment_runner leaderboard')


if __name__ == '__main__':
    main()
