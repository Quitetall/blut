#!/usr/bin/env python3
"""ai_models/experiment_runner.py — unified experiment runner.

One script to launch, track, compare, and reproduce any training
experiment. Replaces the scattered launch pattern of "cd into the right
directory, sys.path.insert the right modules, remember the right CLI
flags, then grep the log for results."

Design:

  from lamquant.common import ExperimentRunner

  runner = ExperimentRunner()

  # Single experiment
  result = runner.run(tier=3, preset='fast', seed=0,
                       notes='baseline fullband Tier 3')

  # A/B comparison
  a, b = runner.ab(
      a=dict(asymmetric_weight=0.0, notes='baseline'),
      b=dict(asymmetric_weight=0.2, notes='asymmetric env w=0.2'),
      tier=3, preset='fast', seeds=[0],
  )
  runner.compare(a, b)

  # Sweep
  results = runner.sweep(
      tier=3, preset='fast',
      grid={'asymmetric_weight': [0.0, 0.1, 0.2, 0.3]},
      seeds=[0, 42],
  )
  runner.leaderboard()

  # Reproduce from checkpoint provenance
  runner.reproduce('sha256:abc123...')   # looks up config hash in log

CLI:

  python -m lamquant.common.experiment_runner run --tier 3 --preset fast
  python -m lamquant.common.experiment_runner leaderboard
  python -m lamquant.common.experiment_runner compare <run_id_a> <run_id_b>
  python -m lamquant.common.experiment_runner sweep --tier 3 --grid '{"prd_weight": [0.05, 0.1, 0.2]}'

Every experiment is auto-logged to outputs/experiment_log.jsonl with
full provenance (manifest_hash + training_config_hash). The runner
is the ONLY way experiments should be launched during the
iterate-until-saturated phase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# MOVE-B (2026-05-29): now at blut/python/lamquant/common/. parents[2]
# is the blut/python package root, so `_REPO / 'lamquant' / <area>` adds
# the sibling area dirs for the bare cross-area imports this runner does.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'lamquant'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'common'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'student'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'oracle'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'decoder'))


# ============================================================
# Runner
# ============================================================

class ExperimentRunner:
    """Unified experiment launcher + tracker.

    Wraps train_joint.run() with:
      - Manifest preflight (once per runner, cached)
      - Experiment logging (every run auto-appended)
      - Config construction from keyword overrides
      - A/B and sweep orchestration
      - Leaderboard / compare output
    """

    def __init__(self, manifest_path: Optional[str] = None,
                 log_path: Optional[str] = None):
        from data_types import DatasetManifest
        self._manifest_path = str(manifest_path or (
            _REPO / 'lamquant' / 'dataset' / 'manifest_v3.json'))
        self._manifest = DatasetManifest.load(self._manifest_path)
        if log_path:
            from experiment_log import set_log_path
            set_log_path(log_path)
        print(f'[runner] manifest: {self._manifest.train_files:,} train, '
              f'{self._manifest.val_files:,} val '
              f'({len(self._manifest.datasets)} datasets)')

    # ---- Single experiment ----

    def run(self, *,
            tier: int = 3,
            preset: str = 'fast',
            seed: int = 0,
            fullband_mode: str = 'auto',
            amp: bool = True,
            compile_decoder: bool = True,
            asymmetric_weight: float = 0.0,
            asymmetric_kind: str = 'envelope',
            notes: str = '',
            tags: Optional[List[str]] = None,
            **config_overrides) -> dict:
        """Run one experiment. Returns the result dict from train_joint.run().

        `config_overrides` are applied on top of the named preset:

            runner.run(preset='fast', prd_weight=0.2, lr_quant=5e-4)

        The override produces a new TrainingConfig (the preset is never
        mutated) whose hash is distinct from the base preset's.
        """
        from training_config import CONFIGS
        from train_joint import run as _train_run

        cfg = CONFIGS[preset]
        if config_overrides:
            cfg = cfg.replace(**config_overrides)

        t0 = time.time()
        result = _train_run(
            cfg,
            vocos_tier=tier,
            seed=seed,
            fullband_mode=fullband_mode,
            amp=amp,
            compile_decoder=compile_decoder,
            asymmetric_weight=asymmetric_weight,
            asymmetric_kind=asymmetric_kind,
        )
        wall = time.time() - t0

        # Enrich result with runner-side metadata.
        result['wall_seconds'] = wall
        result['notes'] = notes
        result['tags'] = tags or []
        result['preset'] = preset
        result['tier'] = tier
        result['seed'] = seed
        result['asymmetric_weight'] = asymmetric_weight
        return result

    # ---- A/B comparison ----

    def ab(self, *,
           a: dict, b: dict,
           tier: int = 3, preset: str = 'fast',
           seeds: Sequence[int] = (0,),
           common: Optional[dict] = None,
           ) -> tuple:
        """Run two configs (A and B) side-by-side, one seed at a time.

        `a` and `b` are keyword dicts that override the preset:

            runner.ab(
                a=dict(asymmetric_weight=0.0, notes='baseline'),
                b=dict(asymmetric_weight=0.2, notes='asymmetric'),
                tier=3, preset='fast', seeds=[0, 42, 99],
            )

        Returns (a_results, b_results) where each is a list of result
        dicts (one per seed).
        """
        common = common or {}
        a_results, b_results = [], []
        for seed in seeds:
            kw = dict(tier=tier, preset=preset, seed=seed, **common)
            print(f'\n{"="*60}')
            print(f'  A/B seed={seed} — Run A')
            print(f'{"="*60}')
            a_results.append(self.run(**{**kw, **a}))

            print(f'\n{"="*60}')
            print(f'  A/B seed={seed} — Run B')
            print(f'{"="*60}')
            b_results.append(self.run(**{**kw, **b}))

        self._print_ab_summary(a_results, b_results, a, b)
        return a_results, b_results

    # ---- Grid sweep ----

    def sweep(self, *,
              tier: int = 3, preset: str = 'fast',
              grid: Dict[str, list],
              seeds: Sequence[int] = (0,),
              common: Optional[dict] = None,
              ) -> List[dict]:
        """Sweep a grid of hyperparameters. Each cell runs all seeds.

        grid is {param_name: [values]}. Total experiments = product of
        all grid lengths × len(seeds).

            runner.sweep(
                tier=3, preset='fast', seeds=[0, 42],
                grid={'prd_weight': [0.05, 0.1, 0.2],
                      'asymmetric_weight': [0.0, 0.2]},
            )
        """
        import itertools
        common = common or {}
        keys = list(grid.keys())
        values = list(grid.values())
        combos = list(itertools.product(*values))
        n_total = len(combos) * len(seeds)
        print(f'[runner] sweep: {len(combos)} combos × {len(seeds)} seeds '
              f'= {n_total} experiments')

        all_results = []
        for i, combo in enumerate(combos):
            overrides = dict(zip(keys, combo))
            for seed in seeds:
                tag = ', '.join(f'{k}={v}' for k, v in overrides.items())
                print(f'\n[runner] sweep {i*len(seeds)+seeds.index(seed)+1}'
                      f'/{n_total}: {tag}, seed={seed}')
                result = self.run(
                    tier=tier, preset=preset, seed=seed,
                    notes=f'sweep: {tag}',
                    tags=['sweep'],
                    **{**common, **overrides},
                )
                all_results.append(result)

        self._print_sweep_summary(all_results, keys)
        return all_results

    # ---- Recipe-based experiment ----

    def run_recipe(self, name: str, *,
                   tier: int = 3, preset: str = 'fast',
                   seed: int = 0, **extra_overrides) -> dict:
        """Run a named experiment recipe.

            runner.run_recipe('snac_balanced', tier=3, preset='fast')
            runner.run_recipe('combined_winners', tier=7, preset='production')

        Recipes are defined in lamquant.common.recipes. Each recipe specifies
        config overrides, augmentation, and FSQ preset. Extra overrides
        are applied on top of the recipe.
        """
        from lamquant.common.recipes import get_recipe
        recipe = get_recipe(name)
        overrides = {**recipe.get('config_overrides', {}), **extra_overrides}
        return self.run(
            tier=tier, preset=preset, seed=seed,
            notes=f"recipe:{name} — {recipe['desc']}",
            tags=['recipe', name],
            **overrides,
        )

    def list_recipes(self):
        """Print available experiment recipes."""
        from lamquant.common.recipes import list_recipes
        print(f'\n{"="*60}')
        print(f'  AVAILABLE RECIPES')
        print(f'{"="*60}')
        for name, desc in list_recipes():
            print(f'  {name:25} {desc}')
        print(f'{"="*60}')

    # ---- Leaderboard ----

    def leaderboard(self, limit: int = 20, metric: str = 'best_val_r'):
        """Print the experiment log as a sorted table."""
        from experiment_log import list_experiments, summary_table
        records = list_experiments(limit=limit)
        print(f'\n[runner] Leaderboard — top {limit} by timestamp '
              f'(newest first)')
        print(summary_table(records))

    # ---- Compare ----

    def compare(self, run_id_a: str, run_id_b: str):
        """Diff two logged experiments."""
        from experiment_log import compare as _compare
        diff = _compare(run_id_a, run_id_b)
        print(f'\n[runner] Diff {run_id_a} vs {run_id_b}:')
        for k, (va, vb) in sorted(diff.items()):
            if isinstance(va, float):
                print(f'  {k:30}  {va:>12.4f}  →  {vb:>12.4f}  '
                      f'(Δ={vb - va:+.4f})')
            else:
                print(f'  {k:30}  {str(va):>12}  →  {str(vb):>12}')

    # ---- Reproduce ----

    def reproduce(self, config_hash: str):
        """Find and reprint the config that produced a given hash."""
        from experiment_log import iter_records
        for r in iter_records():
            if r.training_config_hash == config_hash:
                print(f'\n[runner] Found run {r.run_id} with config hash '
                      f'{config_hash}')
                print(f'  preset={r.preset}, tier={r.vocos_tier}, '
                      f'seed={r.seed}')
                print(f'  R={r.best_val_r:.4f}, PRD={r.best_val_prd:.1f}%, '
                      f'LQS={r.lqs_level or "--"}')
                return r
        print(f'[runner] Config hash {config_hash} not found in log.')
        return None

    # ---- Internals ----

    def _print_ab_summary(self, a_results, b_results, a_kw, b_kw):
        import numpy as np
        def _mean(results, key):
            vals = [r.get(key, 0) for r in results]
            return float(np.mean(vals)) if vals else 0.0

        print(f'\n{"="*60}')
        print(f'  A/B SUMMARY ({len(a_results)} seeds)')
        print(f'{"="*60}')
        for label, kw, results in [('A', a_kw, a_results),
                                    ('B', b_kw, b_results)]:
            tag = ', '.join(f'{k}={v}' for k, v in kw.items()
                            if k != 'notes')
            r = _mean(results, 'best_val_r')
            prd = _mean(results, 'best_val_prd')
            wall = _mean(results, 'wall_seconds')
            print(f'  {label} ({tag}):')
            print(f'    R={r:.4f}  PRD={prd:.1f}%  wall={wall/60:.1f}min')
        delta_r = (_mean(b_results, 'best_val_r') -
                   _mean(a_results, 'best_val_r'))
        delta_prd = (_mean(b_results, 'best_val_prd') -
                     _mean(a_results, 'best_val_prd'))
        print(f'  Delta (B - A):  R={delta_r:+.4f}  PRD={delta_prd:+.1f}%')
        print(f'{"="*60}')

    def _print_sweep_summary(self, all_results, keys):
        import numpy as np
        print(f'\n{"="*60}')
        print(f'  SWEEP SUMMARY ({len(all_results)} experiments)')
        print(f'{"="*60}')
        # Group by config hash (unique combo), average over seeds.
        from collections import defaultdict
        groups = defaultdict(list)
        for r in all_results:
            # Build a stable key from the swept params
            gk = tuple(str(r.get(k, '?')) for k in keys)
            groups[gk].append(r)
        header = '  '.join(f'{k:>12}' for k in keys)
        print(f'  {header}  {"R":>8}  {"PRD":>8}  {"n":>3}')
        for gk, results in sorted(groups.items()):
            vals = '  '.join(f'{v:>12}' for v in gk)
            r = float(np.mean([x.get('best_val_r', 0) for x in results]))
            prd = float(np.mean([x.get('best_val_prd', 0) for x in results]))
            print(f'  {vals}  {r:>8.4f}  {prd:>7.1f}%  {len(results):>3}')
        print(f'{"="*60}')


# ============================================================
# CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog='experiment_runner',
        description='Unified experiment launcher + tracker.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ---- run ----
    p_run = sub.add_parser('run', help='Single experiment')
    p_run.add_argument('--tier', type=int, default=3)
    p_run.add_argument('--preset', default='fast',
                        choices=['fast', 'standard', 'medium', 'production'])
    p_run.add_argument('--seed', type=int, default=0)
    p_run.add_argument('--fullband-mode', default='auto')
    p_run.add_argument('--amp', dest='amp', action='store_true', default=True)
    p_run.add_argument('--no-amp', dest='amp', action='store_false')
    p_run.add_argument('--compile', dest='compile_decoder',
                        action='store_true', default=True)
    p_run.add_argument('--no-compile', dest='compile_decoder',
                        action='store_false')
    p_run.add_argument('--asymmetric-weight', type=float, default=0.0)
    p_run.add_argument('--asymmetric-kind', default='envelope')
    p_run.add_argument('--notes', default='')
    p_run.add_argument('--override', type=str, default='{}',
                        help='JSON dict of TrainingConfig overrides')

    # ---- leaderboard ----
    p_lb = sub.add_parser('leaderboard', help='Print experiment log')
    p_lb.add_argument('--limit', type=int, default=20)

    # ---- compare ----
    p_cmp = sub.add_parser('compare', help='Diff two runs')
    p_cmp.add_argument('run_a', type=str)
    p_cmp.add_argument('run_b', type=str)

    # ---- sweep ----
    p_sweep = sub.add_parser('sweep', help='Grid sweep')
    p_sweep.add_argument('--tier', type=int, default=3)
    p_sweep.add_argument('--preset', default='fast')
    p_sweep.add_argument('--seeds', type=str, default='0',
                          help='Comma-separated seeds')
    p_sweep.add_argument('--grid', type=str, required=True,
                          help='JSON dict of {param: [values]}')

    # ---- recipe ----
    p_recipe = sub.add_parser('recipe', help='Run a named experiment recipe')
    p_recipe.add_argument('name', type=str, help='Recipe name (use "list" to see all)')
    p_recipe.add_argument('--tier', type=int, default=3)
    p_recipe.add_argument('--preset', default='fast')
    p_recipe.add_argument('--seed', type=int, default=0)
    p_recipe.add_argument('--override', type=str, default='{}',
                           help='JSON dict of extra overrides on top of recipe')

    # ---- recipes (list) ----
    sub.add_parser('recipes', help='List available recipes')

    # ---- reproduce ----
    p_rep = sub.add_parser('reproduce', help='Find config by hash')
    p_rep.add_argument('config_hash', type=str)

    args = parser.parse_args()
    runner = ExperimentRunner()

    if args.command == 'run':
        overrides = json.loads(args.override) if args.override != '{}' else {}
        runner.run(
            tier=args.tier, preset=args.preset, seed=args.seed,
            fullband_mode=args.fullband_mode,
            amp=args.amp, compile_decoder=args.compile_decoder,
            asymmetric_weight=args.asymmetric_weight,
            asymmetric_kind=args.asymmetric_kind,
            notes=args.notes,
            **overrides,
        )
    elif args.command == 'leaderboard':
        runner.leaderboard(limit=args.limit)
    elif args.command == 'compare':
        runner.compare(args.run_a, args.run_b)
    elif args.command == 'sweep':
        seeds = [int(s) for s in args.seeds.split(',')]
        grid = json.loads(args.grid)
        runner.sweep(tier=args.tier, preset=args.preset,
                      seeds=seeds, grid=grid)
    elif args.command == 'recipe':
        extra = json.loads(args.override) if args.override != '{}' else {}
        runner.run_recipe(args.name, tier=args.tier, preset=args.preset,
                           seed=args.seed, **extra)
    elif args.command == 'recipes':
        runner.list_recipes()
    elif args.command == 'reproduce':
        runner.reproduce(args.config_hash)

    return 0


if __name__ == '__main__':
    sys.exit(main())
