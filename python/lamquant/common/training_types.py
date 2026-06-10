"""Unified training types for every LamQuant training script.

Single source of truth for:

  - what a training script reports per epoch (EpochReport)
  - what it summarises at the end of a run (RunSummary)
  - how those reports get to disk + the user (TrainingLogger)

Every training script — encoder solo, joint, distillation, SNN, decoder —
populates the same EpochReport dataclass. Every consumer (CSV logger,
dashboard, checkpoint metadata, tests, future TensorBoard hook) reads
from the same schema.

This eliminates the "10 different readouts" problem: when you debug a
failed run, you no longer mentally parse three log formats and
cross-reference by epoch number. There is one CSV per run with every
field always present (zeros where the script doesn't fill them).

Design constraints
------------------
1. Python-stdlib only — no torch / numpy at import time. Lets the
   training_types module be imported from CI tests, dashboards, the GUI,
   etc. without dragging in CUDA.

2. JSON-serialisable. Every field is a primitive or a dict of primitives.

3. Backward-compatible by extension only. Adding a new field is fine
   (defaults to 0 / "" / False). Renaming or removing a field requires
   a CSV/JSON schema bump + migration of historical logs.

4. Extra-strict CSV column ordering — the CSV header order is fixed by
   the dataclass field order. Don't reorder fields casually.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, Union


# ============================================================
# EpochReport — one per validation event
# ============================================================

@dataclass
class EpochReport:
    """Single source of truth for one epoch's training state.

    Every training script populates this. Every consumer reads from it.
    Different scripts fill different subsets of fields — the SNN trainer
    fills `sparsity` but not `decoder_tier`; the decoder trainer fills
    `decoder_params` but not `tau`. Unfilled fields stay at their
    defaults (0.0 / "" / {}).
    """
    # ----- Identity -----
    run_id: str = ''                  # unique per training run
    script: str = ''                  # which training script produced this
    phase: str = ''                   # 'warm', 'qat', 'fine', 'distill'
    epoch: int = 0                    # epoch within current phase
    global_epoch: int = 0             # epoch across all phases
    total_epochs: int = 0             # planned epochs in the whole run
    timestamp: float = 0.0            # time.time() when this report was made

    # ----- Primary metrics — both required for LQS compliance -----
    # R measures shape preservation; PRD measures magnitude preservation.
    # A model can have R = 0.95 with PRD = 25% (right shape, wrong scale)
    # which fails LQS-Clinical. Both gate ship/no-ship.
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_r: float = 0.0                # Pearson R (correlation; shape)
    val_prd: float = 0.0              # PRD % (magnitude; lower is better)
    best_val_r: float = 0.0
    best_val_prd: float = 100.0       # lower is better, start at "worst"
    best_epoch: int = 0

    # Per-band PRD (computed at validation only — bandpass filtering
    # is too expensive every batch). Each is the dataset-wide mean PRD
    # in that EEG band (0 = perfect; 100 = noise dominates the band).
    val_prd_delta: float = 0.0        # 0.5–4 Hz (sleep, encephalopathy)
    val_prd_theta: float = 0.0        # 4–8 Hz   (drowsiness, temporal pathology)
    val_prd_alpha: float = 0.0        # 8–13 Hz  (posterior dominant rhythm)
    val_prd_beta: float = 0.0         # 13–30 Hz (frontal, medication)
    val_prd_gamma: float = 0.0        # 30–50 Hz (mostly EMG)

    # ----- Loss components (each script fills what's relevant) -----
    l_mse: float = 0.0
    l_r: float = 0.0
    l_spectral: float = 0.0
    l_perceptual: float = 0.0
    l_adversarial: float = 0.0
    l_distill: float = 0.0
    l_commitment: float = 0.0

    # ----- Encoder state -----
    encoder_params: int = 0
    sparsity: float = 0.0             # % ternary zeros
    tau: float = 0.0                   # quantization temperature
    quantize_active: bool = False

    # ----- Per-layer alpha (separate CSV for these) -----
    alpha_per_layer: dict = field(default_factory=dict)
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    alpha_mean: float = 0.0

    # ----- Decoder state -----
    decoder_params: int = 0
    decoder_tier: str = ''

    # ----- Optimizer / training dynamics -----
    lr: float = 0.0
    grad_norm: float = 0.0

    # ----- Hardware -----
    gpu_util: float = 0.0
    vram_gb: float = 0.0
    gpu_temp: float = 0.0

    # ----- Timing -----
    secs_per_epoch: float = 0.0        # wall-clock seconds for this epoch
    eta_hours: float = 0.0             # estimated hours remaining

    # ----- Checkpointing -----
    saved_checkpoint: bool = False
    checkpoint_path: str = ''
    improvement: bool = False         # did val_r beat best?

    # ----- Guard status -----
    guard_warnings: int = 0
    early_stopped: bool = False
    halt_reason: str = ''

    def to_dict(self) -> dict:
        """JSON-serializable dict (used by tests + JSON consumers)."""
        return asdict(self)


# ============================================================
# RunSummary — one per training run, written at the end
# ============================================================

@dataclass
class RunSummary:
    """Final summary emitted when training completes (or halts)."""
    # ----- Identity -----
    run_id: str = ''
    script: str = ''
    config: dict = field(default_factory=dict)

    # ----- Outcome -----
    completed: bool = False
    halted: bool = False
    halt_reason: str = ''

    # ----- Best results — R + PRD co-equal -----
    best_val_r: float = 0.0
    best_val_prd: float = 100.0
    best_epoch: int = 0
    final_val_r: float = 0.0
    final_val_prd: float = 100.0
    final_epoch: int = 0

    # Per-band PRD at the best checkpoint — what the LQS check evaluates.
    best_prd_delta: float = 0.0
    best_prd_theta: float = 0.0
    best_prd_alpha: float = 0.0
    best_prd_beta: float = 0.0
    best_prd_gamma: float = 0.0

    # LQS compliance result on the best checkpoint.
    # `lqs_level` is one of 'C' (clinical), 'M' (monitoring), 'A' (alerting),
    # or '' (below LQS-A — model not deployable).
    # `lqs_violations` lists what blocked the next-stricter tier (i.e. the
    # to-do list to reach LQS-C from LQS-M, etc.).
    lqs_level: str = ''
    lqs_violations: list = field(default_factory=list)

    # ----- Timing -----
    start_time: float = 0.0
    end_time: float = 0.0
    total_seconds: float = 0.0

    # ----- Artifacts -----
    encoder_checkpoint: str = ''
    decoder_checkpoint: str = ''
    alpha_csv: str = ''
    epoch_csv: str = ''
    summary_json: str = ''

    # ----- Data -----
    train_windows: int = 0
    val_windows: int = 0
    total_epochs_run: int = 0
    total_guard_warnings: int = 0

    # ----- Per-phase summary -----
    phases: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# TrainingLogger — owns all output formats
# ============================================================

# Fields excluded from the per-epoch CSV (they have separate handling).
_EPOCH_CSV_EXCLUDE = {'alpha_per_layer'}


def _csv_columns() -> list[str]:
    """Stable column order for the per-epoch CSV.

    Ordering follows the dataclass field declaration order. This matters
    because downstream tools may rely on column position; if you add a
    field, add it at the END of EpochReport so old CSV consumers still work.
    """
    return [f.name for f in fields(EpochReport) if f.name not in _EPOCH_CSV_EXCLUDE]


class TrainingLogger:
    """Unified logger for every training script.

    One entry point: `log_epoch(EpochReport)`. The logger handles all
    output formats (per-epoch CSV, per-layer alpha CSV, terminal
    dashboard line, JSON summary at the end).
    """

    def __init__(self, run_id: str, log_dir: Union[str, Path]):
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_csv = self.log_dir / f'epochs_{run_id}.csv'
        self.alpha_csv = self.log_dir / f'alpha_{run_id}.csv'
        self.summary_json = self.log_dir / f'summary_{run_id}.json'
        self._epoch_csv_initialized = False
        self._alpha_csv_initialized = False
        self.history: list[EpochReport] = []

    # --------------------------------------------------------
    # Single entry point per epoch
    # --------------------------------------------------------

    def log_epoch(self, report: EpochReport) -> None:
        """Append one epoch to all consumers (CSV, dashboard, history)."""
        # Stamp identity / timestamp if the caller didn't.
        if not report.run_id:
            report.run_id = self.run_id
        if not report.timestamp:
            report.timestamp = time.time()

        self.history.append(report)
        self._write_epoch_csv(report)
        self._write_alpha_csv(report)
        self._print_dashboard(report)

    def log_summary(self, summary: RunSummary) -> None:
        """Write the JSON summary + print the final block."""
        if not summary.run_id:
            summary.run_id = self.run_id
        # Persist artifact paths into the summary so consumers don't have
        # to reconstruct them from the run_id.
        summary.epoch_csv = str(self.epoch_csv)
        summary.alpha_csv = str(self.alpha_csv)
        summary.summary_json = str(self.summary_json)
        self.summary_json.write_text(
            json.dumps(asdict(summary), indent=2, default=str))
        self._print_summary(summary)

    # --------------------------------------------------------
    # CSV writers
    # --------------------------------------------------------

    def _write_epoch_csv(self, r: EpochReport) -> None:
        """One CSV row per epoch with all fields (except per-layer alpha)."""
        cols = _csv_columns()
        row = {k: v for k, v in asdict(r).items() if k not in _EPOCH_CSV_EXCLUDE}
        # Filter to the column set; force ordering.
        row = {c: row.get(c, '') for c in cols}
        if not self._epoch_csv_initialized:
            with open(self.epoch_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerow(row)
            self._epoch_csv_initialized = True
        else:
            with open(self.epoch_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writerow(row)

    def _write_alpha_csv(self, r: EpochReport) -> None:
        """One row per (layer, epoch). Long format = easy to pivot."""
        if not r.alpha_per_layer:
            return
        rows = [
            {'epoch': r.global_epoch, 'phase': r.phase,
             'layer': name, 'alpha': float(val)}
            for name, val in r.alpha_per_layer.items()
        ]
        cols = ['epoch', 'phase', 'layer', 'alpha']
        if not self._alpha_csv_initialized:
            with open(self.alpha_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)
            self._alpha_csv_initialized = True
        else:
            with open(self.alpha_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writerows(rows)

    # --------------------------------------------------------
    # Terminal dashboard
    # --------------------------------------------------------

    def _print_dashboard(self, r: EpochReport) -> None:
        """One-line per-epoch summary. Always printed — every epoch, not just
        at val_interval. R/PRD only appear when validation ran (non-zero)."""
        phase_tag = f"[{r.phase.upper():5s}]"
        quant_tag = 'T' if r.quantize_active else 'F'
        save_tag = ' ✓' if r.saved_checkpoint else '  '
        # Show R/PRD only when validation ran this epoch (0.0 = not run).
        r_str   = f"R={r.val_r:.4f}  " if r.val_r   else "              "
        prd_str = f"PRD={r.val_prd:.1f}%  " if r.val_prd else "             "
        best_str = f"best={r.best_val_r:.4f}  " if r.best_val_r else ""
        vram = f"vram={r.vram_gb:.1f}G  " if r.vram_gb else ""
        timing = (f"{r.secs_per_epoch/3600:.2f}h/ep  ETA~{r.eta_hours:.0f}h"
                  if r.secs_per_epoch else "")
        print(
            f"  {phase_tag}{save_tag}"
            f" ep {r.global_epoch:>4d}/{r.total_epochs:<4d}"
            f"  loss={r.train_loss:.4f}"
            f"  {r_str}{prd_str}{best_str}"
            f"lr={r.lr:.2e}"
            f"  α={r.alpha_mean:.3f}"
            f"  Q={quant_tag}"
            f"  {vram}{timing}"
        )

    def _print_summary(self, s: RunSummary) -> None:
        """End-of-run block. The single screen that determines ship/no-ship.

        Layout (the user-facing summary the team actually reads after a run):

            ┌─ status (COMPLETED / HALTED / INCOMPLETE)
            ├─ Best R + PRD at best epoch     ← primary numbers
            ├─ Per-band PRD breakdown         ← why something fails LQS
            ├─ LQS level achieved + violations ← the to-do list
            ├─ Final R + PRD (vs best — drift?)
            └─ Artifacts (ckpt paths, csv paths)
        """
        hours = s.total_seconds / 3600
        if s.completed:
            status = 'COMPLETED'
        elif s.halted:
            status = f'HALTED: {s.halt_reason or "(no reason given)"}'
        else:
            status = 'INCOMPLETE'

        # Per-band PRD line (only print if any band was populated)
        band_glyph = (('delta', 'δ'), ('theta', 'θ'), ('alpha', 'α'),
                      ('beta', 'β'), ('gamma', 'γ'))
        band_vals = [getattr(s, f'best_prd_{b}', 0.0) for b, _ in band_glyph]
        have_bands = any(v > 0 for v in band_vals)

        print()
        print('=' * 72)
        print(f'  {status}  ({s.script})')
        print(f'  Best R: {s.best_val_r:.4f}   PRD: {s.best_val_prd:.1f}%   '
              f'at epoch {s.best_epoch}')
        if have_bands:
            band_str = '  '.join(
                f'{glyph} {getattr(s, f"best_prd_{b}", 0.0):.1f}%'
                for b, glyph in band_glyph
            )
            print(f'  Per-band PRD: {band_str}')
        if s.lqs_level:
            level_name = {'C': 'Clinical', 'M': 'Monitoring',
                          'A': 'Alerting'}.get(s.lqs_level, s.lqs_level)
            print(f'  LQS Level:    {s.lqs_level} ({level_name})')
        else:
            print(f'  LQS Level:    -- (below LQS-A floor; not deployable)')
        if s.lqs_violations:
            # Show what blocked the NEXT-stricter tier — this is the to-do list.
            next_tier = {'M': 'C', 'A': 'M', '': 'A'}.get(s.lqs_level, '?')
            print(f'  To reach LQS-{next_tier}, fix:')
            for v in s.lqs_violations[:6]:
                print(f'    - {v}')
            if len(s.lqs_violations) > 6:
                print(f'    ... and {len(s.lqs_violations) - 6} more')
        print(f'  Final R: {s.final_val_r:.4f}   PRD: {s.final_val_prd:.1f}%   '
              f'at epoch {s.final_epoch}')
        print(f'  Duration: {hours:.2f} h ({s.total_epochs_run} epochs)')
        print(f'  Guard warnings: {s.total_guard_warnings}')
        if s.encoder_checkpoint:
            print(f'  Encoder: {s.encoder_checkpoint}')
        if s.decoder_checkpoint:
            print(f'  Decoder: {s.decoder_checkpoint}')
        print(f'  Logs: {s.epoch_csv}')
        if s.alpha_csv:
            print(f'  Alphas: {s.alpha_csv}')
        if s.summary_json:
            print(f'  Summary: {s.summary_json}')
        print('=' * 72)
        print()


# ============================================================
# Convenience helpers
# ============================================================

def alpha_stats_from_model(model) -> dict:
    """Extract per-layer LSQ alpha values into a flat dict.

    Returns {layer_name: alpha_max_value}. Use this to fill
    EpochReport.alpha_per_layer in any training script that has LSQ
    layers. Lazy torch import so this module stays stdlib-only.
    """
    import torch
    out = {}
    for name, m in model.named_modules():
        if hasattr(m, 'lsq_alpha'):
            out[name] = float(m.lsq_alpha.detach().abs().max())
    return out


def reduce_alpha_stats(per_layer: dict) -> tuple[float, float, float]:
    """Return (min, mean, max) across the per-layer alpha dict."""
    if not per_layer:
        return (0.0, 0.0, 0.0)
    vals = list(per_layer.values())
    return (min(vals), sum(vals) / len(vals), max(vals))


__all__ = [
    'EpochReport',
    'RunSummary',
    'TrainingLogger',
    'alpha_stats_from_model',
    'reduce_alpha_stats',
]
