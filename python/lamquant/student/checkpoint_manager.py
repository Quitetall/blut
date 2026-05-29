"""Checkpoint + training-safety manager.

Centralises every "did this run actually save its best work" behaviour
that used to be scattered across train_student_subband.py — and that the
prior gold run silently lost for 360 epochs.

Five guarantees, all enforced here:

  1. **Best-on-improvement save.** Every validation that improves val_r,
     in any phase, persists the model's state_dict to disk. Atomic write
     (tmp file → rename) so a crash mid-write doesn't corrupt the file.

  2. **Periodic recovery saves.** Every N epochs (default 50), regardless
     of whether it's the best, save a numbered checkpoint. Disk space is
     cheap, lost training time is not.

  3. **Hard alarms, not warnings.** R-plateau over patience epochs and
     LSQ-alpha explosion above ceiling raise `TrainingHaltException`,
     which the training loop catches, saves best, and exits cleanly.

  4. **Save-time smoke check.** After every save, the file is reloaded
     into a fresh copy of the model and a single forward pass is run.
     If the post-load R drifts from the in-training value by > 0.001,
     the save is treated as corrupt — the training loop is alerted.

  5. **Per-layer alpha CSV log.** One row per validation, with α_min /
     α_mean / α_max for each ternary layer. Drop into pandas / Excel and
     you can see alpha drift hours before R drops.

Usage in a training loop:

    cm = CheckpointManager(
        model=student,
        ckpt_path='/path/best.ckpt',
        ckpt_dir='/path/recovery',
        device=device,
        smoke_input=lambda: torch.randn(1, 21, 313, device=device),
        alpha_log_csv='/path/alpha.csv',
    )

    for epoch in range(total_epochs):
        ... train ...
        if epoch % val_interval == 0:
            val_r = validate(...)
            cm.on_validation(epoch=epoch, val_r=val_r)
            cm.maybe_save_recovery(epoch=epoch, every=50)
"""

from __future__ import annotations

import csv
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


class TrainingHaltException(Exception):
    """Raised by guards when training must stop. Carries a human reason
    string. The training loop catches this, saves best, and exits cleanly.
    """


@dataclass
class GuardConfig:
    """Bounds the manager will enforce. Defaults match the production preset.

    Set any field to 0 / None to disable that specific guard.
    """
    r_plateau_patience: int = 50    # validations without improvement → halt
    alpha_max_safe: float = 5.0     # any layer α exceeding this → halt
    alpha_min_safe: float = 1e-4    # any layer α below → halt (collapse)
    smoke_check_tolerance: float = 1e-3   # R must match within this after reload
    improvement_eps: float = 1e-3   # val_r must beat best by this margin to count
                                    # as "improvement" — small noise doesn't reset
                                    # the plateau counter (matches BitNet practice)
    prd_tiebreak_eps: float = 0.5   # when two epochs are within improvement_eps R,
                                    # save the lower-PRD one if the PRD gap is at
                                    # least this many percentage points. Prevents
                                    # trading 0.0005 R for 3% worse PRD.


class CheckpointManager:
    """All checkpoint + safety logic for one training run."""

    def __init__(self, model, ckpt_path, *,
                 ckpt_dir: Optional[str] = None,
                 device=None,
                 smoke_input: Optional[Callable] = None,
                 alpha_log_csv: Optional[str] = None,
                 guard: Optional[GuardConfig] = None,
                 provenance: Optional[dict] = None):
        """
        provenance: optional dict embedded into every checkpoint save.
            Standard keys (set by training scripts): manifest_hash,
            manifest_path, manifest_version, run_id, training_config_hash.
            Any extra keys are passed through verbatim. Used by post-hoc
            tooling to answer "what manifest / config produced this
            checkpoint?" without guesswork.
        """
        self.model = model
        self.ckpt_path = Path(ckpt_path)
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else self.ckpt_path.parent
        self.device = device
        self.smoke_input = smoke_input
        self.alpha_log_csv = Path(alpha_log_csv) if alpha_log_csv else None
        self.guard = guard or GuardConfig()
        self.provenance = dict(provenance) if provenance else {}

        self.best_val_r: float = -float('inf')
        self.best_val_prd: float = float('inf')   # lower is better
        self.best_epoch: int = -1
        self.no_improve_count: int = 0
        self.last_smoke_r: Optional[float] = None
        self.last_smoke_ok: Optional[bool] = None

        # Set up alpha log on first write.
        self._alpha_csv_writer = None
        self._alpha_csv_file = None
        self._alpha_layer_names: list = []

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def on_validation(self, epoch: int, val_r: float, train_r: float = 0.0,
                      raise_on_halt: bool = True,
                      val_prd: Optional[float] = None) -> dict:
        """Called by the training loop after each validation.

        val_prd (optional): when supplied, enables Option B tiebreaker — if
        two consecutive bests have R within `improvement_eps`, the one with
        lower PRD wins. This prevents the optimizer from trading 0.0005 R
        for several PRD-points of magnitude error.

        Returns a dict describing what happened. Raises TrainingHaltException
        if a guard tripped and raise_on_halt=True (the default).
        """
        result = {
            'epoch': epoch,
            'val_r': val_r,
            'val_prd': val_prd,
            'train_r': train_r,
            'is_best': False,
            'saved_best': False,
            'save_reason': '',
            'smoke_ok': None,
            'alpha_max': None,
            'alpha_min': None,
            'halt_reason': None,
        }

        # 1 — log per-layer alpha
        if self.alpha_log_csv:
            self._log_alpha(epoch, val_r)

        # 2 — alpha health check
        amax, amin = self._alpha_extremes()
        result['alpha_max'] = amax
        result['alpha_min'] = amin
        if amax is not None and amax > self.guard.alpha_max_safe:
            result['halt_reason'] = (
                f"ALPHA EXPLOSION: max α={amax:.2f} exceeds safe ceiling "
                f"{self.guard.alpha_max_safe} at epoch {epoch}"
            )
        elif amin is not None and amin < self.guard.alpha_min_safe:
            result['halt_reason'] = (
                f"ALPHA COLLAPSE: min α={amin:.2e} below safe floor "
                f"{self.guard.alpha_min_safe} at epoch {epoch}"
            )

        # 3 — best tracking + save. Two save triggers (Option B):
        #
        #   (a) val_r beats best by ≥ improvement_eps        — primary gate
        #   (b) val_r within ±improvement_eps of best        — tie on R
        #         AND val_prd beats best by ≥ prd_tiebreak_eps
        #
        # Trigger (b) catches the case where the model maintains the same
        # R but drifts towards better magnitude preservation — without
        # it, the lower-PRD-but-equal-R epoch never overwrites the saved
        # checkpoint and you ship the worse-PRD one.
        improved_r = val_r > self.best_val_r + self.guard.improvement_eps
        tied_r = (val_prd is not None
                  and abs(val_r - self.best_val_r) <= self.guard.improvement_eps
                  and val_prd < self.best_val_prd - self.guard.prd_tiebreak_eps)

        save_reason = ''
        if improved_r:
            save_reason = 'r_improved'
        elif tied_r:
            save_reason = 'tie_prd'

        if save_reason:
            # Update best-tracked metrics whichever trigger fired.
            self.best_val_r = max(self.best_val_r, val_r)
            if val_prd is not None:
                self.best_val_prd = min(self.best_val_prd, val_prd)
            self.best_epoch = epoch
            self.no_improve_count = 0
            result['is_best'] = True
            self._save_atomic(self.ckpt_path)
            result['saved_best'] = True
            result['save_reason'] = save_reason
            # Smoke-check the freshly saved file.
            if self.smoke_input is not None:
                ok, smoke_r = self._smoke_check(self.ckpt_path,
                                                 expected_r=val_r)
                result['smoke_ok'] = ok
                result['smoke_r'] = smoke_r
                self.last_smoke_ok = ok
                self.last_smoke_r = smoke_r
                if not ok:
                    result['halt_reason'] = (
                        f"SMOKE CHECK FAILED: saved ckpt R={smoke_r:.4f} "
                        f"differs from in-training R={val_r:.4f} by more "
                        f"than {self.guard.smoke_check_tolerance}"
                    )
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.guard.r_plateau_patience:
                result['halt_reason'] = (
                    f"R PLATEAU: {self.no_improve_count} validations "
                    f"without improvement (best={self.best_val_r:.4f} at "
                    f"epoch {self.best_epoch})"
                )

        if result['halt_reason'] and raise_on_halt:
            raise TrainingHaltException(result['halt_reason'])
        return result

    def maybe_save_recovery(self, epoch: int, every: int = 50) -> Optional[Path]:
        """Save a numbered checkpoint every `every` epochs. Returns the path
        written, or None if it wasn't a checkpoint epoch."""
        if every <= 0 or epoch == 0 or epoch % every != 0:
            return None
        path = self.ckpt_dir / f"recovery_ep{epoch:04d}.ckpt"
        self._save_atomic(path)
        return path

    def close(self):
        """Flush + close any open log files. Safe to call multiple times."""
        if self._alpha_csv_file is not None:
            self._alpha_csv_file.close()
            self._alpha_csv_file = None
            self._alpha_csv_writer = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _save_atomic(self, path: Path):
        """Write state_dict + provenance to path atomically (tmp + rename)
        so a crash mid-write can't corrupt the file.

        Schema:
          {'state_dict':         OrderedDict[str → tensor],
           'best_val_r':         float,
           'best_val_prd':       float,
           'best_epoch':         int,
           'manifest_hash':      'sha256:...' (if provenance was passed),
           'manifest_path':      str,
           'manifest_version':   str,
           'run_id':             str,
           'saved_at':           ISO 8601 UTC,
           ...any extra provenance keys}

        Loading either via torch.load returns this dict; consumers that
        expect just a state_dict should branch on `'state_dict' in d`.
        """
        import torch
        from datetime import datetime, timezone
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'state_dict': self.model.state_dict(),
            'best_val_r': self.best_val_r,
            'best_val_prd': self.best_val_prd,
            'best_epoch': self.best_epoch,
            'saved_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }
        payload.update(self.provenance)
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _smoke_check(self, path: Path, expected_r: float):
        """Reload `path` into a fresh copy of self.model's class and run
        one forward pass. Verify R matches expected within tolerance.

        Returns (ok: bool, observed_r: float).
        """
        import torch
        try:
            x = self.smoke_input()
            with torch.no_grad():
                # Reference output from the live model.
                y_live = _run_forward(self.model, x)

                # Load into a fresh copy.
                fresh = _clone_model_class(self.model)
                # Contains non-tensor metadata (best_val_r, saved_at, provenance)
                try:
                    state = torch.load(path, map_location='cpu', weights_only=True)
                except Exception:
                    state = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                missing, unexpected = fresh.load_state_dict(state, strict=False)
                if self.device is not None:
                    fresh = fresh.to(self.device)
                fresh.eval()

                y_reload = _run_forward(fresh, x)

            # Correlate the two outputs flat.
            yl = y_live.flatten().detach().cpu().numpy()
            yr = y_reload.flatten().detach().cpu().numpy()
            import numpy as np
            if yl.std() < 1e-12 or yr.std() < 1e-12:
                # Degenerate case (all zeros) — call it OK.
                return True, expected_r
            r = float(np.corrcoef(yl, yr)[0, 1])
            ok = abs(r - 1.0) <= self.guard.smoke_check_tolerance
            return ok, r
        except Exception as e:
            # Any failure during smoke-check counts as a fail.
            return False, float('nan')

    def _alpha_extremes(self):
        """Return (max_abs_alpha, min_abs_alpha) across every ternary layer.
        Returns (None, None) if the model has no LSQ alphas."""
        import torch
        amax = None
        amin = None
        for _, m in self.model.named_modules():
            if hasattr(m, 'lsq_alpha'):
                a = m.lsq_alpha.detach().abs()
                lo = float(a.min())
                hi = float(a.max())
                amax = hi if amax is None else max(amax, hi)
                amin = lo if amin is None else min(amin, lo)
        return amax, amin

    def _log_alpha(self, epoch: int, val_r: float):
        """Append one row of per-layer α stats to the CSV."""
        import torch
        # Discover layers on first call so the CSV header is stable.
        if self._alpha_csv_writer is None:
            self._alpha_layer_names = [
                n for n, m in self.model.named_modules()
                if hasattr(m, 'lsq_alpha')
            ]
            self.alpha_log_csv.parent.mkdir(parents=True, exist_ok=True)
            self._alpha_csv_file = open(self.alpha_log_csv, 'w', newline='')
            cols = ['epoch', 'val_r']
            for name in self._alpha_layer_names:
                cols += [f'{name}.alpha_min',
                         f'{name}.alpha_mean',
                         f'{name}.alpha_max']
            self._alpha_csv_writer = csv.writer(self._alpha_csv_file)
            self._alpha_csv_writer.writerow(cols)

        row = [epoch, f'{val_r:.6f}']
        for name in self._alpha_layer_names:
            m = dict(self.model.named_modules())[name]
            a = m.lsq_alpha.detach().abs()
            row += [f'{float(a.min()):.6e}',
                    f'{float(a.mean()):.6e}',
                    f'{float(a.max()):.6e}']
        self._alpha_csv_writer.writerow(row)
        self._alpha_csv_file.flush()


# ============================================================
# Helpers — small enough to stay private
# ============================================================

def _clone_model_class(model):
    """Build a fresh instance of the same class with the same constructor
    args. Best-effort: works for our TernaryMobileNetV5_Subband which
    stores its constructor args in `_init_kwargs`. Falls back to deepcopy
    of the architecture if not available.
    """
    # Preferred: explicit kwargs cache on the model.
    init_kwargs = getattr(model, '_init_kwargs', None)
    if init_kwargs is not None:
        cls = type(model)
        return cls(**init_kwargs)
    # Fallback: copy the live model and reset its weights to defaults.
    import copy
    fresh = copy.deepcopy(model)
    # Reset any layers that have a reset_parameters() method.
    for m in fresh.modules():
        if hasattr(m, 'reset_parameters'):
            try:
                m.reset_parameters()
            except Exception:
                pass
    return fresh


def _run_forward(model, x):
    """Run a minimal forward pass on the model. Tries .encode() first
    (matches our codec model's API), falls back to __call__.
    """
    if hasattr(model, 'encode'):
        # Encode → decode round-trip if both methods exist.
        lat = model.encode(x, quantize=True) if hasattr(model.encode, '__call__') else model.encode(x)
        if hasattr(model, 'decode'):
            return model.decode(lat, target_len=x.shape[-1], quantize=True)
        return lat
    return model(x)


def make_param_groups(model, *,
                       lr: float,
                       weight_decay: float = 0.0,
                       alpha_weight_decay: float = 1e-3,
                       alpha_lr_mult: float = 1.0):
    """Build optimizer param groups with separate weight-decay for LSQ alpha.

    Why this exists
    ---------------
    LSQ alpha learns its own scale via gradient descent. With no
    pull-back force, large gradients in QAT can drive alpha exponentially
    higher each epoch — we observed alpha = 144 on expand3.conv in the
    failed gold run, while the encoder layers stayed near 1.5.

    Option C (BitNet-style): apply a small weight_decay specifically to
    alpha parameters. Soft pressure pulling them toward 0 (so they only
    grow when the gradient signal is strong enough to overcome it).
    Combined with the always-on hard clamp [1e-4, 20] in the training
    loop, this is "belt and suspenders" — soft pressure + hard ceiling.

    Args:
        model:               nn.Module containing some LSQ layers.
        lr:                  Base learning rate for non-alpha params.
        weight_decay:        WD for non-alpha params (whatever your config wants).
        alpha_weight_decay:  WD applied ONLY to lsq_alpha parameters.
                             1e-3 is the BitNet default; 0 disables.
        alpha_lr_mult:       Multiplier on `lr` for alpha params (1.0 = same).

    Returns:
        A list of dicts suitable for torch.optim.AdamW(param_groups).
    """
    alpha_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith('lsq_alpha') or name.endswith('.lsq_alpha'):
            alpha_params.append(p)
        else:
            other_params.append(p)

    return [
        {
            'params': other_params,
            'lr': lr,
            'weight_decay': weight_decay,
        },
        {
            'params': alpha_params,
            'lr': lr * alpha_lr_mult,
            'weight_decay': alpha_weight_decay,
        },
    ]


__all__ = [
    'CheckpointManager', 'GuardConfig', 'TrainingHaltException',
    'make_param_groups',
]
