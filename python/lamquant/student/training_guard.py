"""Training guard: automated alarm system for catching problems mid-training.

Runs inline during training. Checks are cheap (~1ms each) and fire every
N epochs. If a check fails, it logs a WARNING but does NOT stop training —
the human decides whether to intervene.

Usage in training loop:
    guard = TrainingGuard(model, config='v2')

    for epoch in range(total_epochs):
        train_one_epoch(...)
        val_r = validate(...)

        # Run all guards — returns list of warnings (empty = all OK)
        warnings = guard.check(epoch, val_r=val_r, train_loss=loss)
        for w in warnings:
            print(f"  WARNING: {w}")

Guards:
    1. R plateau:      val_r hasn't improved in N epochs
    2. R collapse:     val_r dropped by >X from best
    3. Loss explosion: train_loss > 10x initial loss
    4. Dead layers:    >75% ternary sparsity in any layer
    5. Alpha explosion: any lsq_alpha > ceiling
    6. Gradient vanish: max grad norm < threshold
    7. Gradient explode: max grad norm > threshold
    8. NaN/Inf:        any NaN or Inf in parameters
    9. Latent collapse: latent std < threshold (mode collapse)
    10. DW-sep imbalance: DW gradients >> PW gradients (V2-specific)
"""

import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GuardConfig:
    """Thresholds for training guards."""
    # R monitoring
    r_plateau_patience: int = 30          # epochs without improvement
    r_collapse_threshold: float = 0.15    # max drop from best R
    r_minimum: float = 0.05              # absolute minimum (model is broken)

    # Loss
    loss_explosion_factor: float = 10.0   # max ratio vs initial loss

    # Weights
    dead_layer_sparsity: float = 0.75     # ternary zero fraction
    alpha_ceiling: float = 5.0            # absolute max alpha
    alpha_floor: float = 0.001            # absolute min alpha (collapsed)

    # Gradients
    grad_vanish_threshold: float = 1e-7   # max grad norm below this = vanished
    grad_explode_threshold: float = 100.0 # max grad norm above this = exploded

    # Latent
    latent_std_minimum: float = 0.01      # latent std below this = mode collapse

    # Check frequency
    check_every: int = 5                  # run guards every N epochs


# Presets
GUARD_PRESETS = {
    'v1': GuardConfig(r_plateau_patience=30, alpha_ceiling=3.0),
    'v2': GuardConfig(r_plateau_patience=25, alpha_ceiling=5.0),  # V2 may need wider alpha
    'fast': GuardConfig(r_plateau_patience=10, check_every=2),
}


class TrainingGuard:
    """Automated alarm system for training problems."""

    def __init__(self, model, config='v2'):
        if isinstance(config, str):
            self.cfg = GUARD_PRESETS.get(config, GUARD_PRESETS['v2'])
        else:
            self.cfg = config
        self.model = model
        self.best_r = 0.0
        self.best_r_epoch = 0
        self.initial_loss = None
        self.history = []  # (epoch, val_r, train_loss)
        self.total_warnings = 0

    def check(self, epoch: int, val_r: float = None, train_loss: float = None,
              latent: torch.Tensor = None) -> list:
        """Run all guards. Returns list of warning strings (empty = OK)."""
        if epoch % self.cfg.check_every != 0 and epoch > 0:
            return []

        warnings = []

        # Track history
        if val_r is not None:
            self.history.append((epoch, val_r, train_loss))
            if val_r > self.best_r:
                self.best_r = val_r
                self.best_r_epoch = epoch

        if train_loss is not None and self.initial_loss is None:
            self.initial_loss = train_loss

        # 1. R plateau
        if val_r is not None and epoch - self.best_r_epoch > self.cfg.r_plateau_patience:
            warnings.append(
                f"R PLATEAU: no improvement for {epoch - self.best_r_epoch} epochs "
                f"(best={self.best_r:.4f} at ep {self.best_r_epoch})")

        # 2. R collapse
        if val_r is not None and self.best_r > 0.1:
            drop = self.best_r - val_r
            if drop > self.cfg.r_collapse_threshold:
                warnings.append(
                    f"R COLLAPSE: dropped {drop:.4f} from best {self.best_r:.4f} → {val_r:.4f}")

        # 3. R minimum
        if val_r is not None and val_r < self.cfg.r_minimum and epoch > 20:
            warnings.append(f"R BROKEN: {val_r:.4f} < {self.cfg.r_minimum} after {epoch} epochs")

        # 4. Loss explosion
        if train_loss is not None and self.initial_loss is not None:
            ratio = train_loss / max(self.initial_loss, 1e-8)
            if ratio > self.cfg.loss_explosion_factor:
                warnings.append(
                    f"LOSS EXPLOSION: {train_loss:.6f} = {ratio:.1f}x initial {self.initial_loss:.6f}")

        # 5. Dead layers
        for name, m in self.model.named_modules():
            if hasattr(m, 'lsq_alpha') and hasattr(m, 'weight') and m.weight.dim() >= 2:
                with torch.no_grad():
                    w = m.weight.data
                    alpha = m.lsq_alpha.data
                    zero_frac = (w.abs() <= alpha.abs()).float().mean().item()
                    if zero_frac > self.cfg.dead_layer_sparsity:
                        warnings.append(
                            f"DEAD LAYER: {name} is {zero_frac:.0%} zero (threshold {self.cfg.dead_layer_sparsity:.0%})")

        # 6. Alpha explosion/collapse
        for name, m in self.model.named_modules():
            if hasattr(m, 'lsq_alpha'):
                alpha_max = m.lsq_alpha.data.abs().max().item()
                alpha_min = m.lsq_alpha.data.abs().min().item()
                if alpha_max > self.cfg.alpha_ceiling:
                    warnings.append(
                        f"ALPHA EXPLOSION: {name} alpha_max={alpha_max:.3f} > {self.cfg.alpha_ceiling}")
                if alpha_min < self.cfg.alpha_floor:
                    warnings.append(
                        f"ALPHA COLLAPSE: {name} alpha_min={alpha_min:.6f} < {self.cfg.alpha_floor}")

        # 7-8. Gradient vanish/explode + NaN
        max_grad = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                gn = p.grad.data.norm().item()
                max_grad = max(max_grad, gn)
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    warnings.append(f"NaN/Inf GRADIENT: {name}")
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                warnings.append(f"NaN/Inf WEIGHT: {name}")

        if max_grad > 0 and max_grad < self.cfg.grad_vanish_threshold:
            warnings.append(f"GRADIENT VANISH: max_grad_norm={max_grad:.2e}")
        if max_grad > self.cfg.grad_explode_threshold:
            warnings.append(f"GRADIENT EXPLODE: max_grad_norm={max_grad:.2e}")

        # 9. Latent collapse
        if latent is not None:
            lat_std = latent.std().item()
            if lat_std < self.cfg.latent_std_minimum:
                warnings.append(f"LATENT COLLAPSE: std={lat_std:.6f} (mode collapse?)")

        # 10. DW-sep imbalance (V2-specific)
        dw_grads = []
        pw_grads = []
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                if '.dw.' in name and 'weight' in name:
                    dw_grads.append(p.grad.data.norm().item())
                elif '.pw.' in name and 'weight' in name:
                    pw_grads.append(p.grad.data.norm().item())
        if dw_grads and pw_grads:
            dw_mean = np.mean(dw_grads)
            pw_mean = np.mean(pw_grads)
            if pw_mean > 0 and dw_mean / pw_mean > 50:
                warnings.append(
                    f"DW-SEP IMBALANCE: DW grads {dw_mean:.4f} >> PW grads {pw_mean:.4f} "
                    f"(ratio {dw_mean/pw_mean:.0f}x, may need separate LR)")

        self.total_warnings += len(warnings)
        return warnings

    def summary(self) -> str:
        """End-of-training summary."""
        lines = [
            f"Training Guard Summary:",
            f"  Total warnings: {self.total_warnings}",
            f"  Best R: {self.best_r:.4f} at epoch {self.best_r_epoch}",
            f"  Epochs tracked: {len(self.history)}",
        ]
        if self.history:
            last_r = self.history[-1][1]
            if last_r is not None:
                lines.append(f"  Final R: {last_r:.4f} (delta from best: {self.best_r - last_r:.4f})")
        return "\n".join(lines)
