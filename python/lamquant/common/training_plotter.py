"""
LamQuant Training Plotter — 4-panel diagnostic figure saved every N epochs.

Top left:     R curves (train R, val R) vs epoch
Top right:    Sparsity (global + per-layer) vs epoch
Bottom left:  Alpha values (per-layer mean) vs epoch
Bottom right: Loss components (total, MSE, R, spectral, L1) vs epoch

Usage:
    from training_plotter import TrainingPlotter

    plotter = TrainingPlotter(output_dir='outputs')

    # After each epoch:
    plotter.log_epoch(
        epoch=147,
        train_r=0.81,
        val_r=0.73,
        global_sparsity=12.3,
        per_layer_sparsity={'focal2.conv': 8.1, 'focal3.conv': 10.4, ...},
        per_layer_alpha={'focal2.conv': 0.34, 'focal3.conv': 0.31, ...},
        loss_total=0.042,
        loss_mse=0.031,
        loss_r=0.008,
        loss_spectral=0.003,
        loss_l1=0.0001,
    )

    # Saves outputs/training_curves.png every plot_interval epochs
"""

import os
import numpy as np


class TrainingPlotter:
    """Accumulates training metrics and saves a 4-panel diagnostic figure."""

    # Color scheme for per-layer plots
    LAYER_COLORS = {
        'premix': '#e6194b',
        'focal1_conv': '#3cb44b',
        'focal2.conv': '#4363d8',
        'focal3.conv': '#f58231',
        'focal3.shortcut': '#f58231',
        'dw_gate': '#911eb4',
        'bneck_v': '#42d4f4',
        'bneck_g': '#f032e6',
    }

    def __init__(self, output_dir='outputs', plot_interval=25):
        self.output_dir = output_dir
        self.plot_interval = plot_interval
        os.makedirs(output_dir, exist_ok=True)

        # Accumulated data
        self.epochs = []
        self.train_r = []
        self.val_r = []
        self.global_sparsity = []
        self.per_layer_sparsity = {}  # {layer_name: [values]}
        self.per_layer_alpha = {}     # {layer_name: [values]}
        self.loss_total = []
        self.loss_mse = []
        self.loss_r = []
        self.loss_spectral = []
        self.loss_l1 = []
        self.phase_boundaries = []    # [(epoch, label)]

    def mark_phase(self, epoch, label):
        """Mark a phase boundary (e.g., 'QAT', 'Fine')."""
        self.phase_boundaries.append((epoch, label))

    def log_epoch(self, epoch, train_r=0, val_r=0, global_sparsity=0,
                  per_layer_sparsity=None, per_layer_alpha=None,
                  loss_total=0, loss_mse=0, loss_r=0, loss_spectral=0, loss_l1=0):
        """Log metrics for one epoch. Saves plot every plot_interval epochs."""
        self.epochs.append(epoch)
        self.train_r.append(train_r)
        self.val_r.append(val_r)
        self.global_sparsity.append(global_sparsity)
        self.loss_total.append(loss_total)
        self.loss_mse.append(loss_mse)
        self.loss_r.append(loss_r)
        self.loss_spectral.append(loss_spectral)
        self.loss_l1.append(loss_l1)

        if per_layer_sparsity:
            for name, val in per_layer_sparsity.items():
                self.per_layer_sparsity.setdefault(name, []).append(val)
        if per_layer_alpha:
            for name, val in per_layer_alpha.items():
                self.per_layer_alpha.setdefault(name, []).append(val)

        if epoch % self.plot_interval == 0 and len(self.epochs) > 1:
            self.save_plot()

    def save_plot(self):
        """Render and save the 4-panel figure. Non-blocking, no display."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            return  # matplotlib not available, skip silently

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('LamQuant Training Diagnostics', fontsize=14, fontweight='bold')
        ep = np.array(self.epochs)

        # ── Top left: R curves ──
        ax = axes[0, 0]
        ax.plot(ep, self.train_r, 'b-', label='Train R', linewidth=1.5)
        if any(v > 0 for v in self.val_r):
            ax.plot(ep, self.val_r, 'r-', label='Val R (quantized)', linewidth=1.5)
            # Shade quantization gap
            tr = np.array(self.train_r)
            vr = np.array(self.val_r)
            mask = vr > 0.01  # only where val R is meaningful
            if mask.any():
                ax.fill_between(ep[mask], tr[mask], vr[mask],
                               alpha=0.15, color='red', label='Quant gap')
        for e, label in self.phase_boundaries:
            ax.axvline(x=e, color='gray', linestyle='--', alpha=0.5)
            ax.text(e, ax.get_ylim()[1], label, fontsize=8, ha='center', va='bottom')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pearson R')
        ax.set_title('Reconstruction Quality')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # ── Top right: Sparsity ──
        ax = axes[0, 1]
        ax.plot(ep, self.global_sparsity, 'k-', label='Global', linewidth=2)
        # Per-layer (only layers with enough data points)
        for name, vals in sorted(self.per_layer_sparsity.items()):
            if len(vals) < 2:
                continue
            color = self.LAYER_COLORS.get(name, '#999999')
            # Align to epochs (per-layer logged at validation intervals)
            layer_ep = ep[-len(vals):]
            short_name = name.split('.')[-1] if '.' in name else name
            ax.plot(layer_ep, vals, color=color, alpha=0.7, linewidth=1,
                    label=short_name)
        ax.axhline(y=55, color='green', linestyle=':', alpha=0.5, label='Target 55%')
        ax.axhline(y=30, color='orange', linestyle=':', alpha=0.5, label='Min 30%')
        for e, label in self.phase_boundaries:
            ax.axvline(x=e, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sparsity %')
        ax.set_title('Ternary Sparsity (% zero weights)')
        ax.legend(fontsize=7, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 80)

        # ── Bottom left: Alpha values ──
        ax = axes[1, 0]
        for name, vals in sorted(self.per_layer_alpha.items()):
            if len(vals) < 2:
                continue
            color = self.LAYER_COLORS.get(name, '#999999')
            layer_ep = ep[-len(vals):]
            short_name = name.split('.')[-1] if '.' in name else name
            ax.plot(layer_ep, vals, color=color, linewidth=1, label=short_name)
        for e, label in self.phase_boundaries:
            ax.axvline(x=e, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean α')
        ax.set_title('LSQ Alpha per Layer')
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)

        # ── Bottom right: Loss components ──
        ax = axes[1, 1]
        if any(v > 0 for v in self.loss_total):
            ax.plot(ep, self.loss_total, 'k-', label='Total', linewidth=1.5)
        if any(v > 0 for v in self.loss_mse):
            ax.plot(ep, self.loss_mse, 'b-', label='MSE', alpha=0.7)
        if any(v > 0 for v in self.loss_r):
            ax.plot(ep, self.loss_r, 'r-', label='1-R', alpha=0.7)
        if any(v > 0 for v in self.loss_spectral):
            ax.plot(ep, self.loss_spectral, 'g-', label='Spectral', alpha=0.7)
        if any(v > 0 for v in self.loss_l1):
            ax.plot(ep, self.loss_l1, 'm-', label='L1 reg', alpha=0.7)
        for e, label in self.phase_boundaries:
            ax.axvline(x=e, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'training_curves.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
