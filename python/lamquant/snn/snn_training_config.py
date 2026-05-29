"""
Training configuration for LamQuant Mamba SNN activity detector.

Three presets: fast (prototyping), standard (iteration), production (shipping).
Follows the same pattern as ai_models/student/training_config.py.

Usage:
    from snn_training_config import SNNConfig, SNN_CONFIGS

    cfg = SNN_CONFIGS['production']
    print(cfg)

    # Override a single field
    cfg = SNN_CONFIGS['standard'].replace(batch_size=256)
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SNNConfig:
    """Complete SNN training configuration. Immutable after creation."""

    # --- Identity ---
    name: str = 'custom'
    description: str = ''

    # --- Training schedule ---
    epochs: int = 500
    batch_size: int = 128
    lr: float = 1e-3
    lr_min: float = 1e-5        # cosine annealing floor
    weight_decay: float = 1e-4

    # --- Loss ---
    lambda_spike: float = 0.01  # spike rate regularization
    pos_weight: float = 3.0     # DWB base positive class weight

    # --- Architecture (must fit ≤64 KB INT8) ---
    d_model: int = 40
    d_state: int = 16
    n_layers: int = 2

    # --- Dataset ---
    max_windows_per_file: int = 5   # L3 windows per q31 file (seizure-aware)

    @property
    def param_estimate(self) -> int:
        """Rough parameter count estimate (exact depends on expand factor)."""
        d_inner = self.d_model * 2
        ssm_per = (self.d_model * d_inner * 2   # in_proj
                   + d_inner * 4 + d_inner       # conv1d
                   + d_inner * (self.d_state * 2 + 1)  # x_proj
                   + d_inner * self.d_state      # A_log
                   + d_inner                     # D
                   + d_inner * self.d_model       # out_proj
                   + d_inner)                    # dt_bias
        bidir_per = ssm_per * 2 + self.d_model * 2  # fwd+bwd+layernorm
        total = (21 * self.d_model + self.d_model    # spatial_mix
                 + bidir_per * self.n_layers          # SSM blocks
                 + self.d_model * 8 + 8)              # readout
        return total

    def replace(self, **kwargs) -> 'SNNConfig':
        """Create a new config with specific fields overridden."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def __str__(self):
        kb = self.param_estimate * 8 / 8 / 1024
        return (
            f"SNNConfig('{self.name}')\n"
            f"  Epochs: {self.epochs}  |  Batch: {self.batch_size}\n"
            f"  LR:     {self.lr:.0e} → {self.lr_min:.0e} (cosine)\n"
            f"  Arch:   d_model={self.d_model}, d_state={self.d_state}, "
            f"n_layers={self.n_layers} (~{kb:.1f} KB INT8)\n"
            f"  Data:   {self.max_windows_per_file} windows/file (seizure-aware)\n"
            f"  {self.description}"
        )


# ============================================================
# Preset Configurations
# ============================================================

SNN_CONFIGS = {
    'fast': SNNConfig(
        name='fast',
        description='Prototyping — ~1 hour. Sanity checks and architecture experiments.',
        epochs=50,
        batch_size=128,
        lr=2e-3,
        lr_min=1e-4,
        max_windows_per_file=2,
    ),

    'standard': SNNConfig(
        name='standard',
        description='Balanced — ~5 hours. Good seizure sensitivity, reasonable iteration.',
        epochs=200,
        batch_size=128,
        lr=1e-3,
        lr_min=1e-5,
        max_windows_per_file=5,
    ),

    'production': SNNConfig(
        name='production',
        description='Production — 400 epochs, cosine warmup → WSD stable → cosine decay '
                    '(W3 2026-05-21). Match joint trainer schedule. '
                    '~6 h wall at 50 s/epoch warm L3 cache.',
        epochs=400,
        batch_size=128,
        lr=1e-3,
        lr_min=1e-5,
        max_windows_per_file=5,
    ),
}
