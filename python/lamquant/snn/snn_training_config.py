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
    # A2 (run-2 2026-05-29): lambda_spike default 0.0 — the spike penalty is
    # now a base-rate target (mamba_ssm_minimal.SPIKE_TARGET_RATE) rather than
    # the old wrong-sign L1 pull, but it is held off by default until the
    # seizure objective is the sole driver. Set >0 to re-enable.
    lambda_spike: float = 0.0   # spike rate regularization (run-2: off)
    pos_weight: float = 3.0     # DWB base positive class weight
    # A4 (run-2): drop the *3.0 logit gradient amplifier that fed the SSM
    # divergence. 1.0 = use raw logits.
    logit_scale: float = 1.0

    # --- Architecture (must fit ≤64 KB INT8) ---
    d_model: int = 40
    d_state: int = 16
    n_layers: int = 2

    # --- Dataset ---
    max_windows_per_file: int = 5   # L3 windows per q31 file (seizure-aware)

    # --- Run-2 stability / schedule / sampler defaults ---
    # B3 seizure-balanced curriculum (default ON): target seizure-window
    # fraction per batch, annealed from `seizure_batch_frac` toward the
    # natural rate over `seizure_frac_anneal_epochs`.
    seizure_batch_frac: float = 0.5
    seizure_frac_anneal_epochs: int = 20
    # A7 active NaN-guard / clip; A8 sens-floored selection; A9 early-stop.
    grad_clip: float = 0.5
    sens_floor: float = 0.85
    abort_on_collapse: bool = True
    early_stop_patience: int = 30
    warmup_frac: float = 0.10
    # A1 no-WD param group for SSM dynamics (A_log/dt_bias/D/bias/norm).
    no_wd_dynamics: bool = True
    # Natural seizure-window fraction the curriculum anneals toward (the
    # observed train rate; the sampler stops oversampling once reached).
    seizure_frac_natural: float = 0.18

    # --- Seizure-head loss (B4 + B5) ---
    # Dedicated seizure-head pos_weight floor. Overridden at runtime by the
    # data-derived seizure-vs-rest ratio (~40) unless that scan is skipped.
    seizure_pos_weight: float = 40.0
    # B5 focal + soft-Tversky knobs for the seizure channel.
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    tversky_fn_weight: float = 0.7      # FN penalty (recall-favoring)
    tversky_fp_weight: float = 0.3
    seizure_loss_weight: float = 1.5    # >1.0x so seizure can't be drowned

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
        description='Prototyping — ~1 hour. Sanity checks and architecture experiments. '
                    'Run-2 stability defaults (gentle 3e-4 peak, 10% warmup).',
        epochs=50,
        batch_size=128,
        lr=3e-4,
        lr_min=1e-5,
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
    # run-10 (2026-05-31): SPECIFICITY-rebalanced production. Root cause of the
    # flooding (time_spec ~0.35 at sens=1.0) is the over-prediction PRIOR: the
    # sampler hyper-exposes the seizure head to a 50%→18% seizure fraction when
    # true seizure is ~1% of time, and the loss (tversky fn>>fp, focal) rewards
    # any positive. ROC diagnostic on run-9: max time_spec @ sens≥0.99 = 0.405
    # => must RETRAIN for discrimination, not re-threshold. This preset stops
    # teaching "seizures are everywhere" (sampler → ~natural rate) and penalises
    # false-positive TIME (tversky_fp up, fn down). Pair with --optimizer esoap
    # + the base-rate seizure-head bias init (mamba_ssm_minimal). sens has huge
    # margin (1.0) so trading a little recall for specificity is safe.
    'production_spec': SNNConfig(
        name='production_spec',
        description='run-10 — specificity-rebalanced: sampler→natural prior + '
                    'tversky penalises FP-time. Target sens~1.0 & time_spec≥0.90.',
        epochs=250,
        batch_size=128,
        lr=1e-3,
        lr_min=1e-5,
        max_windows_per_file=5,
        early_stop_patience=40,
        # --- rebalance: stop the over-prediction prior (ranks 1-3) ---
        seizure_batch_frac=0.05,        # was 0.5 — the dominant lever
        seizure_frac_natural=0.02,      # was 0.18 — anneal target near true rate
        seizure_frac_anneal_epochs=8,   # was 20 — don't bake in the early flood
        # --- loss: penalise false-positive TIME (ranks 4-6) ---
        tversky_fp_weight=0.6,          # was 0.3
        tversky_fn_weight=0.5,          # was 0.7 (recall-favoring) → balanced
        seizure_loss_weight=1.0,        # was 1.5 — don't let seizure head monopolise
        # --- focal: damp the positive over-emphasis (ranks 8-9) ---
        focal_alpha=0.55,               # was 0.75
        focal_gamma=1.5,                # was 2.0
    ),
}
