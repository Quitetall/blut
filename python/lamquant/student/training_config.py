"""
Training configuration dataclasses for LamQuant student models.

Three presets: fast (prototyping), standard (iteration), production (shipping).
All hyperparameters in one place — no magic numbers in training loops.

Usage:
    from training_config import TrainingConfig, CONFIGS

    cfg = CONFIGS['production']
    print(cfg)
    print(cfg.total_epochs)

    # Override a single field
    cfg = CONFIGS['standard'].replace(batch_size=64)

    # Load from YAML (optional)
    cfg = TrainingConfig.from_yaml('my_config.yaml')
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional
import os


@dataclass(frozen=True)
class TrainingConfig:
    """Complete training configuration. Immutable after creation."""

    # --- Identity ---
    name: str = 'custom'
    description: str = ''

    # --- Epoch schedule ---
    epochs_warmup: int = 50       # Phase 1: FP32 warm-up (no quantization)
    epochs_quant: int = 200       # Phase 2: QAT (ternary + INT16/INT8 activation)
    epochs_fine: int = 300        # Phase 3: Fine-tune (+ spectral loss, τ→0)

    # --- Batch size (AdaBatch: per-phase adaptive sizing) ---
    # Small batches early for exploration, large batches late for throughput.
    # Devarakonda et al. 2018: doubling bs ≈ halving LR for convergence.
    batch_size: int = 32              # base (used if per-phase not set)
    batch_size_warmup: int = 32       # Phase 1: small for loss landscape exploration
    batch_size_quant: int = 64        # Phase 2: moderate after QAT stabilizes
    batch_size_fine: int = 256        # Phase 3: large, smooth valley, maximize GPU

    # --- Learning rates (per phase, cosine-annealed) ---
    lr_warmup: float = 2e-3
    lr_warmup_min: float = 5e-4
    lr_quant: float = 1e-3
    lr_quant_min: float = 1e-5      # Anneals to fine-tune level (was 1e-4, Phase 3 not needed)
    lr_fine: float = 2e-4
    lr_fine_min: float = 1e-5

    # --- Weight decay ---
    wd_warmup: float = 1e-4
    wd_quant: float = 1e-4
    wd_fine: float = 1e-5
    wd_two_stage: bool = False         # BitNet recipe: remove WD in final 1/3 of QAT
                                       # Experiment result: +0.0007 R at 150ep

    # --- SNAC multi-scale FSQ ---
    snac_preset: str = 'none'          # 'none', 'compact', 'balanced', 'quality', 'flat'
                                       # Experiment result: compact +0.0009 R at 150ep

    # --- Gradient clipping ---
    grad_clip_warmup: float = 10.0
    grad_clip_quant: float = 10.0
    grad_clip_fine: float = 5.0

    # --- Loss weights ---
    pearson_r_weight: float = 0.5      # weight on (1-R) loss term
    spectral_weight: float = 0.1       # weight on multi-scale STFT loss (Phase 3 only)
    prd_weight: float = 0.1            # weight on PRD/100 loss (magnitude preservation;
                                       # MSE alone implicitly optimises PRD but adding
                                       # it explicitly stops the optimizer from trading
                                       # 0.0005 R for 3% worse PRD on long-tail amplitudes)

    # --- Spectral loss FFT sizes (tuned for L3 = 313 samples) ---
    spectral_fft_sizes: tuple = (16, 32, 64, 128, 256)

    # --- Channel dropout augmentation ---
    channel_dropout_min: int = 5       # min channels to zero per sample
    channel_dropout_max: int = 13      # max channels to zero per sample

    # --- Tequila deadzone ---
    deadzone_tau_initial: float = 0.1  # τ at start of Phase 2
    deadzone_tau_final: float = 0.0    # τ at end of Phase 3 (pure ternary)

    # --- Ternary sparsity control ---
    l1_lambda: float = 0.0             # L1 penalty on pre-quant weights (0=off, 1e-4=moderate)
    l1_lambda_decoder: float = 0.0     # Separate L1 for decoder (0=use l1_lambda)
    alpha_clamp: bool = False          # Clamp alpha to [0.5×std(W), 2×std(W)] per layer
    alpha_floor: float = 0.0           # Uniform absolute minimum alpha (0=off, 0.05=fallback)
    alpha_floor_init: bool = False     # Per-layer floor anchored to Phase 1 σ_W (0.8×σ_W_init)
    alpha_ceiling: float = 0.0         # Absolute alpha ceiling (0=off, 3.0=recommended)

    # --- Band-weighted MSE ---
    band_weighted_mse: bool = False    # Frequency-weighted MSE emphasizing delta/theta/alpha

    # --- Activation quantization ---
    activation_bits: int = 16          # 16 (production W2A16) or 8 (experimental W2A8)

    # --- Noise-aware training ---
    train_noise_bits: int = 0          # Right-shift input by N bits during training.
                                       # 0 = use full resolution. 6-7 = mask ADS1299
                                       # thermal noise floor. Data is NOT modified on
                                       # disk — masking is training-only. The model
                                       # learns to predict signal, not noise.
    noise_mask_mode: str = 'off'       # 'off' = full data, noise and all (default).
                                       # 'global' = use train_noise_bits for all windows.
                                       # 'per_dataset' = per-dataset from noise_profile.json.
                                       # 'per_window' = per-window estimated noise_bits.
    noise_profile_path: str = ''       # Path to noise_profile.json (for per_dataset/per_window)

    # --- Dataset ---
    windows_per_epoch: int = 400000    # samples drawn per epoch from PrecomputedL3Dataset
    max_windows: Optional[int] = None  # cap total windows loaded into RAM (None = all)

    # --- Validation ---
    val_interval: int = 10             # validate every N epochs
    val_windows: int = 5000            # windows per validation pass

    # --- Early stopping (added 2026-04-16 after the gold run wasted 360
    #     epochs without improvement) ---
    early_stop_patience: int = 60      # QAT validation intervals without
                                       # improvement before aborting.
                                       # 0 = disabled. Default 60 ≈ 600 epochs.

    # --- Architecture (refactor #74 — fields previously implicit in
    #     joint_codec.build_default_joint defaults). Capture these in
    #     the config so the saved checkpoint records the architecture
    #     it was trained for, not just the recipe. ---
    vocos_tier: int = 7                # decoder tier (1, 2, 3, 5, 6, 7, 8)
    latent_dim: int = 32               # FSQ latent channels
    encoder_width: int = 128           # encoder base width
    encoder_blocks: int = 3            # number of focal blocks
    encoder_kernels: str = '3,5,7'     # kernel sizes per block (comma-separated)
    encoder_arch: str = 'ternary_subband'  # architecture name (from registry)
    n_channels: int = 21               # EEG channels
    target_len: int = 313              # L3 length (ignored for iSTFT decoders)

    # --- Training-system / runtime (formerly CLI-only) ---
    data_dir: str = ''                 # Root for training data (manifest, memmaps).
                                       # '' = default (ai_models/dataset_sim/).
                                       # Set to e.g. '/mnt/4tb/data/training/' to
                                       # use data from an external directory.
    device: str = 'cuda'               # 'cuda' / 'cpu'
    precision: str = 'bf16'            # 'bf16' (autocast) / 'fp32'
    compile_decoder: bool = True       # torch.compile(decoder, ...)
    fullband_mode: str = 'auto'        # 'auto' / 'ram' / 'memmap' / 'off'
    asymmetric_weight: float = 0.0     # opt-in clinical-weighted loss
    asymmetric_kind: str = 'envelope'  # 'envelope' / 'band'
    seed: int = 0                      # RNG seed

    # --- Provenance bookkeeping ---
    config_version: str = '1.0.0'      # bumped when the dataclass schema changes

    @property
    def total_epochs(self) -> int:
        return self.epochs_warmup + self.epochs_quant + self.epochs_fine

    def replace(self, **kwargs) -> 'TrainingConfig':
        """Create a new config with specific fields overridden."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Export all fields as a dict (JSON / YAML / log-friendly)."""
        from dataclasses import asdict
        return asdict(self)

    # ------------------------------------------------------------
    # Round-trip + provenance (refactor #74)
    # ------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        """Reconstruct a TrainingConfig from a dict (typically loaded from
        a checkpoint's `training_config` payload).

        Tolerant of extra keys (logged + ignored) and missing keys (use
        defaults). This makes loading old checkpoints across schema
        versions a soft failure with a warning, not a hard crash.
        """
        from dataclasses import fields
        kw = dict(d)
        # spectral_fft_sizes round-trips through JSON as a list — coerce
        # back to tuple for hashability.
        if isinstance(kw.get('spectral_fft_sizes'), list):
            kw['spectral_fft_sizes'] = tuple(kw['spectral_fft_sizes'])
        known = {f.name for f in fields(cls)}
        extra = set(kw) - known
        if extra:
            print(f"[!] TrainingConfig.from_dict: ignoring unknown keys "
                  f"{sorted(extra)} (config_version mismatch?)")
            for k in extra:
                kw.pop(k)
        # Missing keys → defaults (dataclass handles this).
        return cls(**kw)

    def hash(self) -> str:
        """Deterministic content hash. Two configs that differ in any
        field produce different hashes; two configs equal in every
        field produce the same hash.

        Used as `training_config_hash` in checkpoint provenance so
        "what config produced this checkpoint" is verifiable, not
        guesswork.
        """
        import hashlib
        import json as _json
        d = self.to_dict()
        # spectral_fft_sizes is a tuple; JSON normalises to list.
        canonical = _json.dumps(d, sort_keys=True, separators=(',', ':'),
                                 default=list)
        return 'sha256:' + hashlib.sha256(canonical.encode()).hexdigest()

    def diff(self, other: 'TrainingConfig') -> dict:
        """Return {field_name: (self_value, other_value)} for fields that differ.

        Empty dict means the configs are identical (and `self.hash() ==
        other.hash()`). Useful for understanding why two checkpoints
        produced different results: `cfg_a.diff(cfg_b)` shows exactly
        what changed.
        """
        a = self.to_dict()
        b = other.to_dict()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from a YAML file. Unspecified fields use defaults."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml to use YAML configs")
        with open(path) as f:
            data = yaml.safe_load(f)
        # Convert spectral_fft_sizes from list to tuple if needed
        if 'spectral_fft_sizes' in data and isinstance(data['spectral_fft_sizes'], list):
            data['spectral_fft_sizes'] = tuple(data['spectral_fft_sizes'])
        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml to use YAML configs")
        d = self.to_dict()
        # Convert tuple to list for YAML
        d['spectral_fft_sizes'] = list(d['spectral_fft_sizes'])
        with open(path, 'w') as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    def __str__(self):
        return (
            f"TrainingConfig('{self.name}')\n"
            f"  Epochs: {self.epochs_warmup}+{self.epochs_quant}+{self.epochs_fine} = {self.total_epochs}\n"
            f"  Batch:  {self.batch_size_warmup}→{self.batch_size_quant}→{self.batch_size_fine} (AdaBatch)  |  WPE: {self.windows_per_epoch:,}\n"
            f"  LR:     {self.lr_warmup:.0e} → {self.lr_quant:.0e} → {self.lr_fine:.0e}\n"
            f"  Loss:   MSE + {self.pearson_r_weight}×R + {self.spectral_weight}×spectral\n"
            f"  ChDrop: {self.channel_dropout_min}-{self.channel_dropout_max} of 21\n"
            f"  Act:    W2A{self.activation_bits}  |  τ: {self.deadzone_tau_initial}→{self.deadzone_tau_final}\n"
            f"  {self.description}"
        )


# ============================================================
# Preset Configurations
# ============================================================

CONFIGS = {
    'fast': TrainingConfig(
        name='fast',
        description='Iteration loop — target ~3 min on RTX 4090 with AMP+compile. '
                    'Pre-2026-04-16 fast preset (10+40+50 ep, WPE 50K) took ~25 min '
                    'with the new fullband loss; that was incompatible with the '
                    'iterate-until-saturated workflow (30+ experiments per week). '
                    'Reduced to 5+15 ep / WPE 16K so the architecture/loss/HP '
                    'iteration loop completes a cycle while you stay focused.',
        epochs_warmup=5,         # was 10 — encoder needs FP32 stabilisation,
                                 # 5 epochs is enough at the smaller WPE.
        epochs_quant=15,         # was 40 — early QAT plateau is informative
                                 # for A/B but full convergence is not needed
                                 # at fast scale.
        epochs_fine=0,           # unused (Phase 3 not in train_joint anyway)
        batch_size=64,
        batch_size_warmup=64,
        batch_size_quant=128,
        batch_size_fine=256,
        windows_per_epoch=16000, # was 50000 — 312 warm batches / 156 QAT
                                 # batches per epoch is enough signal at
                                 # this scale to differentiate experiments.
        max_windows=32000,       # was 100000 — capped dataset cache stays
                                 # under 800 MB RAM (with fullband: ~3.4 GB).
        val_interval=5,          # validate every 5 epochs (so 1 warm val,
                                 # 3 QAT vals — enough to see trajectory).
        val_windows=800,         # was 2000 — CI on R is ~0.025 (acceptable
                                 # for differentiating ≥0.005 experimental
                                 # gaps; smaller would conflate experiments
                                 # with noise).
        channel_dropout_min=3,
        channel_dropout_max=10,
        # Stability fixes preserved (alpha clamping is cheap insurance).
        alpha_clamp=True,
        alpha_ceiling=5.0,
        alpha_floor=0.001,
        early_stop_patience=3,   # was 8 — ≈ 15 epochs of no improvement at
                                 # val_interval=5. Cuts early when the
                                 # trajectory has clearly plateaued.
    ),

    'standard': TrainingConfig(
        name='standard',
        description='Balanced — ~1 hour on RTX 4090. Good accuracy, reasonable iteration speed.',
        epochs_warmup=30,
        epochs_quant=120,
        epochs_fine=150,
        batch_size=32,
        batch_size_warmup=32,
        batch_size_quant=64,
        batch_size_fine=256,
        windows_per_epoch=200000,
        val_interval=10,
        val_windows=5000,
    ),

    # ----- Medium preset (joint-training diagnostic) -----
    # 200 epochs, full dataset, single continuous QAT cosine schedule.
    # No separate Fine phase — the cosine tail at lr_quant_min=1e-6
    # provides the same low-LR settling behavior with one fewer phase
    # transition to debug. Used to verify a 3.5M Tier 3 decoder
    # converges before committing 22h to the 800M Tier 7 production run.
    'medium': TrainingConfig(
        name='medium',
        description='v7.7 dress rehearsal — 200ep, full data, ~3h on RTX 4090. '
                    'SOAP + V1 decoder + GAN + clinical sampling. Validates the '
                    'production stack before committing to the long run.',
        epochs_warmup=10,
        epochs_quant=190,
        epochs_fine=0,
        batch_size=32,
        batch_size_warmup=16,        # GAN VRAM budget
        batch_size_quant=16,         # GAN doubles memory — keep batch small
        batch_size_fine=16,
        lr_warmup=2e-3,
        lr_warmup_min=1e-3,
        lr_quant=1e-3,
        lr_quant_min=1e-6,
        wd_warmup=1e-4,
        wd_quant=1e-4,
        windows_per_epoch=200000,
        max_windows=200000,          # cap for RAM fullband (~24 GB)
        val_interval=10,
        val_windows=5000,
        pearson_r_weight=0.5,
        spectral_weight=0.03,
        prd_weight=0.1,
        snac_preset='compact',
        alpha_clamp=True,
        alpha_ceiling=5.0,
        alpha_floor=0.001,
        early_stop_patience=8,           # 8 × val_interval=10 = 80 ep max plateau
        channel_dropout_min=3,
        channel_dropout_max=10,
    ),

    'production': TrainingConfig(
        name='production',
        data_dir='/mnt/4tb/data/training',
        encoder_width=256,
        encoder_blocks=6,
        encoder_kernels='3,3,5,5,7,7',
        description='Gen 7.7 — V1 encoder + V2 decoder (AA-Snake+SE+dilated). '
                    'SOAP optimizer + GAN + EMA + augmentation + seizure_head '
                    '+ clinical_sampling all ON. WSD schedule: 20ep warmup, '
                    '340ep stable, 40ep decay. Stable checkpoint for continual training. '
                    'SOAP validated +0.0135 R over AdamW (3-seed A/B/C).',
        epochs_warmup=20,           # FP32 warm start (5% of 400)
        epochs_quant=380,           # Single QAT with WSD: 20 warmup + 320 stable + 40 decay
        epochs_fine=0,              # No Phase 3 — WSD decay IS the fine phase
        batch_size=32,
        batch_size_warmup=32,       # exploratory, small batches
        batch_size_quant=16,        # GAN doubles memory — keep batch small
        batch_size_fine=256,        # unused (epochs_fine=0)
        lr_quant=1e-3,              # Peak LR for WSD stable phase
        lr_quant_min=1e-6,          # WSD decay floor
        windows_per_epoch=400000,
        max_windows=None,           # Full dataset — no cap
        val_interval=10,
        val_windows=5000,
        pearson_r_weight=0.5,
        spectral_weight=0.03,       # integrated from QAT start
        prd_weight=0.1,             # explicit PRD preservation
        snac_preset='compact',      # Experiment winner: +0.0009 R
        # Stability fixes
        alpha_clamp=True,
        alpha_ceiling=5.0,
        alpha_floor=0.001,
        early_stop_patience=12,     # 12 val intervals × 10 = 120 ep max plateau
                                    # (generous — WSD stable phase should not plateau)
        channel_dropout_min=3,
        channel_dropout_max=10,
    ),
}
