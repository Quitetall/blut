"""Named experiment recipes for the unified ExperimentRunner.

Each recipe is a dict of config overrides + metadata. Composable with
TrainingConfig via `.replace(**recipe['config_overrides'])`.

Usage:
    from lamquant.common.recipes import RECIPES, get_recipe
    recipe = get_recipe('snac_balanced')
    cfg = base_config.replace(**recipe['config_overrides'])
"""

RECIPES = {
    # ── Baselines ──
    'baseline': {
        'desc': 'V1 baseline (w=128, 3 focal, full conv, SEQ+SubLN, DitheredFSQ)',
        'config_overrides': {},
        'augmentor': None,
        'multiscale_fsq': None,
    },

    # ── Augmentation variants ──
    'selfeeg_light': {
        'desc': 'selfEEG augmentation, light (SNR=30dB, p=0.3)',
        'config_overrides': {},
        'augmentor': ('selfeeg', 'light'),
        'multiscale_fsq': None,
    },
    'selfeeg_moderate': {
        'desc': 'selfEEG augmentation, moderate (SNR=20dB, p=0.5)',
        'config_overrides': {},
        'augmentor': ('selfeeg', 'moderate'),
        'multiscale_fsq': None,
    },
    'selfeeg_aggressive': {
        'desc': 'selfEEG augmentation, aggressive (SNR=15dB, p=0.7)',
        'config_overrides': {},
        'augmentor': ('selfeeg', 'aggressive'),
        'multiscale_fsq': None,
    },
    'builtin_aug': {
        'desc': 'Built-in augmentation (noise+dropout+mask, p=0.5)',
        'config_overrides': {},
        'augmentor': ('builtin', 'moderate'),
        'multiscale_fsq': None,
    },

    # ── SNAC multi-scale FSQ ──
    'snac_balanced': {
        'desc': 'SNAC balanced (strides=[8,4,2,1], L=[3,3,5,5], 82:1)',
        'config_overrides': {},
        'augmentor': None,
        'multiscale_fsq': 'balanced',
    },
    'snac_compact': {
        'desc': 'SNAC compact (strides=[8,4,2,1], L=[2,2,3,3], 122:1)',
        'config_overrides': {},
        'augmentor': None,
        'multiscale_fsq': 'compact',
    },
    'snac_flat': {
        'desc': 'SNAC flat (stride=[1], L=[5], 143:1)',
        'config_overrides': {},
        'augmentor': None,
        'multiscale_fsq': 'flat',
    },

    # ── Schedule variants ──
    'progressive_tau': {
        'desc': 'Progressive ternary: tau=0.1 soft 80%, anneal 20%',
        'config_overrides': {'tau_schedule': 'progressive'},
        'augmentor': None,
        'multiscale_fsq': None,
    },
    'two_stage_wd': {
        'desc': 'Two-stage WD: normal 2/3, zero final 1/3',
        'config_overrides': {'wd_schedule': 'two_stage'},
        'augmentor': None,
        'multiscale_fsq': None,
    },

    # ── Extended experiments ──
    'q2d2_l5': {
        'desc': 'Q2D2 pairwise channel quantization (L=5, 25 joint codes per pair)',
        'config_overrides': {},
        'augmentor': None,
        'multiscale_fsq': None,
        'q2d2': True,
    },
    'snac_balanced_long': {
        'desc': 'SNAC balanced at 400 epochs (latent reorganization hypothesis)',
        'config_overrides': {'epochs_quant': 380, 'wd_schedule': 'two_stage'},
        'augmentor': None,
        'multiscale_fsq': 'balanced',
    },
    'snac_compact_long': {
        'desc': 'SNAC compact at 400 epochs (baseline for balanced comparison)',
        'config_overrides': {'epochs_quant': 380, 'wd_schedule': 'two_stage'},
        'augmentor': None,
        'multiscale_fsq': 'compact',
    },
    'combined_winners': {
        'desc': 'Combined: two-stage WD + SNAC compact (stack both winners)',
        'config_overrides': {'wd_schedule': 'two_stage'},
        'augmentor': None,
        'multiscale_fsq': 'compact',
    },
}


def get_recipe(name: str) -> dict:
    """Get a named recipe. Raises KeyError if not found."""
    if name not in RECIPES:
        available = ', '.join(sorted(RECIPES.keys()))
        raise KeyError(f"Unknown recipe '{name}'. Available: {available}")
    return RECIPES[name]


def list_recipes() -> list:
    """Return list of (name, description) tuples."""
    return [(name, r['desc']) for name, r in RECIPES.items()]
