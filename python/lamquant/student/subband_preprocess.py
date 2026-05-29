"""LamQuant Gen 7.1 — subband-preprocess re-export shim.

This module is a thin re-export layer over two real sources:

  * ``lamquant_codec.ops.{lpc, lifting, wht, pipeline}`` — the shipped
    codec DSP primitives.
  * ``lamquant.student._subband_int_helpers`` — the seven training-only
    Mode 3 / Approach 4 orchestration helpers that previously lived
    inline in this file (split out for audit finding F2 so the file is
    now a pure shim).

Every existing caller keeps the same name and import path. New code
should reach for the underlying canonical names (``lamquant_codec.ops.*``
for codec primitives, ``_subband_int_helpers`` for the training-only
helpers) and let this shim go away on the next sweep.
"""

# ============================================================
# Re-exports from lamquant_codec.ops
# ============================================================
# Each import uses the ops-internal name and aliases it to the
# legacy name that training scripts expect.

# -- LPC --
from lamquant_codec.ops.lpc import (
    _analyze_channel_pyref as _lpc_analyze_channel_pyref,
    _synthesize_channel_pyref as _lpc_synthesize_channel_pyref,
    analyze_channel as lpc_analyze_channel,
    _synthesize_channel_jit as _lpc_synthesize_channel_jit,
    synthesize_channel as lpc_synthesize_channel,
    analyze as lpc_analyze,
    synthesize as lpc_synthesize,
    _analyze_int_pyref as _lpc_analyze_int_pyref,
    _synthesize_int_pyref as _lpc_synthesize_int_pyref,
    Q_LPC as _Q_LPC,
    analyze_int as lpc_analyze_int,
    synthesize_int as lpc_synthesize_int,
    analyze_jit as lpc_analyze_jit,
    synthesize_jit as lpc_synthesize_jit,
)

# -- Lifting DWT --
from lamquant_codec.ops.lifting import (
    _forward_1d_pyref as _lifting_1d_forward_pyref,
    _inverse_1d_pyref as _lifting_1d_inverse_pyref,
    forward_1d as lifting_1d_forward,
    inverse_1d as lifting_1d_inverse,
    _forward_1d_int_pyref as _lifting_1d_forward_int_pyref,
    _inverse_1d_int_pyref as _lifting_1d_inverse_int_pyref,
    forward_1d_int as lifting_1d_forward_int,
    inverse_1d_int as lifting_1d_inverse_int,
    forward_1d_int_jit as lifting_1d_forward_int_jit,
    inverse_1d_int_jit as lifting_1d_inverse_int_jit,
    forward_3level as lifting_3level_forward,
    inverse_3level as lifting_3level_inverse,
    forward_3level_int as lifting_3level_forward_int,
    inverse_3level_int as lifting_3level_inverse_int,
)

# -- WHT --
from lamquant_codec.ops.wht import (
    forward_32 as wht32_forward,
    inverse_32 as wht32_inverse,
    forward_32_torch as wht32_forward_torch,
    inverse_32_torch as wht32_inverse_torch,
)

# -- Pipeline orchestrators --
from lamquant_codec.ops.pipeline import (
    hp_filter,
    preprocess_subband_single,
    preprocess_subband,
    reconstruct_from_subband,
    preprocess_subband_torch,
    reconstruct_subband_torch,
)


# ============================================================
# Re-exports from _subband_int_helpers (training-only)
# ============================================================
# Previously inline; moved out to keep this file as a pure shim. New
# callers should import from ``_subband_int_helpers`` directly.

from lamquant.student._subband_int_helpers import (
    preprocess_subband_int,
    reconstruct_from_subband_int,
    compute_l3_correction,
    reconstruct_with_l3_correction,
    _solve_detail_for_target,
    recompute_details_approach4_cascade,
    recompute_details_approach4,
)
