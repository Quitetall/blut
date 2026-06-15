"""DEPRECATED — moved to ``legacy/training/shims/train_ternary.py``.

This file has always been a thin re-export shim over
``lamquant.student.ternary_encoder``. New code should import from
``ternary_encoder`` directly. This wrapper is retained for backwards
compatibility; the re-export now goes through the legacy copy.
"""

import warnings

warnings.warn(
    "lamquant.student.train_ternary is a deprecated shim; "
    "import from lamquant.student.ternary_encoder directly. The shim "
    "now lives under legacy/training/shims/.",
    DeprecationWarning,
    stacklevel=2,
)

from ternary_encoder import *  # noqa: F401,F403,E402
