"""DEPRECATED — moved to ``legacy/training/shims/train_student_subband.py``.

This file has always been a thin re-export shim over
``lamquant.student.training_utils``. New code should import from
``training_utils`` directly. This wrapper is retained for backwards
compatibility with older scripts and BLUT stage definitions; the
re-export now goes through the legacy copy.
"""

import warnings

warnings.warn(
    "lamquant.student.train_student_subband is a deprecated shim; "
    "import from lamquant.student.training_utils directly. The shim "
    "now lives under legacy/training/shims/.",
    DeprecationWarning,
    stacklevel=2,
)

from training_utils import *  # noqa: F401,F403,E402
