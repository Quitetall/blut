# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
#
# Make `blut_sdk` importable from the source checkout.
#
# The package uses a src layout, so `sdk/tests/*` can only `import blut_sdk`
# once `sdk/src` is on the path -- which, without this, happened only after a
# `pip install`. Nothing in the repository performs that install, so collection
# raised ModuleNotFoundError for all three test modules and pytest ABORTED
# before running anything. That is why it aborted rather than skipped: a
# collection error is fatal, so an unrelated `pytest -k "async_prefetch or ..."`
# run from `training/engine` died on the SDK's imports without ever reaching the
# tests it asked for. ADR 0103's acceptance gate is exactly that command.
#
# pytest loads a conftest before collecting its directory regardless of which
# rootdir is in effect, so this fixes the standalone run (`pytest` from `sdk/`)
# and the engine-wide run alike. An installed blut-sdk still wins: this appends
# rather than prepends, so it never shadows the real package.

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.append(str(SRC))
