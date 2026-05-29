"""Backward-compat shim — channel_resolver moved to lamquant_codec.

The 21-channel 10-20 resolver canonical home is
``lamquant_codec.channel_resolver`` (audit 2026-05-18 finding F1, B2
move in commit 19a00c2). This shim re-exports every public symbol so
external scripts that still import the old path keep working:

    from lamquant.dataset import channel_resolver         # ok
    from lamquant.dataset.channel_resolver import resolve # ok
    from lamquant.dataset.channel_resolver import *       # ok

New code should import directly from ``lamquant_codec.channel_resolver``.
"""
from lamquant_codec.channel_resolver import *  # noqa: F401,F403

# Bare ``from lamquant.dataset import channel_resolver`` returns
# this module object, so callers also need attribute access
# (``channel_resolver.resolve``) to land on the canonical impl. Mirror
# every public attribute of the canonical module onto this shim.
from lamquant_codec import channel_resolver as _canonical
import sys as _sys

_this = _sys.modules[__name__]
for _name in dir(_canonical):
    if not _name.startswith("_") and not hasattr(_this, _name):
        setattr(_this, _name, getattr(_canonical, _name))

del _sys, _this, _name, _canonical
