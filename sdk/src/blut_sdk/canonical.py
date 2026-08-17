# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""Canonical JSON encoding — a byte-faithful port of the engine's ``canonical_json``.

The engine hashes plans and stage args through ``write_canonical``
(``src/framework/cache.rs``): no whitespace, object keys sorted, and every
scalar rendered by ``serde_json``.  ``PlanSpec::canonical_bytes`` is that
function applied to a serialized spec, and it is what ADR 0111's parity gate
compares.

**Why port it instead of shelling out.**  The parity gate has to answer "do the
Python builder and the Starlark builder describe the same plan?" without a
compiled engine in the loop — the SDK is a pure-Python package installable in a
notebook with nothing but a ``blut`` binary maybe on PATH.  A port that is
merely *close* would make that gate worse than useless: it would pass on graphs
that differ and, being the thing that certifies parity, nobody would suspect it.

**Where this port is authoritative and where it is not.**  The engine
re-canonicalizes any spec it receives, so a divergence here cannot corrupt a
provenance fingerprint or a cache key — those are computed engine-side from the
parsed JSON.  What a divergence *would* corrupt is a comparison of these bytes
against engine-produced bytes.  That is exactly what the parity gate does, so
the encoding below is pinned against a golden table measured from the real
``serde_json`` (``tests/fixtures/serde_float_repr.json``), not derived from
reading its source.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["canonical_json", "canonical_json_bytes", "format_number"]


def _format_float(value: float) -> str:
    """Render a float exactly as ``serde_json`` (via ryu) would.

    Python's ``repr`` and ryu agree on which digits are needed — both emit the
    shortest string that round-trips — and disagree only on how to lay them
    out.  Two differences, both measured against the real encoder rather than
    inferred:

    1. **Exponent padding.**  Python pads the exponent to two digits
       (``1e-07``); ryu does not (``1e-7``).
    2. **The notation switchover.**  ryu writes plain decimal for decimal
       exponents in ``[-5, 15]``; Python's threshold is ``[-4, 15]``.  They
       therefore disagree on exactly one band — magnitudes in ``[1e-5, 1e-4)``,
       which ryu writes as ``0.00001`` and Python as ``1e-05``.

    Anything outside those two rules is already identical, which is why this
    function edits ``repr`` rather than reimplementing shortest-float printing.
    """
    # NaN and the infinities have no JSON form at all: `serde_json::Number`
    # cannot hold them, so a spec containing one could never have come from the
    # engine. Refuse rather than invent a spelling (`null`, `NaN`, `1e999`) that
    # would silently make an unrepresentable plan look encodable.
    if math.isnan(value) or math.isinf(value):
        raise ValueError(
            f"{value!r} has no JSON representation; serde_json cannot encode "
            "NaN or infinity, so no plan containing it can reach the engine"
        )

    text = repr(value)
    if "e" not in text:
        return text

    mantissa, exponent = text.split("e")
    sign, digits = exponent[0], exponent[1:]
    exp = int(exponent)

    # The one band where the two encoders choose different notation. Re-render
    # in decimal with enough places to carry the mantissa's own fraction: a
    # mantissa of `1` needs 5 places (0.00001), `1.5` needs 6 (0.000015).
    if exp == -5:
        fraction_digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return f"%.{5 + fraction_digits}f" % value

    return f"{mantissa}e{sign}{digits.lstrip('0') or '0'}"


def format_number(value: int | float) -> str:
    """Render a JSON number the way ``serde_json`` does."""
    # `bool` is a subclass of `int` in Python, so an unguarded `isinstance(v,
    # int)` claims `True` is the number 1 and would emit `1` where the engine
    # emits `true` — a canonical-bytes mismatch that no amount of graph
    # inspection would explain. Callers dispatch on bool first; this assert
    # documents the contract for anyone reaching in directly.
    assert not isinstance(value, bool), "bool must be encoded as a JSON boolean"
    if isinstance(value, int):
        # Rust prints integers as plain digits with no exponent or fraction,
        # and Python's `str` on an arbitrary-precision int does the same.
        return str(value)
    return _format_float(value)


def _write(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif isinstance(value, bool):
        out.append("true" if value else "false")
    elif isinstance(value, (int, float)):
        out.append(format_number(value))
    elif isinstance(value, str):
        # `ensure_ascii=False` matches serde_json, which escapes only the
        # characters JSON requires (quote, backslash, C0 controls) and passes
        # every other code point through as UTF-8.
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write(item, out)
        out.append("]")
    elif isinstance(value, dict):
        out.append("{")
        # Rust sorts `&String` keys with `Ord for str`, which compares UTF-8
        # bytes. Sorting on the encoded bytes here is the same order Python's
        # default code-point sort would give — UTF-8 preserves code-point
        # order — but it states the rule being matched instead of relying on
        # the reader to recall that property.
        for index, key in enumerate(sorted(value, key=lambda k: _require_str(k).encode("utf-8"))):
            if index:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _write(value[key], out)
        out.append("}")
    else:
        raise TypeError(
            f"{type(value).__name__} is not JSON data; a plan's args must be "
            "plain dict/list/str/int/float/bool/None"
        )


def _require_str(key: Any) -> str:
    if not isinstance(key, str):
        # JSON object keys are strings. Python would happily let `{1: "a"}`
        # through and `json.dumps` would coerce the key to "1", quietly
        # producing a plan whose args differ from what was written.
        raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
    return key


def canonical_json(value: Any) -> str:
    """Canonical JSON text for ``value`` (sorted keys, no whitespace)."""
    out: list[str] = []
    _write(value, out)
    return "".join(out)


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON bytes — the unit ADR 0111's parity gate compares."""
    return canonical_json(value).encode("utf-8")
