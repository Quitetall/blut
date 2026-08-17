# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""``blut-sdk`` — a pure-Python client for the BLUT engine (ADR 0111).

Author a DAG in Python, emit canonical PlanSpec v1 JSON, and drive the run by
shelling the ``blut`` CLI::

    from blut_sdk import PlanBuilder, RegistryManifest, Run

    b = PlanBuilder("nightly")
    root = b.add("prepare_data", {"corpus": "tuh"})
    left = b.add("train_model", {"tier": 5}, after=root)
    right = b.add("train_model", {"tier": 7}, after=root)
    b.add("compare_report", after=[left, right])

    RegistryManifest.from_engine("lqt").validate_plan(b.build())
    run = Run(b.build(), blut="lqt")
    run.submit()

There is no compiled extension and no engine linkage: the only requirement is a
``blut`` binary (or a cookbook binary such as ``lqt``) on PATH.  The SDK is not
a second runtime — it emits the same IR the Starlark front-end does, and the
engine behaves identically regardless of which one authored the JSON.
"""

from .canonical import canonical_json, canonical_json_bytes
from .decorators import stage_ref, workflow
from .plan import (
    PLAN_SPEC_VERSION,
    MapSpec,
    PlanBuilder,
    PlanSpec,
    SpecNode,
    spec_from_json,
)
from .registry import (
    SUPPORTED_MANIFEST_VERSION,
    RecipeManifest,
    RegistryManifest,
    StageManifest,
    ValidationError,
    resolve_schema,
)
from .run import JobState, Run, RunError, submit

__version__ = "0.2.0a1"

#: The engine release this SDK's wire expectations were verified against.  The
#: SDK tracks the PlanSpec IR, not the engine's full surface, so a newer engine
#: is fine as long as `PLAN_SPEC_VERSION` and `SUPPORTED_MANIFEST_VERSION` still
#: match — which `RegistryManifest.from_json` checks rather than assumes.
VERIFIED_AGAINST_ENGINE = "0.2.0-alpha.1"

__all__ = [
    "PLAN_SPEC_VERSION",
    "SUPPORTED_MANIFEST_VERSION",
    "VERIFIED_AGAINST_ENGINE",
    "JobState",
    "MapSpec",
    "PlanBuilder",
    "PlanSpec",
    "RecipeManifest",
    "RegistryManifest",
    "Run",
    "RunError",
    "SpecNode",
    "StageManifest",
    "ValidationError",
    "__version__",
    "canonical_json",
    "canonical_json_bytes",
    "resolve_schema",
    "spec_from_json",
    "stage_ref",
    "submit",
    "workflow",
]
