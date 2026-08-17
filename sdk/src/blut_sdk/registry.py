# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""Client-side validation against the engine's exported registry manifest.

``blut registry export`` prints the live catalog; this module turns it into a
check that rejects an unknown stage or an unknown argument *before* a
subprocess launch.  The engine re-validates authoritatively on ``recipe
declare`` — this only shortens the loop, it does not move the authority.

**Two namespaces, and only one of them is what a plan names.**  A ``PlanSpec``
node's ``stage`` resolves through the engine's *stage* registry
(``find_erased_stage`` over ``Cookbook::stages_erased``).  A *recipe* is a
named, pre-composed chain resolved through a different registry.  Their
contents barely overlap, so validating plan nodes against the recipe list
rejects every legitimate plan.  :meth:`RegistryManifest.validate_plan`
therefore checks ``stages``; ``recipes`` is exported alongside for lookup and
for driving ``recipe run``.

**The trap this module exists to avoid.**  ``schemars`` renders an args schema
as a draft-07 document whose root is a ``$ref`` into ``definitions``::

    {"$ref": "#/definitions/Args", "definitions": {"Args": {"properties": {...}}}}

A validator that reads ``schema["properties"]`` finds nothing, concludes the
stage declares no arguments, and then accepts *every* argument name — passing
loudly while checking nothing.  That is worse than having no validator, because
it reports success.  :func:`resolve_schema` follows the ``$ref``, and the
parity gate asserts a real manifest resolves to a non-zero argument count.
"""

from __future__ import annotations

import difflib
import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable

from .plan import PLAN_SPEC_VERSION, PlanSpec

__all__ = [
    "RecipeManifest",
    "RegistryManifest",
    "StageManifest",
    "ValidationError",
    "resolve_schema",
    "SUPPORTED_MANIFEST_VERSION",
]

#: The manifest shape this SDK understands (``MANIFEST_VERSION`` in
#: ``src/cli/registry_export.rs``).  A newer engine that moved a field bumps
#: this, and :meth:`RegistryManifest.from_json` refuses rather than mis-reading.
#: Additive fields do NOT bump it, matching the rule on ``PLAN_SPEC_VERSION``.
SUPPORTED_MANIFEST_VERSION = 1


class ValidationError(Exception):
    """A plan names something the engine's catalog does not contain."""


def resolve_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Follow a schemars root ``$ref`` to the object that carries ``properties``.

    Only the root-level ``$ref`` into the sibling ``definitions`` map is
    followed — that is the shape ``schemars`` emits.  A ``$ref`` pointing
    anywhere else is returned untouched, because guessing at general
    JSON-Pointer resolution risks returning the *wrong* schema, and a wrong
    schema rejects valid plans.
    """
    if not isinstance(schema, dict):
        return {}
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/definitions/"):
        return schema
    key = ref[len("#/definitions/") :]
    target = (schema.get("definitions") or {}).get(key)
    return target if isinstance(target, dict) else schema


def _closest(name: str, candidates: Iterable[str]) -> str | None:
    """Nearest declared name, for a typo hint.  Stdlib only."""
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _check_args(what: str, name: str, schema: dict[str, Any], args: Any) -> list[str]:
    """Problems with ``args`` against a (possibly ``$ref``-rooted) schema."""
    resolved = resolve_schema(schema)
    declared = resolved.get("properties") or {}
    if not declared:
        # Not an error — something may genuinely take no arguments. Say nothing
        # further, though: with no declared names there is nothing to compare
        # against, and "accepting" every name here is the false confidence this
        # module is written to avoid.
        return []
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return [f"{what} '{name}': args must be an object, got {type(args).__name__}"]
    problems: list[str] = []
    for key in args:
        if key not in declared:
            suggestion = _closest(key, declared)
            hint = f"; did you mean '{suggestion}'?" if suggestion else ""
            problems.append(f"{what} '{name}' has no argument '{key}'{hint}")
    for key in resolved.get("required") or []:
        if key not in args:
            problems.append(f"{what} '{name}' requires argument '{key}'")
    return problems


@dataclass
class StageManifest:
    """One stage — the vocabulary a ``PlanSpec`` node names."""

    name: str
    input_kind: str = ""
    output_kind: str = ""
    element_kind: str | None = None
    args_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "StageManifest":
        return cls(
            name=data["name"],
            input_kind=data.get("input_kind", ""),
            output_kind=data.get("output_kind", ""),
            element_kind=data.get("element_kind"),
            args_schema=data.get("args_schema") or {},
        )

    @property
    def properties(self) -> dict[str, Any]:
        """Declared argument names, with the root ``$ref`` resolved."""
        return resolve_schema(self.args_schema).get("properties") or {}

    @property
    def required(self) -> list[str]:
        return list(resolve_schema(self.args_schema).get("required") or [])

    @property
    def is_source(self) -> bool:
        """True when this stage takes the unit kind, i.e. needs no predecessor."""
        return self.input_kind in ("()", "unit", "")

    def check_args(self, args: Any) -> list[str]:
        return _check_args("stage", self.name, self.args_schema, args)


@dataclass
class RecipeManifest:
    """One recipe — a named pre-composed chain, NOT a plan node's vocabulary."""

    name: str
    description: str = ""
    backend_id: str = ""
    category: str = ""
    input_kinds: list[str] = field(default_factory=list)
    output_kind: str = ""
    schedule: str | None = None
    args_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RecipeManifest":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            backend_id=data.get("backend_id", ""),
            category=data.get("category", ""),
            input_kinds=list(data.get("input_kinds") or []),
            output_kind=data.get("output_kind", ""),
            schedule=data.get("schedule"),
            args_schema=data.get("args_schema") or {},
        )

    @property
    def properties(self) -> dict[str, Any]:
        return resolve_schema(self.args_schema).get("properties") or {}

    @property
    def required(self) -> list[str]:
        return list(resolve_schema(self.args_schema).get("required") or [])

    def check_args(self, args: Any) -> list[str]:
        return _check_args("recipe", self.name, self.args_schema, args)


@dataclass
class RegistryManifest:
    """The exported catalog."""

    manifest_version: int
    engine_version: str
    plan_spec_version: int
    stages: dict[str, StageManifest]
    recipes: dict[str, RecipeManifest]

    @classmethod
    def from_json(cls, data: Any) -> "RegistryManifest":
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValidationError(
                f"registry manifest must be a JSON object, got {type(data).__name__}"
            )
        version = data.get("manifest_version")
        if version != SUPPORTED_MANIFEST_VERSION:
            # Refuse rather than best-effort. A manifest whose shape moved is
            # one whose fields may mean something else; reading it anyway is
            # how a validator starts silently approving the wrong things.
            raise ValidationError(
                f"registry manifest version {version!r} is not supported by this SDK "
                f"(expected {SUPPORTED_MANIFEST_VERSION}); upgrade blut-sdk to match "
                "the engine"
            )
        # Default to the SDK's own version when the field is absent, so a
        # manifest from an engine predating `plan_spec_version` still loads —
        # it can only have been v1, since that is when the field was added.
        ir = int(data.get("plan_spec_version", PLAN_SPEC_VERSION))
        if ir != PLAN_SPEC_VERSION:
            # ADR 0111's named rot: an SDK mirroring v1 against a v2 engine
            # emits plans that still parse and mean something subtly different.
            raise ValidationError(
                f"engine compiles PlanSpec v{ir} but this SDK emits v{PLAN_SPEC_VERSION}; "
                "the IR versions have diverged, so plans authored here may not mean "
                "what the engine reads — upgrade blut-sdk"
            )
        stages = [StageManifest.from_json(s) for s in data.get("stages") or []]
        recipes = [RecipeManifest.from_json(r) for r in data.get("recipes") or []]
        return cls(
            manifest_version=version,
            engine_version=data.get("engine_version", ""),
            plan_spec_version=ir,
            stages={s.name: s for s in stages},
            recipes={r.name: r for r in recipes},
        )

    @classmethod
    def from_engine(cls, blut: str = "blut", *, timeout: float = 60.0) -> "RegistryManifest":
        """Load by shelling ``blut registry export`` (the exec-bridge, ADR 0093)."""
        try:
            completed = subprocess.run(
                [blut, "registry", "export"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ValidationError(
                f"'{blut}' is not on PATH; blut-sdk drives the engine through its CLI, "
                "so a blut binary (or a cookbook binary such as lqt) must be installed"
            ) from exc
        if completed.returncode != 0:
            raise ValidationError(
                f"'{blut} registry export' failed with exit {completed.returncode}: "
                f"{completed.stderr.strip()[:400]}"
            )
        return cls.from_json(completed.stdout)

    def validate_plan(self, spec: PlanSpec) -> None:
        """Raise :class:`ValidationError` if ``spec`` names anything unknown."""
        problems: list[str] = []
        self._collect(spec, problems, path="")
        if problems:
            raise ValidationError(
                "plan does not match the engine's registry:\n  - " + "\n  - ".join(problems)
            )

    def _collect(self, spec: PlanSpec, problems: list[str], *, path: str) -> None:
        for index, node in enumerate(spec.nodes):
            where = f"{path}node {index}"
            stage = self.stages.get(node.stage)
            if stage is None:
                suggestion = _closest(node.stage, self.stages)
                hint = f"; did you mean '{suggestion}'?" if suggestion else ""
                # Name the right namespace in the hint when the typo is a
                # recipe name: reaching for a recipe where a stage belongs is
                # the predictable first mistake, and "not in any cookbook" on
                # its own does not explain it.
                if node.stage in self.recipes:
                    hint = (
                        "; that is a RECIPE, not a stage — a plan names stages "
                        "(run it with `blut recipe run` instead)"
                    )
                problems.append(f"{where}: '{node.stage}' is not a registered stage{hint}")
                continue
            problems.extend(f"{where}: {p}" for p in stage.check_args(node.args))
        for expansion in spec.expansions:
            self._collect(
                expansion.template, problems, path=f"{path}map over node {expansion.parent} / "
            )

    def __len__(self) -> int:
        return len(self.stages)

    def __contains__(self, name: object) -> bool:
        return name in self.stages
