# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""ADR 0111's parity gate: ``python -m blut_sdk.tests.parity``.

Asserts that the Python builder and the Starlark front-end emit **byte-identical
canonical PlanSpec** for the same logical graph.  Two front-ends onto one IR
only stay honest if something fails when they drift, and this is that something.

Deliberately dependency-free — no pytest, no network.  The gate has to be
runnable in a bare CI container and by anyone reading the ADR, so it is a
``__main__`` that prints checks and exits non-zero on the first failure.

The golden fixture ``fixtures/parity_graph.json`` was produced by the real
``blut-dsl`` binary from ``fixtures/parity_graph.star``.  When that binary is
available the gate ALSO regenerates it live and compares, so the fixture cannot
rot silently against a front-end that moved.  When it is not, the fixture check
still runs — a checked-in golden that no longer matches the builder is caught
either way; only the "fixture vs. today's blut-dsl" question needs the binary.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from ..canonical import canonical_json_bytes, format_number
from ..plan import PlanBuilder, spec_from_json
from ..registry import RegistryManifest, ValidationError, resolve_schema

FIXTURES = Path(__file__).parent / "fixtures"
PARITY_STAR = FIXTURES / "parity_graph.star"
PARITY_JSON = FIXTURES / "parity_graph.json"
FLOAT_TABLE = FIXTURES / "serde_float_repr.json"
MANIFEST_JSON = FIXTURES / "registry_manifest.json"

_failures: list[str] = []
_checks = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        if detail:
            print("\n".join(f"          {line}" for line in detail.splitlines()))
        _failures.append(label)


def build_parity_graph() -> "PlanBuilder":
    """The Python twin of ``fixtures/parity_graph.star``.

    The plan name is the ``.star`` file's stem, because ``blut-dsl`` derives it
    that way (``plan_name_from`` in ``crates/blut-dsl/src/lib.rs``).  Getting
    that wrong is the most boring possible parity failure and the easiest to
    stare past, so it is spelled out here rather than passed in.
    """
    builder = PlanBuilder("parity_graph")
    root = builder.add(
        "prepare_data",
        {
            "corpus": "tuh",
            "limit": 3,
            "ratio": 0.25,
            "flag": True,
            "nothing": None,
            "nested": {"b": [1, 2], "a": "z"},
        },
    )
    left = builder.add("train_model", {"tier": 1}, after=root)
    right = builder.add("train_model", {"tier": 2}, after=root)
    merged = builder.add("compare_report", after=[left, right])
    shards = builder.add("prepare_data", after=merged)

    def per_shard() -> None:
        head = builder.add("train_model", {"tier": 9})
        builder.add("compare_report", after=head)

    builder.map_output(shards, per_shard, label="per-shard")
    return builder


def _diff(expected: bytes, actual: bytes) -> str:
    if expected == actual:
        return ""
    for index, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            lo = max(0, index - 60)
            return (
                f"first difference at byte {index}\n"
                f"starlark: ...{expected[lo:index + 60].decode('utf-8', 'replace')}\n"
                f"python:   ...{actual[lo:index + 60].decode('utf-8', 'replace')}"
            )
    shorter, longer = ("python", "starlark") if len(actual) < len(expected) else ("starlark", "python")
    return (
        f"{shorter} output is a prefix of {longer} "
        f"({len(actual)} vs {len(expected)} bytes)"
    )


def check_builder_matches_fixture() -> bytes:
    fixture = spec_from_json(json.loads(PARITY_JSON.read_text(encoding="utf-8")))
    expected = fixture.canonical_bytes()
    actual = build_parity_graph().canonical_bytes()
    check(
        expected == actual,
        "python builder emits the starlark fixture's canonical bytes",
        _diff(expected, actual),
    )
    return expected


def check_fixture_matches_live_dsl(expected: bytes) -> None:
    # …/engine/sdk/src/blut_sdk/tests/parity.py -> parents[4] is the engine root.
    engine_root = Path(__file__).resolve().parents[4]
    candidates = [shutil.which("blut-dsl")] + [
        str(engine_root / f"crates/blut-dsl/target/{profile}/blut-dsl")
        for profile in ("release", "debug")
    ]
    binary = next((c for c in candidates if c and Path(c).exists()), "")
    if not binary:
        # Not a failure: the fixture comparison above already pins the builder.
        # Say so out loud, though — a silently skipped check is how a gate
        # becomes vacuous, which is the failure mode this whole file guards.
        print("  SKIP  fixture vs live blut-dsl (binary not found; fixture check still ran)")
        return
    completed = subprocess.run(
        [binary, str(PARITY_STAR), "--args", '{"corpus":"tuh"}'],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        check(False, "live blut-dsl evaluates the parity script", completed.stderr[:800])
        return
    live = spec_from_json(json.loads(completed.stdout)).canonical_bytes()
    check(
        live == expected,
        "checked-in fixture still matches today's blut-dsl output",
        _diff(live, expected),
    )


def check_float_encoding() -> None:
    table = json.loads(FLOAT_TABLE.read_text(encoding="utf-8"))
    mismatches = [
        (key, expected, format_number(float(key)))
        for key, expected in table.items()
        if format_number(float(key)) != expected
    ]
    check(
        not mismatches,
        f"float encoding matches serde_json across {len(table)} measured values",
        "\n".join(f"{k}: serde={e} sdk={g}" for k, e, g in mismatches[:10]),
    )


def check_manifest_ref_resolution() -> None:
    """The manifest's schema root is a ``$ref``; a validator that ignores it
    silently accepts every argument name.  Pin that it does not."""
    if not MANIFEST_JSON.exists():
        print("  SKIP  registry manifest fixture absent")
        return
    raw = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest = RegistryManifest.from_json(raw)
    check(len(manifest.stages) > 0, "manifest fixture carries the stage palette")
    check(len(manifest.recipes) > 0, "manifest fixture carries recipes too")

    total = sum(len(s.properties) for s in manifest.stages.values())
    naive = sum(len(s.args_schema.get("properties") or {}) for s in manifest.stages.values())
    check(
        total > 0,
        f"resolving the schema $ref finds {total} declared stage arguments",
        "every stage resolved to zero arguments — either the $ref was not followed "
        "or the engine dropped its definitions map, and validation would then "
        "accept any argument name",
    )
    check(
        naive == 0 and total > 0,
        "reading `properties` without resolving $ref would have found nothing",
        f"naive read found {naive}; this fixture no longer exercises the $ref trap",
    )

    # And the validator must actually reject, not merely inspect.
    named = next((s for s in manifest.stages.values() if s.properties and not s.required), None)
    if named is not None:
        builder = PlanBuilder("typo_probe")
        builder.add(named.name, {"definitely_not_a_real_argument_xyz": 1})
        try:
            manifest.validate_plan(builder.build())
        except ValidationError:
            check(True, "an unknown argument is rejected before submission")
        else:
            check(False, "an unknown argument is rejected before submission",
                  f"validate_plan accepted a bogus argument on '{named.name}'")

    required = next((s for s in manifest.stages.values() if s.required), None)
    if required is not None:
        builder = PlanBuilder("missing_required_probe")
        builder.add(required.name, {})
        try:
            manifest.validate_plan(builder.build())
        except ValidationError:
            check(True, "a missing required argument is rejected before submission")
        else:
            check(False, "a missing required argument is rejected before submission",
                  f"validate_plan accepted '{required.name}' without {required.required}")

    builder = PlanBuilder("unknown_stage_probe")
    builder.add("no_such_stage_xyz")
    try:
        manifest.validate_plan(builder.build())
    except ValidationError:
        check(True, "an unknown stage is rejected before submission")
    else:
        check(False, "an unknown stage is rejected before submission",
              "validate_plan accepted a stage in no cookbook")

    # Naming a RECIPE where a stage belongs is the predictable first mistake,
    # and "not a registered stage" alone does not explain it.
    recipe_name = next(iter(manifest.recipes))
    builder = PlanBuilder("recipe_as_stage_probe")
    builder.add(recipe_name)
    try:
        manifest.validate_plan(builder.build())
    except ValidationError as exc:
        check("RECIPE" in str(exc), "using a recipe name as a stage says so explicitly",
              f"message was: {exc}")
    else:
        check(False, "using a recipe name as a stage says so explicitly",
              "validate_plan accepted a recipe name as a plan node")


def check_the_gate_can_fail(expected: bytes) -> None:
    """Prove the comparison is sensitive, not just green.

    A byte-equality gate that never fails is indistinguishable from one that
    never ran.  Each mutation below is a real way the two front-ends could
    drift, and each must move the canonical bytes:

    * **merge order** — ``after=[a, b]`` is the tuple element order a merge
      node's ``gather_input`` assembles.  A builder that sorted or deduplicated
      its edges would silently swap a merge's inputs.
    * **int where a float belongs** — ``0`` and ``0.25`` are different values,
      but so are ``0`` and ``0.0``; the encoder must not normalize them
      together.
    * **``True`` written as ``1``** — ``bool`` subclasses ``int`` in Python, so
      an unguarded numeric branch emits ``1`` where the engine emits ``true``.
    * **a dropped ``map_output`` label** — ``label`` is on the wire as ``null``
      when unset, so losing it is a byte change rather than an omission.
    """
    base_args = {
        "corpus": "tuh",
        "limit": 3,
        "ratio": 0.25,
        "flag": True,
        "nothing": None,
        "nested": {"b": [1, 2], "a": "z"},
    }

    def make(args: dict, merge_reversed: bool = False, label: str | None = "per-shard"):
        builder = PlanBuilder("parity_graph")
        root = builder.add("prepare_data", args)
        left = builder.add("train_model", {"tier": 1}, after=root)
        right = builder.add("train_model", {"tier": 2}, after=root)
        merged = builder.add(
            "compare_report", after=[right, left] if merge_reversed else [left, right]
        )
        shards = builder.add("prepare_data", after=merged)

        def per_shard() -> None:
            head = builder.add("train_model", {"tier": 9})
            builder.add("compare_report", after=head)

        builder.map_output(shards, per_shard, label=label)
        return builder.canonical_bytes()

    mutations = {
        "reversed merge order": make(base_args, merge_reversed=True),
        "int where the fixture has a float": make({**base_args, "ratio": 0}),
        "True written as 1": make({**base_args, "flag": 1}),
        "dropped map_output label": make(base_args, label=None),
    }
    missed = [name for name, actual in mutations.items() if actual == expected]
    check(
        not missed,
        f"the gate detects all {len(mutations)} planted drifts",
        "these mutations did NOT change the canonical bytes, so the gate is "
        "blind to them: " + ", ".join(missed),
    )


def check_resolve_schema_is_conservative() -> None:
    """A ``$ref`` this SDK cannot resolve must leave the schema alone rather
    than return something wrong — a wrong schema rejects valid plans."""
    foreign = {"$ref": "https://example.invalid/schema.json", "properties": {"a": {}}}
    check(
        resolve_schema(foreign) is foreign,
        "a non-schemars $ref is left untouched instead of mis-resolved",
    )
    dangling = {"$ref": "#/definitions/Missing", "definitions": {}}
    check(
        resolve_schema(dangling) is dangling,
        "a dangling #/definitions ref falls back to the original schema",
    )


def main() -> int:
    print("blut-sdk parity gate (ADR 0111)")
    print(f"fixture: {PARITY_JSON}")
    expected = check_builder_matches_fixture()
    check_fixture_matches_live_dsl(expected)
    check_the_gate_can_fail(expected)
    check_float_encoding()
    check_manifest_ref_resolution()
    check_resolve_schema_is_conservative()
    print()
    if _failures:
        print(f"FAIL — {len(_failures)}/{_checks} checks failed: {', '.join(_failures)}")
        return 1
    print(f"PASS — {_checks} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
