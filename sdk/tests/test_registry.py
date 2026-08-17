# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""Client-side validation.

The recurring theme: a validator's dangerous failure is not rejecting a good
plan, it is ACCEPTING a bad one while appearing to have checked.  Most of these
assert a rejection actually happens.
"""

from __future__ import annotations

import json

import pytest

from blut_sdk import PlanBuilder, RegistryManifest, ValidationError, resolve_schema

REF_SCHEMA = {
    "$ref": "#/definitions/Args",
    "definitions": {
        "Args": {
            "type": "object",
            "properties": {"tier": {"type": "integer"}, "seed": {"type": "integer"}},
            "required": ["tier"],
        }
    },
}


def manifest(**overrides) -> RegistryManifest:
    data = {
        "manifest_version": 1,
        "engine_version": "0.2.0-alpha.1",
        "plan_spec_version": 1,
        "stages": [
            {
                "name": "train_model",
                "input_kind": "()",
                "output_kind": "ckpt",
                "args_schema": REF_SCHEMA,
            },
            {"name": "no_args", "input_kind": "ckpt", "output_kind": "report", "args_schema": {}},
        ],
        "recipes": [{"name": "nightly_train", "args_schema": {}}],
    }
    data.update(overrides)
    return RegistryManifest.from_json(data)


def plan(stage: str, args=None):
    b = PlanBuilder("p")
    b.add(stage, args)
    return b.build()


# -- the $ref trap -------------------------------------------------------


def test_resolve_schema_follows_a_schemars_root_ref():
    assert resolve_schema(REF_SCHEMA)["properties"].keys() == {"tier", "seed"}


def test_reading_properties_naively_would_see_nothing():
    # Pins the trap itself: if this ever starts finding properties, the schema
    # shape changed and the resolution logic needs revisiting.
    assert REF_SCHEMA.get("properties") is None


def test_a_foreign_ref_is_returned_untouched():
    foreign = {"$ref": "https://example.invalid/s.json", "properties": {"a": {}}}
    assert resolve_schema(foreign) is foreign


def test_a_dangling_ref_falls_back_rather_than_raising():
    dangling = {"$ref": "#/definitions/Gone", "definitions": {}}
    assert resolve_schema(dangling) is dangling


# -- validation ----------------------------------------------------------


def test_a_good_plan_passes():
    manifest().validate_plan(plan("train_model", {"tier": 3}))


def test_an_unknown_stage_is_rejected():
    with pytest.raises(ValidationError, match="not a registered stage"):
        manifest().validate_plan(plan("trian_model"))


def test_a_near_miss_gets_a_suggestion():
    with pytest.raises(ValidationError, match="did you mean 'train_model'"):
        manifest().validate_plan(plan("train_modle", {"tier": 1}))


def test_a_recipe_name_used_as_a_stage_says_which_namespace():
    with pytest.raises(ValidationError, match="that is a RECIPE, not a stage"):
        manifest().validate_plan(plan("nightly_train"))


def test_an_unknown_argument_is_rejected():
    with pytest.raises(ValidationError, match="has no argument 'teir'"):
        manifest().validate_plan(plan("train_model", {"tier": 1, "teir": 2}))


def test_a_missing_required_argument_is_rejected():
    with pytest.raises(ValidationError, match="requires argument 'tier'"):
        manifest().validate_plan(plan("train_model", {"seed": 1}))


def test_a_stage_with_no_declared_args_accepts_anything():
    # Nothing to compare against, so nothing is claimed. The alternative --
    # rejecting every arg for a stage whose schema we could not read -- would
    # break valid plans on a schema-shape change.
    manifest().validate_plan(plan("no_args", {"whatever": 1}))


def test_args_must_be_an_object_when_the_stage_declares_some():
    with pytest.raises(ValidationError, match="must be an object"):
        manifest().validate_plan(plan("train_model", [1, 2]))


def test_validation_reaches_inside_a_map_template():
    # A template is a plan too; skipping it would let a typo through in exactly
    # the place a user is least likely to be watching.
    b = PlanBuilder("p")
    parent = b.add("train_model", {"tier": 1})
    b.map_output(parent, lambda: b.add("nope_not_a_stage"))
    with pytest.raises(ValidationError, match="map over node 0"):
        manifest().validate_plan(b.build())


def test_every_problem_is_reported_not_just_the_first():
    b = PlanBuilder("p")
    b.add("bogus_one")
    b.add("bogus_two")
    with pytest.raises(ValidationError) as exc:
        manifest().validate_plan(b.build())
    assert "bogus_one" in str(exc.value) and "bogus_two" in str(exc.value)


# -- version skew --------------------------------------------------------


def test_an_unsupported_manifest_version_is_refused():
    with pytest.raises(ValidationError, match="not supported by this SDK"):
        manifest(manifest_version=2)


def test_a_diverged_plan_spec_version_is_refused():
    # ADR 0111's named rot: an SDK mirroring v1 against a v2 engine emits plans
    # that still parse and mean something subtly different.
    with pytest.raises(ValidationError, match="IR versions have diverged"):
        manifest(plan_spec_version=2)


def test_a_manifest_without_plan_spec_version_still_loads():
    # Predates the field, so it can only have been v1.
    m = RegistryManifest.from_json(
        {"manifest_version": 1, "engine_version": "x", "stages": [], "recipes": []}
    )
    assert m.plan_spec_version == 1


def test_a_non_object_manifest_is_refused():
    with pytest.raises(ValidationError, match="must be a JSON object"):
        RegistryManifest.from_json("[1,2,3]")


def test_from_engine_reports_a_missing_binary():
    with pytest.raises(ValidationError, match="not on PATH"):
        RegistryManifest.from_engine("definitely-not-a-real-binary-xyz")
