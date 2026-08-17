# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""Builder behaviour the parity gate does not reach.

The gate proves the builder agrees with Starlark on ONE graph.  These cover the
edges around it: what the builder refuses, and the wire details that would
still round-trip if they were wrong.
"""

from __future__ import annotations

import json

import pytest

from blut_sdk import PLAN_SPEC_VERSION, PlanBuilder, canonical_json, spec_from_json


def test_add_returns_dense_ids_and_wires_after_in_order():
    b = PlanBuilder("p")
    a = b.add("make")
    c = b.add("x", {"k": 1}, after=a)
    d = b.add("merge", after=[a, c])
    assert (a, c, d) == (0, 1, 2)
    spec = b.build()
    # Edges into a node are contiguous and in `after` order — that order IS the
    # merge's tuple element order, so a builder that sorted edges would swap a
    # merge's inputs without changing anything visible in the node list.
    assert spec.edges == [(0, 1), (0, 2), (1, 2)]


def test_a_fork_is_two_adds_sharing_a_predecessor():
    b = PlanBuilder("p")
    root = b.add("root")
    b.add("left", after=root)
    b.add("right", after=root)
    assert b.build().edges == [(0, 1), (0, 2)]


def test_omitted_args_are_null_on_the_wire_not_absent():
    # `args` is `#[serde(default)]` with no `skip_serializing_if`, so the engine
    # emits `"args": null`. Dropping the key would still deserialize, and the
    # canonical bytes would silently stop matching the Starlark front-end.
    b = PlanBuilder("p")
    b.add("s")
    node = b.build().to_json()["nodes"][0]
    assert "args" in node and node["args"] is None


def test_condition_gates_are_never_emitted():
    # `condition_gates` is `skip_serializing_if = "Vec::is_empty"`. Emitting an
    # empty list would change the canonical bytes without changing the plan.
    b = PlanBuilder("p")
    b.add("s")
    assert "condition_gates" not in b.build().to_json()


def test_version_is_stamped():
    b = PlanBuilder("p")
    b.add("s")
    assert b.build().to_json()["version"] == PLAN_SPEC_VERSION


@pytest.mark.parametrize("bad", ["", None, 3])
def test_a_stage_name_must_be_a_non_empty_string(bad):
    with pytest.raises(TypeError):
        PlanBuilder("p").add(bad)


def test_after_rejects_a_handle_that_does_not_exist():
    b = PlanBuilder("p")
    b.add("a")
    with pytest.raises(ValueError, match="does not exist"):
        b.add("b", after=7)


def test_after_rejects_a_negative_handle():
    b = PlanBuilder("p")
    b.add("a")
    with pytest.raises(ValueError, match="non-negative"):
        b.add("b", after=-1)


def test_after_rejects_a_string():
    # `str` is iterable, so an accidental after="0" would otherwise be read as
    # a sequence of characters.
    b = PlanBuilder("p")
    b.add("a")
    with pytest.raises(TypeError, match="not a string"):
        b.add("b", after="0")


def test_a_bad_handle_does_not_half_add_the_node():
    b = PlanBuilder("p")
    b.add("a")
    with pytest.raises(ValueError):
        b.add("b", after=99)
    # The failed add must not have appended: predecessors resolve BEFORE the
    # node is created, so a rejected wiring leaves the draft untouched.
    assert len(b) == 1


def test_args_must_be_json_data():
    with pytest.raises(TypeError):
        PlanBuilder("p").add("s", {"x": object()})


def test_args_object_keys_must_be_strings():
    with pytest.raises(TypeError, match="keys must be strings"):
        PlanBuilder("p").add("s", {1: "a"})


def test_an_empty_plan_is_refused():
    with pytest.raises(ValueError, match="at least one stage"):
        PlanBuilder("p").build()


def test_a_plan_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        PlanBuilder("")


# -- map_output ----------------------------------------------------------


def test_map_output_builds_a_separate_template():
    b = PlanBuilder("p")
    shards = b.add("sharder")
    b.add("done", after=shards)

    def body():
        head = b.add("train_item")
        b.add("eval_item", after=head)

    b.map_output(shards, body, label="shard")
    spec = b.build()
    # The template is NOT inlined into the main plan.
    assert [n.stage for n in spec.nodes] == ["sharder", "done"]
    assert len(spec.expansions) == 1
    m = spec.expansions[0]
    assert m.parent == shards and m.label == "shard"
    assert m.template.name == "<map-template>"
    assert [n.stage for n in m.template.nodes] == ["train_item", "eval_item"]
    assert m.template.edges == [(0, 1)]


def test_a_template_handle_cannot_reference_the_enclosing_plan():
    # Handles are indices into their own scope. Letting an outer handle through
    # would wire an edge to an unrelated template node -- a graph the engine
    # would happily compile, wrongly.
    b = PlanBuilder("p")
    outer = b.add("a")
    b.add("b", after=outer)
    second = b.add("c", after=outer)

    def body():
        b.add("first")
        # `second` is 2 in the outer scope; the template has only node 0.
        with pytest.raises(ValueError, match="does not exist in this scope"):
            b.add("nope", after=second)

    b.map_output(second, body)


def test_a_raising_map_body_does_not_leave_the_scope_open():
    b = PlanBuilder("p")
    parent = b.add("a")

    def body():
        b.add("in_template")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        b.map_output(parent, body)
    # If the scope leaked, this add would land in the dead template instead of
    # the plan, and `build()` would raise about an open scope.
    b.add("after_the_failure", after=parent)
    spec = b.build()
    assert [n.stage for n in spec.nodes] == ["a", "after_the_failure"]
    assert spec.expansions == []


def test_map_output_cannot_nest():
    b = PlanBuilder("p")
    parent = b.add("a")

    def outer_body():
        inner = b.add("t")
        with pytest.raises(ValueError, match="cannot nest"):
            b.map_output(inner, lambda: b.add("deeper"))

    b.map_output(parent, outer_body)


def test_map_output_refuses_an_empty_body():
    b = PlanBuilder("p")
    parent = b.add("a")
    with pytest.raises(ValueError, match="added no stages"):
        b.map_output(parent, lambda: None)


def test_map_output_refuses_an_unknown_parent():
    b = PlanBuilder("p")
    b.add("a")
    with pytest.raises(ValueError, match="does not exist"):
        b.map_output(5, lambda: b.add("t"))


# -- round trip ----------------------------------------------------------


def test_json_round_trip_is_byte_stable():
    b = PlanBuilder("p")
    root = b.add("a", {"z": 1, "a": [1, 2.5, None, True]})
    b.add("b", after=root)
    spec = b.build()
    again = spec_from_json(json.loads(spec.to_json_text()))
    assert again.canonical_bytes() == spec.canonical_bytes()


def test_canonical_output_sorts_keys_and_omits_whitespace():
    assert canonical_json({"b": 1, "a": {"d": 2, "c": 3}}) == '{"a":{"c":3,"d":2},"b":1}'
