# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""The PlanSpec v1 IR and the builder that emits it (ADR 0111, on ADR 0078).

``PlanBuilder`` mirrors the Starlark ``add()`` builtin one-to-one, because the
two front-ends must not be able to disagree about what a graph means:

===================  ==========================================
Starlark             Python
===================  ==========================================
``add(s)``           ``b.add(s)``
``add(s, a, after=h)``  ``b.add(s, a, after=h)``
fork                 two ``add`` calls sharing ``after``
merge                ``after=[h1, h2]`` (order = tuple order)
compile-time map     a ``for`` loop
runtime map          ``b.map_output(parent, body)``
===================  ==========================================

The builder's only output is canonical PlanSpec v1 JSON, so a plan authored
here and a plan authored in Starlark hit the same ``from_erased_graph``
kind-check and the same ``provenance_fingerprint``.  This module is a *mirror*
of ``src/framework/plan_spec.rs``, and the field-level notes below record where
the wire format would bite a naive re-implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .canonical import canonical_json_bytes

__all__ = [
    "PLAN_SPEC_VERSION",
    "SpecNode",
    "MapSpec",
    "PlanSpec",
    "PlanBuilder",
    "spec_from_json",
]

#: Mirrors ``PLAN_SPEC_VERSION`` in ``src/framework/plan_spec.rs``.  Bumped only
#: on a backward-incompatible IR change; additive fields do not move it.
PLAN_SPEC_VERSION = 1


def _check_args(args: Any) -> Any:
    """Reject args that cannot survive the trip to the engine.

    ``SpecNode`` is ``#[serde(deny_unknown_fields)]`` and its ``args`` is a
    ``serde_json::Value``: anything that is not plain JSON data fails at the
    engine, after a subprocess launch, with a serde error naming a byte offset.
    Catching it here costs one traversal and reports the offending type.
    """
    if args is None:
        return None
    # canonical_json raises TypeError on non-JSON data and on non-string keys.
    canonical_json_bytes(args)
    return args


@dataclass
class SpecNode:
    """One node: a registered stage name plus its args.

    ``retry``/``timeout``/``priority``/``pure`` exist in PlanSpec v1.1 but are
    deliberately not builder surface (see ``PlanBuilder.add``).  They are
    ``skip_serializing_if`` on the Rust side, so omitting them is what keeps a
    builder-authored node byte-identical to a Starlark-authored one.
    """

    stage: str
    args: Any = None

    def to_json(self) -> dict[str, Any]:
        # `args` is `#[serde(default)]` WITHOUT `skip_serializing_if`, so the
        # engine emits `"args": null` for an argless node rather than dropping
        # the key. Dropping it here would still deserialize — but the canonical
        # bytes would differ from the Starlark front-end's, and the parity gate
        # exists to catch exactly that class of near-miss.
        return {"stage": self.stage, "args": self.args}


@dataclass
class MapSpec:
    """A runtime fan-out: run ``template`` once per element of ``parent``'s list."""

    parent: int
    template: "PlanSpec"
    label: str | None = None

    def to_json(self) -> dict[str, Any]:
        # `label` is `#[serde(default)]` with no `skip_serializing_if`, so —
        # unlike the node knobs — it IS on the wire as null when unset.
        return {
            "parent": self.parent,
            "template": self.template.to_json(),
            "label": self.label,
        }


@dataclass
class PlanSpec:
    """The plan IR.  Node ids are dense indices into ``nodes``."""

    name: str
    nodes: list[SpecNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    expansions: list[MapSpec] = field(default_factory=list)
    version: int = PLAN_SPEC_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_json() for n in self.nodes],
            # Rust `(u32, u32)` serializes as a two-element array.
            "edges": [[a, b] for a, b in self.edges],
            "expansions": [m.to_json() for m in self.expansions],
            # `condition_gates` is `skip_serializing_if = "Vec::is_empty"`, so
            # an ungated plan must NOT carry the key. The SDK builds no gates,
            # so it is never emitted — emitting `[]` would change the canonical
            # bytes without changing the plan.
            "version": self.version,
        }

    def canonical_bytes(self) -> bytes:
        """The bytes ADR 0111's parity gate compares."""
        return canonical_json_bytes(self.to_json())

    def to_json_text(self, *, indent: int | None = 2) -> str:
        """Serialize for ``blut recipe declare``.

        Written with ``ensure_ascii=False`` so a non-ASCII arg reaches the
        engine as the UTF-8 the author wrote, matching what serde produces.
        """
        return json.dumps(self.to_json(), indent=indent, ensure_ascii=False)


class PlanBuilder:
    """Accumulates nodes and edges, then freezes into a :class:`PlanSpec`.

    Mirrors ``PlanDraft`` in ``crates/blut-dsl/src/builder.rs``: ``add`` appends
    one node and wires its predecessors immediately, so every edge into a node
    is contiguous and in ``after`` order — which is the tuple element order a
    merge node's ``gather_input`` assembles.  Wiring lazily, or sorting edges,
    would silently permute a merge's inputs.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("a plan needs a name (it labels the run)")
        self.name = name
        self._nodes: list[SpecNode] = []
        self._edges: list[tuple[int, int]] = []
        self._expansions: list[MapSpec] = []
        # A stack, matching the Starlark `DslStore`: `map_output` pushes a
        # template scope so `add` calls inside the body build the template.
        self._scopes: list[tuple[list[SpecNode], list[tuple[int, int]], list[MapSpec]]] = []

    # -- scope plumbing ---------------------------------------------------

    @property
    def _current(self) -> tuple[list[SpecNode], list[tuple[int, int]], list[MapSpec]]:
        return self._scopes[-1] if self._scopes else (self._nodes, self._edges, self._expansions)

    # -- authoring surface ------------------------------------------------

    def add(
        self,
        stage: str,
        args: Any = None,
        *,
        after: int | Iterable[int] | None = None,
    ) -> int:
        """Append a stage node; return its handle (its dense index).

        ``after`` omitted makes a graph source, a single handle makes a linear
        step, and a list makes a merge in that order.  ``args`` defaults to
        JSON ``null``, matching a declarative recipe's default.

        The v1.1 per-node knobs (``retry``, ``timeout``, ``priority``,
        ``pure``) are intentionally absent.  ``pure`` in particular is not the
        SDK's to grant: the engine accepts it only when the registered stage
        independently certifies ``SPECULATION_SAFE``, so a builder flag would
        be a request the engine is free to reject — a confusing surface whose
        only honest value is the default.  Authors who need the knobs write the
        JSON, which is still a supported door.
        """
        if not isinstance(stage, str) or not stage:
            raise TypeError("stage must be a non-empty string naming a registered stage")
        nodes, edges, _ = self._current
        # Resolve predecessors BEFORE appending, so a bad handle raises without
        # having already mutated the draft — and so the bounds check compares
        # against the nodes that actually exist, which cannot include this one.
        predecessors = self._parse_after(after)
        node_id = len(nodes)
        nodes.append(SpecNode(stage=stage, args=_check_args(args)))
        for predecessor in predecessors:
            edges.append((predecessor, node_id))
        return node_id

    def _parse_after(self, after: int | Iterable[int] | None) -> list[int]:
        if after is None:
            return []
        # `str` is iterable, so an accidental `after="0"` would otherwise be
        # read as a sequence of characters and fail with a confusing message.
        if isinstance(after, str):
            raise TypeError("after takes node handles (ints), not a string")
        handles = [after] if isinstance(after, int) and not isinstance(after, bool) else list(after)
        parsed: list[int] = []
        node_count = len(self._current[0])
        for handle in handles:
            if isinstance(handle, bool) or not isinstance(handle, int):
                raise TypeError(f"after: every handle must be an int, got {type(handle).__name__}")
            if handle < 0:
                raise ValueError(f"after: node handle must be non-negative, got {handle}")
            # Handles are indices into THIS scope. A handle from an enclosing
            # scope used inside a map body would point at an unrelated template
            # node -- an edge the engine would happily compile into the wrong
            # graph. Bounds-check against the scope that owns the edge.
            if handle >= node_count:
                in_scope = f"0..{node_count - 1}" if node_count else "none yet"
                raise ValueError(
                    f"after: node handle {handle} does not exist in this scope "
                    f"(in scope: {in_scope}); handles from an enclosing plan are "
                    "not valid inside a map_output body"
                )
            parsed.append(handle)
        return parsed

    def map_output(
        self,
        parent: int,
        body: Callable[[], None],
        *,
        label: str | None = None,
    ) -> None:
        """Declare a runtime fan-out over ``parent``'s list output.

        ``body`` is a zero-argument callable whose ``add`` calls build the
        template; the template's root (the node with no ``after``) consumes one
        element.  Nested maps are not allowed in v1, and the engine rejects
        them, so this refuses locally rather than after a subprocess launch.
        """
        if isinstance(parent, bool) or not isinstance(parent, int):
            raise TypeError("map_output: parent must be a node handle (int)")
        if parent < 0:
            raise ValueError(f"map_output: parent handle must be non-negative, got {parent}")
        if parent >= len(self._current[0]):
            raise ValueError(f"map_output: parent handle {parent} does not exist in this scope")
        if self._scopes:
            raise ValueError(
                "map_output cannot nest: PlanSpec v1 allows no map inside a map "
                "template, and the engine rejects it at compile time"
            )
        scope: tuple[list[SpecNode], list[tuple[int, int]], list[MapSpec]] = ([], [], [])
        self._scopes.append(scope)
        try:
            body()
        finally:
            # Pop unconditionally: a body that raises must not leave a dangling
            # template scope that silently swallows every later `add`.
            self._scopes.pop()
        if not scope[0]:
            raise ValueError("map_output: the body added no stages, so there is nothing to map")
        template = PlanSpec(
            # The Starlark front-end names every template `<map-template>`;
            # matching it is what makes the two emitters byte-identical.
            name="<map-template>",
            nodes=scope[0],
            edges=scope[1],
            expansions=scope[2],
        )
        self._current[2].append(MapSpec(parent=parent, template=template, label=label))

    # -- freeze -----------------------------------------------------------

    def build(self) -> PlanSpec:
        """Freeze into a :class:`PlanSpec`."""
        if self._scopes:
            raise RuntimeError("internal: a map_output template scope was left open")
        if not self._nodes:
            raise ValueError("a plan needs at least one stage")
        return PlanSpec(
            name=self.name,
            nodes=list(self._nodes),
            edges=list(self._edges),
            expansions=list(self._expansions),
        )

    def validate(self, manifest: Any) -> None:
        """Check stage names and args against an exported registry manifest.

        Delegates to :mod:`blut_sdk.registry`; kept here so the common flow is
        ``build → validate → submit`` on one object.
        """
        from .registry import RegistryManifest

        if not isinstance(manifest, RegistryManifest):
            manifest = RegistryManifest.from_json(manifest)
        manifest.validate_plan(self.build())

    def to_json_text(self, *, indent: int | None = 2) -> str:
        return self.build().to_json_text(indent=indent)

    def canonical_bytes(self) -> bytes:
        return self.build().canonical_bytes()

    def __len__(self) -> int:
        return len(self._nodes)


def spec_from_json(data: dict[str, Any]) -> PlanSpec:
    """Rebuild a :class:`PlanSpec` from parsed engine/DSL JSON.

    Used by the parity gate to canonicalize the Starlark fixture through the
    same code path as a builder-authored plan, so a difference in the result is
    a difference in the *graph* and not in how two functions happened to
    serialize it.
    """
    return PlanSpec(
        name=data["name"],
        nodes=[SpecNode(stage=n["stage"], args=n.get("args")) for n in data["nodes"]],
        edges=[(int(a), int(b)) for a, b in data.get("edges", [])],
        expansions=[
            MapSpec(
                parent=int(m["parent"]),
                template=spec_from_json(m["template"]),
                label=m.get("label"),
            )
            for m in data.get("expansions", [])
        ],
        version=int(data.get("version", PLAN_SPEC_VERSION)),
    )
