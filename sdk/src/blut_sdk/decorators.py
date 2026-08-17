# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""``@stage_ref`` / ``@workflow`` — naming ergonomics over :class:`PlanBuilder`.

This is the layer that makes the SDK *feel* like Flyte or Prefect, and it is
the layer most likely to be misread, so the boundary is worth stating plainly:

**A decorated function's body never runs in the engine.**  ``@stage_ref`` binds
a Python name to a *registered stage name* — the decorated function is a
signature and a docstring, nothing more.  Calling it appends a node that
*names* that stage.  Whatever Python you write inside the body is dead code as
far as BLUT is concerned.

That is not a limitation to be engineered around later; it is ADR 0078's
anti-re-litigation clause, restated by ADR 0111.  A front-end that could ship
task bodies into the executor is arbitrary code entering the engine, which is
exactly what the charter forbids.  If these decorators ever grow the ability to
send a body across, the response is a superseding ADR, not a patch.

To make the misreading loud rather than silent, ``@stage_ref`` rejects a
function whose body is anything but a docstring and/or ``pass``/``...``.  A user
who writes real logic there gets told at import time that it will not run,
instead of discovering it from a result that ignored their code.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from typing import Any, Callable

from .plan import PlanBuilder, PlanSpec

__all__ = ["stage_ref", "workflow", "current_builder"]

# The builder a @workflow body is currently filling. Module-level rather than
# passed explicitly so a decorated stage reads like a call; @workflow is the
# only writer and always restores the previous value.
_ACTIVE: list[PlanBuilder] = []


def current_builder() -> PlanBuilder:
    """The builder being filled by the enclosing ``@workflow``."""
    if not _ACTIVE:
        raise RuntimeError(
            "no active workflow: a @stage_ref may only be called inside a "
            "@workflow body (or a map_output body within one)"
        )
    return _ACTIVE[-1]


def _body_is_empty(func: Callable[..., Any]) -> bool:
    """True when ``func``'s body is only a docstring and/or ``pass``/``...``."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        # No source (a REPL, a C function, an exec'd string). Cannot judge, so
        # do not accuse — the check is a guard against a specific mistake, not
        # a security boundary.
        return True
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return True
    for statement in node.body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            # A docstring, or a bare `...`.
            if isinstance(statement.value.value, str) or statement.value.value is Ellipsis:
                continue
        return False
    return True


def stage_ref(stage: str | Callable[..., Any]) -> Any:
    """Bind a Python callable to a registered stage name.

    ::

        @stage_ref("prepare_data")
        def prepare(corpus: str): ...

        @workflow("nightly")
        def build():
            root = prepare(corpus="tuh")

    Called with a bare name (``@stage_ref``), the function's own name is the
    stage name.  Keyword arguments become the node's args; ``after=`` wires
    predecessors exactly as :meth:`PlanBuilder.add` does.
    """
    if callable(stage):
        return _make_stage_ref(stage.__name__, stage)

    def decorate(func: Callable[..., Any]) -> Callable[..., int]:
        return _make_stage_ref(stage, func)

    return decorate


def _make_stage_ref(stage_name: str, func: Callable[..., Any]) -> Callable[..., int]:
    if not _body_is_empty(func):
        raise ValueError(
            f"@stage_ref('{stage_name}') decorates {func.__name__}, whose body contains "
            "code. That code will NEVER run: a stage_ref only NAMES a stage the engine "
            "already has compiled in — it cannot ship a task body into the engine "
            "(ADR 0078). Move the logic into a registered stage, and leave this body "
            "as a docstring or `...`."
        )
    signature = inspect.signature(func)

    @functools.wraps(func)
    def call(*args: Any, after: Any = None, **kwargs: Any) -> int:
        if args:
            # Positional args would have to be matched to the signature to
            # become named JSON keys; requiring keywords keeps the emitted args
            # unambiguous and matches how a recipe's args are written.
            raise TypeError(
                f"{stage_name}: pass stage arguments by keyword (got {len(args)} positional)"
            )
        # Bind against the declared signature so an unknown argument is caught
        # here, with the function's own name in the message, rather than later
        # against the registry manifest or the engine.
        signature.bind(**kwargs)
        node_args = dict(kwargs) if kwargs else None
        return current_builder().add(stage_name, node_args, after=after)

    call.stage_name = stage_name  # type: ignore[attr-defined]
    return call


def workflow(name: str | Callable[..., Any]) -> Any:
    """Turn a function into a plan factory.

    The decorated function takes no engine arguments; it calls ``@stage_ref``
    functions to compose the graph.  Calling it returns a :class:`PlanSpec`.
    """
    if callable(name):
        return _make_workflow(name.__name__, name)

    def decorate(func: Callable[..., Any]) -> Callable[..., PlanSpec]:
        return _make_workflow(name, func)

    return decorate


def _make_workflow(plan_name: str, func: Callable[..., Any]) -> Callable[..., PlanSpec]:
    @functools.wraps(func)
    def build(*args: Any, **kwargs: Any) -> PlanSpec:
        builder = PlanBuilder(plan_name)
        _ACTIVE.append(builder)
        try:
            func(*args, **kwargs)
        finally:
            # Pop in `finally` so a raising body cannot leave a stale builder
            # active and silently capture the NEXT workflow's stages.
            _ACTIVE.pop()
        return builder.build()

    build.plan_name = plan_name  # type: ignore[attr-defined]
    return build
