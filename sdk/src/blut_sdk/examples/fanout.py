# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""A fan-out/merge plan the engine actually compiles — ADR 0111's gate example.

Run it two ways::

    python -m blut_sdk.examples.fanout                    # print the PlanSpec
    python -m blut_sdk.examples.fanout --emit plan.json   # write it, echo the path

The second form echoes the path so it composes with the engine::

    blut recipe declare $(python -m blut_sdk.examples.fanout --emit /tmp/p.json)

Note there is no ``--dry-run`` there.  ``recipe declare`` compiles, kind-checks
and renders the DAG, executing nothing, *unless* ``--run`` is passed — the
dry run is the default, and ``--dry-run`` is an unexpected-argument error on
that subcommand (it belongs to ``recipe run``).

The graph is chosen so the engine's kind-check is a real check rather than a
formality::

    materialize_dataset_path        ()             -> dataset.jsonl
      |-- filter_dataset            dataset.jsonl  -> dataset.jsonl
      |-- split_train_eval          dataset.jsonl  -> dataset.split
      |     `-- take_train          dataset.split  -> dataset.jsonl
      `-- eval_loss(after=[..])     tuple<2>       -> eval.report

The fork's two branches end in the same kind by different routes, and the
merge consumes them as a 2-tuple in ``after`` order.  A builder that wired the
merge lazily, sorted its edges, or dropped the ``dataset.split`` hop would
still produce a plausible-looking plan — and the engine would reject it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..plan import PlanBuilder, PlanSpec


def build() -> PlanSpec:
    """The fan-out plan, using stages every standard cookbook registers."""
    builder = PlanBuilder("sdk_fanout")

    # A graph source: input kind `()`, so no `after`.
    dataset = builder.add(
        "materialize_dataset_path",
        {"registered_name": "sdk-example"},
    )

    # Fork: both branches name `dataset` as their predecessor.
    filtered = builder.add(
        "filter_dataset",
        {"min_turns": 2, "drop_errors": True},
        after=dataset,
    )
    split = builder.add(
        "split_train_eval",
        # Both are `required` in the stage's schema, so omitting either is
        # exactly the mistake `RegistryManifest.validate_plan` catches before
        # a subprocess ever starts.
        {"eval_ratio": 0.1, "seed": 7},
        after=dataset,
    )
    train = builder.add("take_train", after=split)

    # Merge: a list `after` becomes the tuple, in this order.
    builder.add("eval_loss", {"batch_size": 4}, after=[filtered, train])

    return builder.build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--emit",
        metavar="PATH",
        help="write the PlanSpec to PATH and print the path (for shell substitution)",
    )
    parser.add_argument(
        "--validate-against",
        metavar="BLUT",
        help="check the plan against a live engine's registry export first",
    )
    args = parser.parse_args(argv)

    spec = build()

    if args.validate_against:
        from ..registry import RegistryManifest

        RegistryManifest.from_engine(args.validate_against).validate_plan(spec)

    if args.emit:
        path = Path(args.emit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spec.to_json_text() + "\n", encoding="utf-8")
        # ONLY the path on stdout: this is consumed by `$(...)`, so a banner
        # here would be passed to the engine as a filename.
        print(path)
    else:
        print(spec.to_json_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
