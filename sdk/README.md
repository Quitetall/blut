# blut-sdk

Author BLUT DAGs in Python. The builder emits canonical **PlanSpec v1** JSON;
runs are submitted and monitored by shelling the `blut` CLI.

Pure Python, no compiled extension, no runtime dependencies. The only
requirement is a `blut` binary — or a cookbook binary such as `lqt` — on `PATH`.

```python
from blut_sdk import PlanBuilder, RegistryManifest, Run

b = PlanBuilder("nightly")
data  = b.add("materialize_dataset_path", {"registered_name": "my-corpus"})
clean = b.add("filter_dataset", {"min_turns": 2}, after=data)
split = b.add("split_train_eval", {"eval_ratio": 0.1, "seed": 7}, after=data)
train = b.add("take_train", after=split)
b.add("eval_loss", {"batch_size": 4}, after=[clean, train])   # merge, in order

plan = b.build()
RegistryManifest.from_engine("lqt").validate_plan(plan)       # catch typos first
Run(plan, blut="lqt").submit()
```

## What it is, and what it deliberately is not

The SDK is a **client of the CLI**, in the sense `kubectl` is a client — except
there is no server. It opens no socket and embeds no runtime, because ADR 0034
forbids an in-engine HTTP surface and ADR 0078 forbids any front-end shipping
executable stage bodies into the engine.

That is why `@stage_ref` decorates an *empty* function:

```python
@stage_ref("train_model")
def train(tier: int): ...        # a NAME and a signature — never a body
```

The decorated function is a signature and a docstring. Calling it appends a
node that *names* a stage the engine already has compiled in. Writing real code
in that body raises at import time rather than being silently ignored — a
misreading you find immediately beats one you find in a result that quietly
skipped your code.

The SDK is **not a second runtime**. It emits the same IR the Starlark
front-end emits, so the engine behaves identically no matter which authored the
JSON — same `from_erased_graph` kind-check, same per-stage cache keys, same
plan `provenance_fingerprint`. It inherits typing and caching rather than
reimplementing them, and the clinical hard-block fires on the same path.

## Composing a graph

`PlanBuilder.add` mirrors the Starlark `add()` builtin one-to-one:

| shape             | how                                             |
| ----------------- | ----------------------------------------------- |
| graph source      | `b.add(stage)` — no `after`                     |
| linear step       | `b.add(stage, after=handle)`                    |
| fork              | two `add` calls sharing the same `after`        |
| merge             | `b.add(stage, after=[h1, h2])`                  |
| compile-time map  | a plain `for` loop                              |
| runtime map       | `b.map_output(parent, body, label=...)`         |

**`after` order is load-bearing.** For a merge it is the tuple element order
`gather_input` assembles, so `after=[a, b]` and `after=[b, a]` are different
plans. The builder wires edges at `add` time and never sorts them.

## Validating before you submit

```python
manifest = RegistryManifest.from_engine("lqt")   # shells `blut registry export`
manifest.validate_plan(plan)
```

This catches an unknown stage, an unknown argument, and a missing required
argument before a subprocess starts. The engine still re-validates
authoritatively on `recipe declare` — this shortens the loop, it does not move
the authority.

One distinction the validator enforces, because it is the predictable first
mistake: a plan node names a **stage**, not a **recipe**. They are separate
registries. Naming a recipe where a stage belongs is reported as such rather
than as a bare "not found".

## Running

```python
run = Run(plan, blut="lqt")
run.declare()                # compile + kind-check, execute nothing
job = run.submit()           # LAUNCH; returns the job id
for event in run.stream():   # StatusUpdate events as they land
    print(event)
run.cancel()
```

`declare()` is the dry run. Note there is no `--dry-run` flag on `recipe
declare` — compiling without `--run` *is* the dry run, and passing `--dry-run`
there is an unexpected-argument error (that flag belongs to `recipe run`).

## The parity gate

ADR 0111's acceptance gate ships with the package:

```sh
python -m blut_sdk.tests.parity
blut recipe declare $(python -m blut_sdk.examples.fanout --emit /tmp/plan.json)
```

The first asserts the Python builder and the Starlark front-end produce
**byte-identical canonical PlanSpec** for the same graph, and — because a
byte-equality check that never fails is indistinguishable from one that never
ran — it also plants four known drifts (reversed merge order, an int where a
float belongs, `True` written as `1`, a dropped map label) and requires each to
be caught.

The second compiles an SDK-authored plan through the engine's real
`from_erased_graph` kind-check.

## Version coupling

The SDK mirrors the PlanSpec IR, so it tracks two numbers and refuses on skew
rather than guessing:

- `PLAN_SPEC_VERSION` — the IR it emits. The engine publishes its own in the
  manifest; a mismatch raises, because plans would still parse and mean
  something subtly different.
- `SUPPORTED_MANIFEST_VERSION` — the manifest shape it reads. Additive fields
  do not move it, matching the engine's own rule.

## Development

```sh
pip install -e '.[dev]'
python -m pytest tests -q
python -m blut_sdk.tests.parity
```

Regenerate the Starlark golden fixture after an intentional IR change:

```sh
cargo build --release --manifest-path ../crates/blut-dsl/Cargo.toml
../crates/blut-dsl/target/release/blut-dsl \
    src/blut_sdk/tests/fixtures/parity_graph.star --args '{"corpus":"tuh"}' \
    > src/blut_sdk/tests/fixtures/parity_graph.json
```
