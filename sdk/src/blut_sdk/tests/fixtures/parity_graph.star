# parity_graph.star — the reference graph for the blut-sdk parity gate (ADR 0111).
#
# Exercises every wiring primitive `add()` offers, so the Python PlanBuilder is
# compared against the Starlark front-end on structure that actually varies:
# a source, a linear step, a fork, an ordered merge, and a runtime map_output
# with a template and a label. Args cover every JSON scalar type, because the
# canonical encoding is per-type and a builder could get objects right while
# getting nulls or floats wrong.
#
# Stage names here are deliberately arbitrary: this fixture is never submitted,
# only compared. The engine-compiles half of the gate lives in the fanout
# example, which names REGISTERED stages.

def per_shard():
    # A map template's single root takes the element — no `after`.
    add("train_model", {"tier": 9})
    add("compare_report", after = 0)

def build(args):
    root = add("prepare_data", {
        "corpus": args["corpus"],
        "limit": 3,
        "ratio": 0.25,
        "flag": True,
        "nothing": None,
        "nested": {"b": [1, 2], "a": "z"},
    })
    left = add("train_model", {"tier": 1}, after = root)
    right = add("train_model", {"tier": 2}, after = root)
    merged = add("compare_report", after = [left, right])
    shards = add("prepare_data", after = merged)
    map_output(shards, per_shard, label = "per-shard")
