# demo.star — a Starlark BLUT recipe (ADR 0078).
#
# Defines build(args); calls the add() builtin to compose registered stages.
# Hermetic: no load(), no I/O, no clock/randomness — so this script's emitted
# plan is a pure function of (source, args) and is content-hashable.
#
#   add(stage, args=None, *, after=None) -> handle(int)
#     after=None        -> a graph source
#     after=handle      -> a linear step
#     after=[h1, h2, …] -> a merge (list order = tuple element order)

def build(args):
    root = add("prepare_data", {"corpus": args["corpus"]})
    heads = []
    for tier in args["tiers"]:
        heads.append(add("train_model", {"tier": tier}, after = root))
    add("compare_report", after = heads)
