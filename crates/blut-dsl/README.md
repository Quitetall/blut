# blut-dsl

Hermetic Starlark front-end for BLUT `PlanSpec` JSON. Evaluation happens out of
process so Starlark dependency features cannot alter BLUT engine serialization.

```bash
cargo install blut-dsl --version =0.2.0-alpha.2
blut-dsl examples/demo.star --args '{"corpus":"demo","tiers":[1,2]}'
```

The path is `crates/blut-dsl/examples/demo.star` in a clone of the repository.
The script must define `build(args)`; `--args` is the JSON object it receives.
The PlanSpec JSON is printed to stdout. `examples/demo.star` reads both
`corpus` and `tiers`, so it fails with a Starlark `Key "tiers" was not found`
error if `tiers` is omitted.

Licensed under AGPL-3.0-or-later.
