# blut-dsl

Hermetic Starlark front-end for BLUT `PlanSpec` JSON. Evaluation happens out of
process so Starlark dependency features cannot alter BLUT engine serialization.

```bash
cargo install blut-dsl --version =0.2.0-alpha.1
blut-dsl recipe.star --args '{"corpus":"demo"}'
```

Licensed under AGPL-3.0-or-later.
