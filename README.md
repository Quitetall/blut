# BLUT — Brian Lam's Universal Trainer

Rust framework for orchestrating local ML training: SFT, DPO,
distillation, eval. Compile-time DAG via typed stages; recipes
compose stages into reusable workflows.

**API reference:** [API.md](API.md) (WIP — stabilizing post-refactor).

## Why

ML training pipelines accumulate ad-hoc shell scripts: dump some
data, kick off `trainer.py`, wait, convert checkpoint, copy to
serving box. Each step has its own retry logic, its own logging,
its own cache. Failures replay everything.

BLUT replaces the shell glue with a typed pipeline:

```rust
let plan = Plan::new("finetune", json!({}))
    .start(MaterializeConversations, since_30d())
    .then(SftTrain, sft_args())
    .then(ConvertGguf, q4_k_m())
    .then(RegisterModel, register_args())
    .finish();
```

- **Stages** declare typed `Input → Output` and the resources
  they need (`Gpu`, `Cpu`, `Network`, `Disk`).
- **Plans** are typed DAGs. Wrong wiring = `cargo build` error,
  not runtime panic.
- **Recipes** compile typed args into Plans (a catalog of saved
  workflows).
- **Cache** content-addresses every stage output by
  `(stage_name, schema, input_hash, args_hash)`. Crash mid-run +
  re-run = pick up where it left off.
- **Status events** stream to `status.jsonl` for live + post-hoc
  inspection.

## Install

```bash
cargo install --path .
# binary lands at ~/.cargo/bin/blut
```

Plus the Python trainer venv (one-time, ~3 GB):

```bash
cd python
pip install -e .
```

You'll also need a `llama.cpp` checkout for GGUF convert +
quantize tools. Default location `~/llama.cpp`; override with
`$BLUT_LLAMACPP_DIR`.

## Usage

List recipes:

```bash
blut recipe list
```

Run a recipe:

```bash
blut recipe run finetune_from_conversations --args '{
    "output_name": "perso-30d",
    "since": "30d",
    "base_model": "Qwen/Qwen3-7B"
}'
```

Watch a running job:

```bash
blut jobs                          # list active + completed
blut log <id>                      # rendered status timeline
blut cancel <id>                   # SIGTERM
```

Auto-trigger heuristic (cron-driven background re-training):

```bash
blut policy enable                 # writes ~/.config/blut/train-policy.toml
blut auto                          # cron-mode: decide + maybe spawn
```

## Architecture

```
artifacts/      typed Rust structs that reference on-disk bytes
                (DatasetJsonl, HfCheckpoint, GgufModel, …)

stages/         atomic units of work, typed Input → Output
                (materialize_conversations, sft_train,
                 convert_gguf, register_model, dpo_train,
                 distill_train, …)

framework/      Plan<Out> typed DAG builder, executor with
                per-Resource semaphores, content-addressed cache,
                status broadcast channel + status.jsonl writer

recipes/        saved compositions of stages
                (finetune_from_conversations,
                 dpo_from_preferences, …)

python/         trainer.py + paradigm-specific siblings
                (trainer_dpo.py, trainer_distill.py)
```

## Built-in recipes (v1)

- `finetune_from_conversations` — SFT a base on user's
  conversation history. Pulls from SQLite, filters, trains via
  `trainer.py`, converts to GGUF, registers.
- `dpo_from_preferences` — DPO from a preference JSONL. (Stub
  trainer in v1; full impl WIP.)
- More on the way (distill_from_teacher, eval_suite, …).

## Status

Pre-1.0. Core framework is working; SFT recipe is end-to-end
runnable. DPO + distill trainer scripts are stubs that accept the
typed args and emit `--self-check` JSON; full Python
implementations land iteratively.

## License

MIT. See [LICENSE](LICENSE).
