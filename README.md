# BLUT — Brian Lam's Universal Trainer

Rust-native orchestrator for local ML training. A **compile-time typed
DAG** (stages → recipes → plans) with a content-addressed cache, memory
containment via systemd cgroups, and built-in observability. One binary
drives two domains:

- **LamQuant** — the neural EEG-codec pipeline (data-prep → encoder /
  oracle / SNN / decoder / joint-codec → PCCP promotion gate). This is
  the primary workload.
- **Generic LLM** — SFT, DPO, distillation, and eval over local
  (llama.cpp) or HuggingFace Trainer backends.

**API reference:** [`API.md`](API.md) · `cargo doc --no-deps --open`.

## Why

ML pipelines accrete ad-hoc shell glue: dump data, kick off a trainer,
wait, convert a checkpoint, copy it to a serving box. Each step grows
its own retry logic, logging, and cache; a crash mid-run replays
everything. BLUT replaces the glue with a typed pipeline:

```rust
let plan = Plan::new("finetune", json!({}))
    .start(MaterializeDataset, dataset_args())
    .then(SftTrain, sft_args())
    .then(ConvertGguf, q4_k_m())
    .then(RegisterModel, register_args())
    .finish();
```

- **Stages** declare a typed `Input → Output` and the resources they
  hold (`Gpu`, `Cpu`, `Network`, `Disk`). Wrong wiring is a
  `cargo build` error, not a runtime panic.
- **Plans** are typed DAGs; **recipes** compile typed args into plans
  (a catalog of saved workflows).
- **Cache** content-addresses every stage output by
  `(stage_name, schema, input_hash, args_hash)` — crash mid-run, re-run,
  pick up where it left off.
- **Containment** runs each stage under a `systemd-run --user` transient
  unit with `MemoryMax`, so a DataLoader OOM is cgroup-killed in
  isolation instead of taking down the whole login session.
- **Observability** streams per-epoch metrics to a Parquet/CSV log
  (`pyarrow`) and, optionally, wandb (offline by default) — plus a
  `status.jsonl` event stream per job for live + post-hoc inspection.

## Install

```bash
cargo install --path .          # binary → ~/.cargo/bin/blut
```

The LamQuant recipes drive a Python payload (config-dumb subprocess
scripts). Install its venv once:

```bash
cd python
pip install -e '.[lamquant,observability]'
```

Generic-LLM GGUF recipes additionally need a `llama.cpp` checkout for
the convert + quantize tools (default `~/llama.cpp`; override with
`$BLUT_LLAMACPP_DIR`).

## Usage

Bare `blut` opens the interactive TUI cockpit. The common operations are
also subcommands (run `blut --help` for the full set — `recipe`, `jobs`,
`log`, `cancel`, `plan`, `cache`, `stage`, `data`, `auto`, `policy`,
`tui`):

```bash
blut recipe list                  # catalog (name · category · I/O kinds)
blut recipe run <name> --args '{…}'
blut jobs                         # active + completed jobs
blut log <id>                     # rendered status timeline
blut cancel <id>                  # SIGTERM the job's process group
blut cache prune                  # drop stale content-addressed outputs
blut plan resume <id>             # resume a crashed DAG at first-incomplete
blut plan inspect <name>          # show a plan's stage DAG
blut tui                          # the cockpit explicitly
```

Run a codec training under memory containment:

```bash
BLUT_CONTAINED=1 blut recipe run lamquant_joint_codec --args '{
    "lma_roots":   ["/mnt/4tb/data/Archive/lma/tuh"],
    "manifest":    "/mnt/4tb/data/Training/splits/v11.json",
    "epochs":      40,
    "logger":      "wandb"
}'
```

Background re-training (cron-driven) is gated behind an explicit policy:

```bash
blut policy enable                # writes ~/.config/blut/train-policy.toml
blut auto                         # cron-mode: decide + maybe spawn
```

## Recipes

| Recipe | Category | Backend | Output |
|---|---|---|---|
| `lamquant_data_prep` | data-prep | lamquant | `lma_corpus` |
| `lamquant_encoder` | pipeline | lamquant | `pccp_verdict` |
| `lamquant_oracle` | pipeline | lamquant | `pccp_verdict` |
| `lamquant_snn` | pipeline | lamquant | `pccp_verdict` |
| `lamquant_combined_decoder` | pipeline | lamquant | `pccp_verdict` |
| `lamquant_full_pipeline` | pipeline | lamquant | `pccp_verdict` |
| `lamquant_joint_codec` | train | lamquant | `pccp_verdict` |
| `finetune_from_dataset` | train | lamu | `model.gguf` |
| `finetune_from_conversations` | train | lamu | `model.gguf` |
| `dpo_from_preferences` | train | lamu | `model.gguf` |
| `distill_from_teacher` | train | lamu | `model.gguf` |
| `hf_finetune_from_dataset` | train | hf_trainer | `checkpoint.hf` |
| `eval_suite` | eval | lamu | `eval.report` |

`blut recipe list` is the source of truth; `recipe show <name>` prints
its args schema.

## Architecture

```
artifacts/   typed structs that reference on-disk bytes
             (LmaCorpus, JointCkpt, GgufModel, PccpVerdict, …)
stages/      atomic typed Input → Output units of work
framework/   Plan<Out> DAG builder · executor (per-Resource semaphores)
             · content-addressed cache · status broadcast + status.jsonl
             · Cookbook trait + Registry (see below)
recipes/     saved compositions of stages (the catalog above)
config/      layered config resolver + Cartesian sweep + Launcher trait
backends/    backend adapters (LamquantBackend, lamu, hf_trainer)
tui/         the interactive cockpit (ratatui)
python/      lamquant/ — config-dumb subprocess payload, fed frozen JSON
```

**Cookbook direction (ADR 0037).** BLUT is moving toward a
domain-agnostic core plus loadable *cookbooks* — a cookbook is a domain
pack (typed Rust stages/artifacts + a Python payload of config-dumb
scripts); selecting one unlocks its recipes. The `Cookbook` trait +
`Registry` seam and a standalone joint-codec recipe have landed; the
full extraction of the LamQuant pack into `cookbooks/lamquant/` is
in progress. Config layering + sweep expansion reuse the
[Lerna](https://github.com/nbprint/lerna) Rust Hydra core (vendored,
Apache-2.0) so Hydra-style overrides resolve natively in Rust.

## PCCP gate

LamQuant model recipes terminate in a `pccp_verdict` — the
Predetermined Change Control Plan promotion gate (fail-closed). A
checkpoint cannot be promoted unless the gate scores it ≥ its registered
LQS floor on real, measured fullband metrics. See the meta repo's
`pccp/` and `LamQuant-Neural/ai_models/pccp_gate.py`.

## Status

Pre-1.0. The framework, cache, containment, observability, and the
LamQuant codec recipes are end-to-end runnable. Some generic-LLM trainer
scripts (DPO, distill) accept the typed args and emit `--self-check`
JSON; full Python impls land iteratively. Cookbook extraction (ADR 0037
C-track) and cluster launchers (Slurm / Ray) are in progress / deferred.

## License

MIT. See [`LICENSE`](LICENSE).
