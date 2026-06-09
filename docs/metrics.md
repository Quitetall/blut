# Verbatim metrics — `read_metric` + training-run logging

New tooling (2026-06-08) for **anti-confabulation**: training runs write a
structured metric store; `read_metric` reads it back **verbatim, with no LLM in
the path**. Use this whenever you need "what did the run actually do" — never
trust a prose digest of a trajectory.

The loop:

```
trainer  ──>  MetricLog (CSV/Parquet)  ──┐
         ──>  status.jsonl (StatusUpdate)─┼──>  read_metric  ──>  exact values + timestamps
         ──>  journald (systemd unit)     │
         ──>  wandb (optional)  ──────────┘
```

---

## 1. `read_metric` — the reader

A standalone, stdlib-only reader. Returns one JSON object on stdout. A missing
key / file / unit yields an explicit `error` — **a value is never synthesized**.

**Invoke** (from `blut/python`, or with it on `PYTHONPATH`):

```bash
python -m blut_core.read_metric <source> [selectors]
```

### Sources (exactly one required)

| Flag | Reads | Notes |
|---|---|---|
| `--run RUN_ID` | `training_logs/metrics_<RUN_ID>.csv` (or `.parquet`) | the per-epoch MetricLog feed (val_r, val_loss, train_loss, …) |
| `--csv PATH` | a metrics CSV/parquet by path | parquet needs pyarrow; else falls back to the `.csv` sibling |
| `--job JOB_ID` | `~/.local/share/lamu/train-jobs/<JOB_ID>/status.jsonl` | lamu-train job status (`StatusUpdate`) |
| `--status PATH` | a `status.jsonl` by path | one `StageEvent`/`StatusUpdate` per line |
| `--unit UNIT` | journald for a `blut-*.service` (`--user`) unit | parses JSON `StatusUpdate` lines; else the raw `MESSAGE` |
| `--run RUN_ID --manifest` | `training_logs/<RUN_ID>/RUN_MANIFEST.json` | run provenance (git_sha, config_hash, hw, ckpt SHA) |
| `--wandb ENTITY/PROJECT/RUN` | `wandb.Api().run(...).history()` | key from `~/.netrc`; entity via `wandb whoami` |

### Selectors

| Flag | Effect |
|---|---|
| `--key K` | exact column/field name; **repeatable**. On a miss → error listing available keys. CSV/wandb only. |
| `--kind K` | filter `status.jsonl`/journald by `StatusUpdate` kind (`step`/`eval`/`saved`/`done`/`failed`). |
| `--last N` | only the last N rows. For `--unit`, also caps the journald fetch. |
| `--log-dir DIR` | metrics dir. Default: `$BLUT_JOB_DIR` if set (ADR 0044 P10), else `blut/python/training_logs`. |

### Output contract

```json
{ "source": "...", "selector": "...", "available_keys": [...],
  "rows": [ {"epoch": "...", "timestamp": "...", "<key>": "..."} ],
  "n": 3, "errors": [] }
```

- Values are returned **as stored** (CSV strings stay strings — no coercion that
  could drift). Index columns (`run_id`, `epoch`, `global_epoch`, `step`,
  `phase`, `timestamp`) are attached to any `--key` selection for context.
- **Never fabricates**: a missing key/file → `rows: []` + a populated `errors`.
- **Exit code**: `0` if rows returned, `2` on no-data/not-found, `1` on usage error.
  (Lets a caller branch without parsing JSON.)

### Examples (verified)

```bash
cd blut/python

# last 3 val_r values for a run, verbatim
python -m blut_core.read_metric --run joint_fast_t6_1780898543 --key val_r --last 3

# multiple keys at once
python -m blut_core.read_metric --run <RUN_ID> --key val_r --key train_loss --key lr

# discover available keys: ask for a bogus one; the error lists them all
python -m blut_core.read_metric --run <RUN_ID> --key __list__   # -> errors:[...available...]

# a live systemd training unit's journald
python -m blut_core.read_metric --unit blut-20260608-235824-737923340-lamquant_train_joint --last 5

# a lamu-train job's status.jsonl, only failures
python -m blut_core.read_metric --job 20260510-120046-475802379 --kind failed

# run provenance
python -m blut_core.read_metric --run <RUN_ID> --manifest

# from wandb (after a run with --logger wandb).
# <entity> = your wandb entity (run `wandb whoami`, or read it off the run URL);
# project defaults to WANDB_PROJECT (=lamquant).
python -m blut_core.read_metric --wandb <entity>/lamquant/<run_id> --key val_r
```

Pipe to `jq` for shaping, e.g. `... --key val_r --last 1 | jq -r '.rows[0].val_r'`.

---

## 2. Writing metrics from a trainer

### MetricLog (always on)

`blut_core.metric_log.MetricLog` writes `metrics_<run_id>.csv`
(or `.parquet` if pyarrow is present), **rewritten atomically every epoch** so a
reader always sees a complete, valid file mid-run.

```python
from blut_core.metric_log import MetricLog
mlog = MetricLog(run_id=run_id, log_dir=Path(ROOT_DIR) / "training_logs")
mlog.append({"epoch": e, "val_r": r, "train_loss": loss, ...})  # never raises
mlog.close()
```

Wired in: `student/train_joint.py`, `oracle/train_teacher.py`,
`snn/train_4state_controller.py`.

### wandb (optional, online)

The key is already configured in `~/.netrc` (wandb ≥0.27 in the trainer venv).
Enable per-run and stream live:

```bash
WANDB_MODE=online WANDB_PROJECT=lamquant \
  python -m lamquant.snn.train_4state_controller --logger wandb ... <other args>
```

- `--logger {none,wandb}` (default `none`). `none` still writes the MetricLog CSV.
- `WANDB_MODE` env: `offline` (default) or `online`. Offline still records locally.
- run name = the MetricLog `run_id`; config = the argparse args; tags identify the
  trainer. Per-epoch scalars go to both MetricLog and (if enabled) wandb.

After a wandb run, pull it back verbatim with `read_metric --wandb`.

---

## 3. Where things land

| Artifact | Path |
|---|---|
| MetricLog CSV/Parquet | `$BLUT_JOB_DIR/metrics_<run_id>.csv` (BLUT stage) or `blut/python/training_logs/metrics_<run_id>.csv` (standalone) |
| Run provenance | `training_logs/<run_id>/RUN_MANIFEST.json` |
| Job status stream | `~/.local/share/lamu/train-jobs/<job_id>/status.jsonl` |
| systemd unit journal | `journalctl --user -u blut-<ts>-<id>-<recipe>.service` |

> Note: codec/SNN trainers (`train_joint`, `train_4state_controller`) flush python
> stdout in blocks, so their **journald has little per-epoch detail — use `--run`
> (the CSV) as the source of truth**. journald `--unit` is most useful for
> lamu-train LLM jobs that emit `StatusUpdate` lines.

---

## 4. Schemas referenced

- CSV columns = whatever the trainer `append()`s (union, first-seen order).
- `StatusUpdate` (status.jsonl / journald): `"kind"`-tagged JSON — `step`
  {step,total,loss,lr,vram_mb}, `eval` {step,eval_loss}, `saved` {path},
  `done` {final_loss,checkpoint_dir}, `failed` {error}. Defined in
  `blut/src/protocol.rs`; mirror it exactly if you add a parser.

## 5. `blut_core` — the core-cookbook primitives

`blut_core` (at `blut/python/blut_core/`) holds the domain-agnostic,
implement-once building blocks every training run needs — reusable by ANY
cookbook (lamquant, lamu), no EEG/LamQuant coupling. `torch` is lazy-imported
only inside `checkpoint`/`sysgauge`, so `import blut_core` is cheap.

| Primitive | Use | Contract |
|---|---|---|
| `runctx` | `runctx.job_dir(fallback)` / `runctx.resolve(run_id)` | resolves the `$BLUT_JOB_DIR`-or-`training_logs` anchor ONCE (ADR 0044 P10) |
| `MetricLog` | `from blut_core import MetricLog` | atomic per-epoch CSV/Parquet metric writer |
| `read_metric` | `python -m blut_core.read_metric …` | verbatim reader (§1) |
| `status` | `status.step(...)`, `.eval_pass(...)`, `.saved/.done/.failed` | emit a flushed `StatusUpdate` JSON line → stdout **+** `status.jsonl` (defeats block-buffered stdout; ADR 0044 P4) |
| `RunManifest` | `from blut_core import RunManifest` | run provenance (git_sha/config/hw/ckpt SHA), written even on crash |
| `checkpoint` | `checkpoint.save(payload, path, contract=…)` / `.load(path)` | atomic save (tmp+fsync+rename) + sidecar SHA + free-space preflight + resume-payload contract; corrupt → `CheckpointError`, never silent garbage (ADR 0044 P7) |
| `sysgauge` | `sysgauge.snapshot()` | best-effort GPU/host gauges as a dict; a missing source is **omitted, never fabricated** (ADR 0044 P10) |

```python
from blut_core import runctx, status, MetricLog, checkpoint, sysgauge

mlog = MetricLog(run_id, log_dir=runctx.job_dir(fallback))
mlog.append({"epoch": e, "val_r": r, **sysgauge.snapshot()})
status.eval_pass(step, eval_loss=v)                       # live event to journald + status.jsonl
checkpoint.save(payload, ckpt_path, contract=checkpoint.RESUME_CONTRACT)
```

## 6. ADR alignment

- **ADR 0038 (metric discipline)** — this tool IS the mandated mechanical
  defense: verify every metric against the raw source; LLM poller prose is
  liveness-only, never a quantitative trajectory. Codec quality = **fullband
  R/PRD** only (not latent/distill proxies); graded against the LQS floors in
  `Eagle/lqs/src/levels.rs` (M = R≥0.85, ADR 0043).
- **ADR 0044 P10 (observability)** — metric/log writes anchor to
  `$BLUT_JOB_DIR` (the per-job dir BLUT `--setenv`s across the systemd unit
  boundary) when run as a stage; `read_metric` + `train_4state_controller`
  honor it (fall back to `training_logs` standalone). NOT yet done for
  `train_joint` or the Rust `find_logs` (csv-only) — owner's Phase E lane.

_Source: `blut/python/blut_core/` (`runctx.py`, `metric_log.py`,
`read_metric.py`, `status.py`, `run_manifest.py`, `checkpoint.py`,
`sysgauge.py`; tests in `blut_core/tests/`), `blut/src/protocol.rs`;
decisions/0037, 0038, 0044._
