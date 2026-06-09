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
python -m lamquant.common.read_metric <source> [selectors]
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
| `--log-dir DIR` | metrics dir (default `blut/python/training_logs`). |

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
python -m lamquant.common.read_metric --run joint_fast_t6_1780898543 --key val_r --last 3

# multiple keys at once
python -m lamquant.common.read_metric --run <RUN_ID> --key val_r --key train_loss --key lr

# discover available keys: ask for a bogus one; the error lists them all
python -m lamquant.common.read_metric --run <RUN_ID> --key __list__   # -> errors:[...available...]

# a live systemd training unit's journald
python -m lamquant.common.read_metric --unit blut-20260608-235824-737923340-lamquant_train_joint --last 5

# a lamu-train job's status.jsonl, only failures
python -m lamquant.common.read_metric --job 20260510-120046-475802379 --kind failed

# run provenance
python -m lamquant.common.read_metric --run <RUN_ID> --manifest

# from wandb (after a run with --logger wandb).
# <entity> = your wandb entity (run `wandb whoami`, or read it off the run URL);
# project defaults to WANDB_PROJECT (=lamquant).
python -m lamquant.common.read_metric --wandb <entity>/lamquant/<run_id> --key val_r
```

Pipe to `jq` for shaping, e.g. `... --key val_r --last 1 | jq -r '.rows[0].val_r'`.

---

## 2. Writing metrics from a trainer

### MetricLog (always on)

`lamquant.common.metric_log.MetricLog` writes `metrics_<run_id>.csv`
(or `.parquet` if pyarrow is present), **rewritten atomically every epoch** so a
reader always sees a complete, valid file mid-run.

```python
from lamquant.common.metric_log import MetricLog
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
| MetricLog CSV/Parquet | `blut/python/training_logs/metrics_<run_id>.csv` |
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

_Source: `blut/python/lamquant/common/read_metric.py` (+ tests in
`common/tests/test_read_metric.py`), `metric_log.py`, `src/protocol.rs`._
