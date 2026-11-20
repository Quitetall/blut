# BLUT Trainer Contract v1

Frozen 2026-08-08 (ADR 0037 Stage 0); revision 1.1 (2026-10-23) adds §8, mid-stage
checkpoint slots. This is the wire contract between the orchestrator
(the Rust engine) and every trainer subprocess, across all cookbooks (LLM, LamQuant/EEG).
The reference implementation will ship publicly as `tritium.torch.contract` (ADR 0037
Stage 1); this document is
the authority, and the golden fixtures under `tests/contract/` are its executable form.
Both this repo's CI and consumers' CIs validate against those fixtures.

Versioning: additive changes (new optional fields, new `BLUT_*` line prefixes) bump the
minor revision of this document only; anything a v1 reader would misparse requires
`BLUT_CONTRACT 2` and a new fixture set. Readers MUST treat streams with no announcement
line as v0-legacy and still parse channel 1.

## 1. Control channel — kind-tagged JSON status lines (stdout)

One JSON object per line, no envelope, each line standing alone (a trainer that crashes
mid-line must leave parseable lines up to the crash). Serde-style `kind` tag:

| kind | required fields | notes |
|---|---|---|
| `step` | `step:u32, total:u32, loss:f32, lr:f32, vram_mb:u32` | 1-indexed |
| `eval` | `step:u32, eval_loss:f32` | step at which eval triggered |
| `saved` | `path:string` | many per run is normal |
| `done` | `final_loss:f32, checkpoint_dir:string` | terminal |
| `failed` | `error:string` | terminal; process then exits non-zero; `error` is human-readable, never parsed |
| `heartbeat` | — (optional `phase:string`, `vram_mb:u32`) | liveness only; SHOULD be emitted every ≤30 s during legitimately silent phases (model load, tokenization, checkpoint save) |

Rules:
- Emitters MUST NOT invent kinds. Readers reject unknown kinds as protocol errors but
  MUST tolerate and log (not stall on) malformed lines.
- Extra fields on known kinds are ignored by readers (serde default) — emitters MAY add
  fields, but a field needed for correctness belongs in this table first.
- Exactly one terminal event (`done` | `failed`) per run, as the last control-channel line.
- Exit-code semantics: `0` means the process ran to completion; quality gating is the
  orchestrator's job (PCCP stages), never the trainer's exit code.

### Contract announcement (RECOMMENDED in v1)

First stdout line: `BLUT_CONTRACT 1`. Current readers treat non-JSON lines as malformed —
warn-log-and-continue, never stall — so this is backward-compatible today; contract-aware
readers (from Stage 2) recognize the prefix explicitly. Absence ⇒ v0-legacy stream.

## 2. Metric channel — `BLUT_METRIC <json>` lines (stdout, multiplexed)

Free-form observability that must not bloat the typed enum:

```
BLUT_METRIC {"kind":"epoch","phase":"quant","val_r":0.912,"prd":4.31,...}
```

- Payload values MUST be numeric (int/float, bools excluded); `kind` tags the event
  (`"epoch"` is the per-epoch line), `phase` is optional.
- Readers route these to metric logs/ledger; they carry no control semantics and never
  terminate a run.

## 3. Liveness and crash gating

Two mechanisms, both frozen:
- **In-band**: the `heartbeat` control kind (above) feeds the engine's liveness watchdog.
- **Durable**: trainers with resumable state write `heartbeat_unix` into their
  `state.json` every `HEARTBEAT_INTERVAL = 60` seconds. This constant MUST equal the Rust
  engine's `resume::HEARTBEAT_INTERVAL_SECS`; the engine — never the trainer — decides
  whether a prior run crashed (stale heartbeat) and whether resume is permitted.

## 4. Checkpoint envelope

A checkpoint is a dict (torch.save or equivalent) with keys:

| key | v1 status | contents |
|---|---|---|
| `model` | REQUIRED | model state_dict |
| `opt` | REQUIRED | optimizer state_dict |
| `config` | REQUIRED | the run's resolved config (dict), hashable canonically |
| `step` | REQUIRED | int global step |
| `rng` | REQUIRED (new in v1) | `{python, numpy, torch, cuda?}` RNG states |
| `ema` | optional | EMA weights state |
| `sched` | optional | LR-schedule state (needed for stateful schedules, e.g. WSD∞) |
| `manifest_ref` | optional | path/digest of the RUN_MANIFEST this checkpoint belongs to |

**v1 key aliases** (readers MUST accept; writers SHOULD emit canonical; the reference
implementation reads both and writes canonical):

| canonical | accepted alias | source |
|---|---|---|
| `model` | `state_dict` | LamQuant `checkpoint_manager.py` |
| `opt` | `optimizer` | LamQuant |
| `rng` | `rng_state` | LamQuant |
| `config` | `training_config_hash` (config-by-hash in provenance) | LamQuant; full config becomes canonical going forward |

Extra keys (e.g. LamQuant's `epoch`, `data_cursor`, `saved_at`, provenance fields) are
allowed and preserved by readers.

Writers MUST save atomically (tmp + fsync + rename, keeping the previous checkpoint until
the new one is durable) and SHOULD perform a post-save reload + forward smoke check. The
reference implementation (`tritium.torch.checkpoint`, Stage 1) provides both.

Known v1 deviations (documented per Stage-0 acceptance; to be closed in Stage 3/4):
- `~/blut python/trainer_distill.py` saves `{model, opt, config, step}` — missing `rng`,
  non-atomic save, no heartbeat, no announcement line.
- LamQuant satisfies the envelope via aliases (above) split across
  `checkpoint_manager.py`/`durable_resume.py`; its `resume_key` guard predates §5's
  derivation; no announcement line.

## 5. Resume key

Pinned composition (every implementation, the Rust engine included, must byte-match):

```
config_hash = blake3_hex( canonical_json(config) )
resume_key  = blake3_hex( config_hash + 0x1F + data_digest + 0x1F + contract_version )
```

where `canonical_json` is EXACTLY Python's
`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`
(sorted keys, no whitespace, `\uXXXX`-escaped non-ASCII, shortest-round-trip float repr,
NaN/Inf rejected; `1` and `1.0` canonicalize differently — configs are single-sourced and
type-stable), 0x1F is the unit-separator byte, all pieces UTF-8, and `data_digest` is the
ABIR snapshot digest (LamQuant) or the dataset content hash (LLM cookbook). The engine
derives cache keys and resume admission from it; a trainer handed a checkpoint whose
stored resume key mismatches MUST refuse to resume. Reference:
`tritium.torch.contract.resume_key`.

## 6. RUN_MANIFEST

Written once per run (ULID id), stored by the engine in the ledger/lineage DB and
referenced by the registry entry: argv, canonical config hash, data digests (ABIR or
dataset hashes), tritium version + conformance-manifest hash, git SHAs of every involved
repo, torch/numpy/CUDA versions, GPU + driver, seed, final checkpoint SHA-256.

## 7. Golden fixtures

- `tests/contract/status_stream_v1.golden.jsonl` — a complete valid run stream exercising
  every kind, the announcement line, and interleaved `BLUT_METRIC` lines.
- `tests/contract/envelope_v1.example.json` — the envelope key schema with example values
  (JSON description; real envelopes are torch pickles).
- `scripts/contract_lint.py` — validates a stream (`--stream`) or statically checks a
  trainer source (`--source`) for REQUIRED/RECOMMENDED conformance.

The Rust engine's golden test (`tests/contract_golden.rs`) parses the fixture stream with
the engine's own `protocol.rs` reader — all six kinds including `heartbeat`, prefix
recognition for both `BLUT_*` channels, one-terminal-event ordering, and unknown-kind
rejection.

## 8. Mid-stage checkpoint slots (revision 1.1)

Without this section, an interrupted stage restarts from step 0: each attempt runs in a
fresh stage directory. A stage that opts in (`Stage::checkpoint_slot` returns `true`)
gets a **slot**, a directory that outlives the attempt, and its trainer resumes from
the newest checkpoint in it. The section is additive: a trainer that ignores the
variables below writes no marker, every attempt starts fresh, and nothing else changes.

### Where the slot is

`<root>/<tenant>/<stage>-<cache key>`. The cache key covers the stage's arguments,
schema, code fingerprint and the content of its inputs, so a changed config or input
gets a different slot and cannot resume a stale checkpoint. `<root>` is
`$BLUT_CHECKPOINTS_DIR`, or by default a sibling of the jobs directory named
`<jobs-dir-name>-checkpoints`; it is outside every job directory, so a requeued run,
which is a new job, finds the slot its predecessor used. Rank 0 must be able to reach
it; other ranks never read it.

### Environment the trainer receives

| variable | meaning |
|---|---|
| `BLUT_CHECKPOINT_DIR` | the slot; checkpoints and the marker go here |
| `BLUT_RESUME` | `1`: resume from the newest valid checkpoint; `0`: start at step 0 |
| `BLUT_RUN_ID` | the job id, copied into the marker's `run_id` |
| `BLUT_ATTEMPT_ID` | this attempt's identity, copied into the marker's `attempt_id` |

The stage forwards them with `CheckpointSlot::apply_to`. `srun` and `torchrun` pass the
environment on to every rank.

### What the trainer does

- Writes checkpoints atomically: temp file, fsync, rename, then fsync the directory. The
  envelope is §4's, with canonical keys. Keeps at least the previous checkpoint until the
  new one is durable.
- Under DDP, only rank 0 writes; the others receive the loaded state from rank 0.
- On `BLUT_RESUME=1`, loads the newest checkpoint that verifies and skips, saying why,
  any that is partial (missing or mismatched integrity record), unloadable, missing an
  envelope key, or written for another config or world size. If none verifies, it starts
  at step 0.
- Restores model, optimizer, schedule, RNG and the position in the data, so the step
  count and the loss continue rather than restart.
- Maintains the marker `state.json` (§3) with these fields: `status` (`running`),
  `run_id`, `pid` and `host` (`gethostname`) of the writing process, `heartbeat_unix`,
  `step` and `epoch` of the newest durable checkpoint, `attempt_id`, and
  `resumed_from_step` (the step it loaded, or `null`). It writes the marker atomically
  and refreshes the heartbeat at least every `HEARTBEAT_INTERVAL` seconds.

### What the engine does

For each local attempt it takes an exclusive, non-blocking lock on the slot (held until
the node finishes, released if the process dies). If another process holds it, the
attempt fails with `CheckpointBusy`. Otherwise it reads the marker and its own
`owner.json` record of the last attempt, and decides:

| marker | decision |
|---|---|
| absent, unparseable, or `status: finished` | fresh: the slot is cleared |
| `running`, writer abandoned, `step` set | resume |
| `running`, writer abandoned, no `step` | fresh |
| `running`, writer may be alive | refuse with `CheckpointBusy` |

The writer counts as abandoned when the last attempt belonged to the same engine
invocation (an in-process retry: the previous attempt has returned), when the last
attempt ran in the same Slurm job with a lower `SLURM_RESTART_COUNT` (Slurm requeued the
job), when the marker's `host` is this host and its `pid` no longer exists, or when the
heartbeat is at least `DEFAULT_STALE_AFTER_SECS` old. The engine records each step in the
job's `status.jsonl` as a `stage_checkpoint` event: `fresh`, `resume_offered` (with the
marker's step), `resumed` (with the step the trainer recorded), and after the output is
committed to the cache, `cleaned` (slot deleted) or `kept` (with the reason, e.g. a
failed cache insert). A slot whose stage failed is kept for the next attempt.

The engine never reads checkpoint files. Slots are not created for remote (P2P) attempts,
pipelined or speculative execution.

### Relation to ADR 0088 and `resume::decide_resume`

ADR 0088 planned mid-epoch resume through the `--resume`/`--resume-dir` flags and
`decide_resume`, which the executor consults only on a retry. That path is unchanged and
still serves stages that implement `Stage::resume_handle`. Slots differ in three ways:
the trainer is told through the environment rather than flags; the decision is made for
every attempt, so `plan resume` and a requeued job resume too, not only retries; and a
marker with this job's `run_id` is not taken as proof that its writer is gone.
`decide_resume` assumes that, which holds for a retry but not for a `plan resume` started
while a trainer orphaned by a killed engine is still running. `decide_slot` judges such a
marker by process liveness, Slurm restart count and heartbeat instead.

