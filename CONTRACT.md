# BLUT Trainer Contract v1

Frozen 2026-08-08 (ADR 0037 Stage 0). This is the wire contract between the orchestrator
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
