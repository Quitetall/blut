# blut-notify

`blut-notify` is BLUT's out-of-process notification boundary. It owns the
load-bearing custody rule: a `clinical`/`restricted` tenant or a
`Restricted` envelope may be handled by a local sink, but it cannot reach an
off-box sink. The policy runs before the sink receives the envelope.

Notification envelopes arrive as JSON on stdin, never as command-line payloads:

```bash
printf '%s' '{"tenant":"research/dev","data_class":"Internal","summary":"run complete"}' \
  | blut-notify --boundary off-box
```

With `--config`, the sidecar durably tails every job's `status.jsonl` (including
the rotated `.1` generation) plus `sla.jsonl`, matches declarative rules, and
routes them to Slack, Discord, ntfy, SMTP, or `exec`. Cursors are crash-safe and
single-writer locked; delivery is at-least-once. Network credentials are
`SecretRef` names resolved only at send time, never plaintext config fields.

```toml
[[sink]]
name = "ops"
kind = "slack"
webhook = { name = "BLUT_SLACK_WEBHOOK" }

[[rule]]
name = "stage-failed"
source = "status"
field = "kind"
equals = "stage_failed"
sinks = ["ops"]
```

`field` is a dotted path into one `status.jsonl` line. Stage lifecycle lines
carry `kind` = `stage_begin`, `stage_step`, `stage_end`, `stage_skipped`,
`stage_failed`, and so on; there is no `kind = "failed"` line, so a rule
matching `"failed"` never fires.

```bash
blut-notify --config ~/.blut/notify.toml
blut-notify --config ~/.blut/notify.toml --once   # smoke/CI
```

Without `--jobs-dir`, `--sla` and `--cursor` the daemon reads where the engine
writes (`blut::paths`): jobs from `$LAMU_TRAIN_JOBS_DIR`, else
`$LAMU_TRAIN_DATA_DIR/train-jobs`, else `$XDG_DATA_HOME/lamu/train-jobs`
(`~/.local/share/lamu/train-jobs`); the SLA log from `$BLUT_SLA_PATH`, else
`sla.jsonl` in `$LAMU_TRAIN_DATA_DIR` or `~/.blut`, where the cursors live too.

A status rule's summary names the job, the event kind, the stage, and the
event's error (or a pruned stage's reason) cut to 200 characters, for example
`rule 'stage-failed': status event for job 20261030-…: stage_failed in stage
'train' — CUDA out of memory`. Error text is free-form, so it is included only
when custody would let the event leave the box: for a restricted tenant or a
`restricted` line the summary ends `(error withheld: restricted data)` on every
sink, local ones included.

Every sink implements `NotifySink` and therefore passes through the same
`deliver` chokepoint. Restricted summaries may reach local sinks; they are
structurally refused before any off-box sink sees them. The legacy one-envelope
stdin mode remains available for process composition.
