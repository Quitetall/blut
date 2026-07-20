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
name = "run-failed"
source = "status"
field = "kind"
equals = "failed"
sinks = ["ops"]
```

```bash
blut-notify --config ~/.blut/notify.toml
blut-notify --config ~/.blut/notify.toml --once   # smoke/CI
```

Every sink implements `NotifySink` and therefore passes through the same
`deliver` chokepoint. Restricted summaries may reach local sinks; they are
structurally refused before any off-box sink sees them. The legacy one-envelope
stdin mode remains available for process composition.
