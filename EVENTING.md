<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Eventing, SLA, and notifications

BLUT keeps the engine socket-free. `blut-web` authenticates webhook requests and
spools files; `blut sensord` reads those files plus local spool/cron sources and
launches the cookbook CLI; `blut-notify` independently tails status and SLA
streams. Every triggered recipe still acquires its exact tenant reservation and
scheduler lock through the ordinary recipe path before sensord records the event
as admitted.

## Trigger and webhook configuration

Both sensord and blut-web read the same `triggers.toml` (by default
`~/.blut/triggers.toml`, or `$BLUT_TRIGGERS`). An explicitly configured missing
or invalid file is an error.

```toml
webhook_max_skew_secs = 300
admission_timeout_secs = 60

[[trigger]]
name = "incoming-eval"
kind = "file-drop"
dir = "/srv/blut/events/incoming-eval"
ext = "json"
plan = "registry://plan@eval-prod"
tenant = "research/prod"
data_class = "Internal"
webhook_secret = { name = "BLUT_EVAL_WEBHOOK_KEY" }

[[trigger]]
name = "queue-full"
kind = "spool"
dir = "/srv/blut/queue"
spool_metric = "files"       # files | bytes
spool_direction = "at_least" # at_least | at_most
threshold = 100
plan = "drain_queue"

[[trigger]]
name = "nightly"
kind = "cron"
schedule = "0 0 2 * * * *" # sec min hour day month weekday year
grace_secs = 60
plan = "registry://plan@nightly"
```

Run sensord with the cookbook binary that owns the configured recipes:

```bash
blut sensord --cli /usr/local/bin/my-cookbook
blut sensord --cli /usr/local/bin/my-cookbook --once # deterministic smoke
```

Webhook clients send `POST /events/<trigger>` (the `/api/events/<trigger>`
alias is also supported) with:

- `X-Blut-Timestamp`: current Unix seconds;
- `X-Blut-Signature`: `sha256=<hex HMAC-SHA256>` over
  `<timestamp>.<exact raw request body>`;
- a JSON object body.

The credential value comes from the environment key named by `webhook_secret`.
Deploy public ingress behind HTTPS; BLUT validates the HMAC and bounded timestamp
inside that transport boundary. Identical bodies within the same trigger and
custody binding are content-addressed to one file and do not get rewritten on
replay. Use the ADR 0095 token store as well when binding blut-web outside
loopback.

Restricted triggers must declare `data_class = "Restricted"`, target the exact
restricted tenant, and remain on the same box. A restricted `SecretRef` cannot
resolve in a sidecar; use an ingress credential whose custody policy permits the
web boundary while keeping the event itself Restricted.

## SLA rules

Sensord and `blut sla check` call the same evaluator. Rules default to
`~/.blut/sla.toml` or `$BLUT_SLA_RULES`; breaches append once to
`$BLUT_SLA_PATH` or `~/.blut/sla.jsonl`.

```toml
[[rule]]
name = "training-runtime"
recipe = "train"
max_runtime_secs = 14400

[[rule]]
name = "fresh-output"
recipe = "evaluate"
freshness_secs = 3600
```

`max_runtime_secs`, `deadline_unix`, and `freshness_secs` are positive bounds.
Freshness uses the newest measured `produced_unix` lineage timestamp and fails
closed when a matching run has no measurement. `blut sla check` exits nonzero
while any breach is active, even if its durable row was already written.

## Notification rules

`blut-notify` tails complete lines from every `status.jsonl` generation and
`sla.jsonl`. Its cursor is atomically persisted and single-writer locked.
Delivery is at-least-once: a crash or sink failure can replay a line, but cannot
silently skip an undelivered line. Configure downstream sinks idempotently.

```toml
[[sink]]
name = "ops"
kind = "slack"
webhook = { name = "BLUT_SLACK_WEBHOOK" }

[[sink]]
name = "pager"
kind = "ntfy"
server = "https://ntfy.example.com"
topic = "blut-ops"
token = { name = "BLUT_NTFY_TOKEN" }

[[rule]]
name = "run-failed"
source = "status"
field = "kind"
equals = "failed"
sinks = ["ops", "pager"]

[[rule]]
name = "runtime-breach"
source = "sla"
field = "kind"
equals = "max_runtime"
sinks = ["ops"]
```

```bash
blut-notify --config ~/.blut/notify.toml
blut-notify --config ~/.blut/notify.toml --once
```

Slack, Discord, ntfy, and SMTP are always off-box boundaries. `exec` defaults
to local custody and receives the envelope on stdin, never argv. Restricted
envelopes are refused before any off-box sink sees them; summaries contain only
operational identifiers, not the original status payload or error text.

## Operations and failure semantics

- Supervise sensord, blut-web, and blut-notify as separate services and give
  each only the filesystem and credential access it needs.
- Run one notifier per cursor file. A second process refuses the exclusive lock.
- Preserve the event spool, `.seen`/`.admitted` markers, `sla.jsonl`, and the
  notifier cursor across restarts.
- Monitor daemon exit status and stderr. Invalid configuration, unmeasured
  configured freshness, admission timeout, and sink delivery failure are loud.
- Treat notification endpoints as at-least-once consumers and use provider-side
  dedupe where duplicate alerts are operationally costly.
