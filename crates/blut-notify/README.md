# blut-notify

`blut-notify` is BLUT's out-of-process notification boundary. This first slice
owns the load-bearing custody rule: a `clinical`/`restricted` tenant or a
`Restricted` envelope may be handled by a local sink, but it cannot reach an
off-box sink. The policy runs before the sink receives the envelope.

Notification envelopes arrive as JSON on stdin, never as command-line payloads:

```bash
printf '%s' '{"tenant":"research/dev","data_class":"Internal","summary":"run complete"}' \
  | blut-notify --boundary off-box
```

The current executable sink is stdout, which is sufficient for process piping
and the custody acceptance gate. With `--boundary local`, the caller assumes
custody of stdout and must not pipe it to an off-box or uncontrolled log sink.
Webhook/SMTP/SLA sinks remain additive work under ADR 0094; each must implement
`NotifySink` and therefore pass through the same `deliver` chokepoint. This
sidecar does not constrain cookbook TUIs or their Ratatui architecture.
