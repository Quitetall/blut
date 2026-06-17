"""Logging ingredients (ADR 0050/0051) — the per-epoch metric SINK a trainer's
run() emits to.

One spec, ``kind="logging"``:

  * ``blut_metric`` — the runner-parseable ``BLUT_METRIC <json>`` stdout line
                      (``train_joint``'s ``_emit`` S6/P4 contract).
"""
