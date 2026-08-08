#!/usr/bin/env python3
"""Trainer-contract v1 lint (CONTRACT.md).

Two modes:
  --stream <file>   validate a recorded stdout stream line by line
  --source <file>   static REQUIRED/RECOMMENDED conformance check of a trainer script

Exit code: 0 = all REQUIRED checks pass (RECOMMENDED misses are reported as
deviations), 1 = a REQUIRED check failed, 2 = usage error.

Stdlib only — this runs in both repos' CIs and on bare checkouts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CONTRACT_PREFIX = "BLUT_CONTRACT "
METRIC_PREFIX = "BLUT_METRIC "

# kind -> {field: type}; types checked as isinstance tuples.
_NUM = (int, float)
KINDS: dict[str, dict[str, tuple]] = {
    "step": {"step": (int,), "total": (int,), "loss": _NUM, "lr": _NUM, "vram_mb": (int,)},
    "eval": {"step": (int,), "eval_loss": _NUM},
    "saved": {"path": (str,)},
    "done": {"final_loss": _NUM, "checkpoint_dir": (str,)},
    "failed": {"error": (str,)},
    "heartbeat": {},  # phase/vram_mb optional
}
TERMINAL = {"done", "failed"}

ENVELOPE_REQUIRED = ["model", "opt", "config", "step", "rng"]
ENVELOPE_OPTIONAL = ["ema", "sched", "manifest_ref"]
# CONTRACT.md §4 v1 aliases: readers accept these; writers should emit canonical.
ENVELOPE_ALIASES = {
    "model": ["state_dict"],
    "opt": ["optimizer"],
    "config": ["training_config_hash"],
    "rng": ["rng_state", "get_rng_state", "getstate()"],
}


class Report:
    def __init__(self) -> None:
        self.required_failures: list[str] = []
        self.deviations: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.required_failures.append(msg)

    def deviate(self, msg: str) -> None:
        self.deviations.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def finish(self, subject: str) -> int:
        for n in self.notes:
            print(f"note      {subject}: {n}")
        for d in self.deviations:
            print(f"DEVIATION {subject}: {d}")
        for f in self.required_failures:
            print(f"REQUIRED-FAIL {subject}: {f}")
        verdict = "FAIL" if self.required_failures else "PASS"
        extra = f" ({len(self.deviations)} documented deviation(s))" if self.deviations else ""
        print(f"contract-lint: {verdict} {subject}{extra}")
        return 1 if self.required_failures else 0


def lint_stream(path: Path) -> int:
    r = Report()
    lines = path.read_text().splitlines()
    saw_contract = False
    terminal_at: int | None = None
    control_lines = 0

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line.startswith(CONTRACT_PREFIX):
            version = line[len(CONTRACT_PREFIX):].strip()
            if version != "1":
                r.fail(f"line {i}: announced contract version {version!r}, expected '1'")
            if control_lines:
                r.deviate(f"line {i}: BLUT_CONTRACT after control lines (should be first)")
            saw_contract = True
            continue
        if line.startswith(METRIC_PREFIX):
            try:
                payload = json.loads(line[len(METRIC_PREFIX):])
            except json.JSONDecodeError as e:
                r.fail(f"line {i}: BLUT_METRIC payload is not JSON ({e})")
                continue
            if "kind" not in payload:
                r.fail(f"line {i}: BLUT_METRIC payload missing 'kind'")
            bad = {
                k: v for k, v in payload.items()
                if k not in ("kind", "phase")
                and (isinstance(v, bool) or not isinstance(v, _NUM))
            }
            if bad:
                r.fail(f"line {i}: BLUT_METRIC non-numeric metric values {sorted(bad)}")
            continue
        # Control channel.
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Readers tolerate+log malformed lines; the *golden* stream must not have any.
            r.fail(f"line {i}: unparseable control line")
            continue
        kind = obj.get("kind")
        if kind not in KINDS:
            r.fail(f"line {i}: unknown kind {kind!r}")
            continue
        if terminal_at is not None:
            r.fail(f"line {i}: control line after terminal event (line {terminal_at})")
        control_lines += 1
        for field, types in KINDS[kind].items():
            v = obj.get(field)
            if isinstance(v, bool) or not isinstance(v, types):
                r.fail(f"line {i}: kind={kind} field {field!r} missing or mistyped ({v!r})")
        if kind in TERMINAL:
            terminal_at = i

    if terminal_at is None:
        r.fail("stream has no terminal event (done|failed)")
    if not saw_contract:
        r.deviate("no BLUT_CONTRACT announcement line (v0-legacy stream)")
    return r.finish(str(path))


def lint_source(paths: list[Path]) -> int:
    """Static conformance check of a trainer MODULE SET (the entry script plus
    its checkpoint/resume companions — LamQuant splits the envelope across
    checkpoint_manager.py / durable_resume.py). Heuristic by design: it
    verifies the observable contract markers exist in the sources, not that
    the runtime behaves — the stream lint covers behavior."""
    r = Report()
    src = "\n".join(p.read_text() for p in paths)
    path = paths[0]

    def has(pattern: str) -> bool:
        return re.search(pattern, src) is not None

    # REQUIRED: emits kind-tagged control lines or the BLUT_METRIC channel.
    emits_control = has(r"[\"']kind[\"']\s*[:=]") and has(r"json\.dumps|dumps\(")
    emits_metric = METRIC_PREFIX.strip() in src
    if not (emits_control or emits_metric):
        r.fail("no status emission found (neither kind-tagged JSON nor BLUT_METRIC)")
    elif emits_control and not emits_metric:
        r.note("control channel only (no BLUT_METRIC observability lines)")
    elif emits_metric and not emits_control:
        r.note("BLUT_METRIC only (control channel presumably via runner wrapper)")

    # REQUIRED: a terminal failure path (failed event or non-zero exit on error).
    if not (has(r"[\"']failed[\"']") or has(r"sys\.exit\(") or has(r"raise SystemExit")):
        r.fail("no failure path found (no 'failed' event and no non-zero exit)")

    # REQUIRED: checkpoints carry model+opt+config+step (canonical or v1 alias).
    def envelope_key_present(key: str) -> tuple[bool, bool]:
        """(present, via_alias)"""
        if has(rf"[\"']{key}[\"']"):
            return True, False
        for alias in ENVELOPE_ALIASES.get(key, []):
            if re.escape(alias) != alias:
                if alias in src:
                    return True, True
            elif has(rf"[\"']{alias}[\"']|{alias}"):
                return True, True
        return False, False

    if has(r"torch\.save|\.save\("):
        for key in ("model", "opt", "config", "step"):
            present, via_alias = envelope_key_present(key)
            if not present:
                r.fail(f"checkpoint save present but envelope key {key!r} not found")
            elif via_alias:
                r.note(f"envelope key {key!r} satisfied via v1 alias")
    else:
        r.note("no checkpoint save found (eval-only trainer?)")

    # v1 REQUIRED (grandfathered as deviation for pre-contract trainers): rng in envelope.
    present, via_alias = envelope_key_present("rng")
    if not present:
        r.deviate("envelope missing 'rng' state (CONTRACT.md §4 v1 requirement)")
    elif via_alias:
        r.note("envelope key 'rng' satisfied via v1 alias")

    # RECOMMENDED markers.
    if not has(r"BLUT_CONTRACT"):
        r.deviate("no BLUT_CONTRACT announcement (RECOMMENDED)")
    if not has(r"heartbeat|HEARTBEAT"):
        r.deviate("no heartbeat emission (in-band kind or state.json heartbeat_unix)")
    if not has(r"os\.replace|\.tmp|rename"):
        r.deviate("checkpoint save may not be atomic (no tmp+rename pattern found)")

    return r.finish(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stream", type=Path, help="recorded stdout stream to validate")
    g.add_argument(
        "--source",
        type=Path,
        nargs="+",
        help="trainer module set: entry script plus checkpoint/resume companions",
    )
    args = ap.parse_args()
    for target in [args.stream] if args.stream else args.source:
        if not target.is_file():
            print(f"contract-lint: no such file: {target}", file=sys.stderr)
            return 2
    return lint_stream(args.stream) if args.stream else lint_source(args.source)


if __name__ == "__main__":
    sys.exit(main())
