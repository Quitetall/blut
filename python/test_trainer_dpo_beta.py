"""Regression test: DPO beta is threaded end-to-end to the trainer.

Audit fix (2026-06-10): dpo_train.rs used to `let _ = args.beta;`, dropping the
DPO temperature. It is now carried on TrainSpec.dpo_beta, serialized to the
trainer subprocess, and surfaced by trainer_dpo.py. This test invokes the
script the way the Rust backend does (spec JSON as argv[1]) and asserts the
received beta is echoed back.
"""

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER = os.path.join(_HERE, "trainer_dpo.py")


def _spec(beta):
    return json.dumps({
        "base_model": "org/name",
        "output_name": "out",
        "output_dir": "/tmp/x",
        "method": {"kind": "q_lora", "rank": 16, "alpha": 32},
        "dataset": {"kind": "jsonl_path", "path": "/tmp/x.jsonl"},
        "optimizer": "adamw_8bit",
        "lr": 2e-4, "epochs": 1, "batch_size": 1, "grad_accum": 8,
        "seq_len": 4096, "seed": 42, "quant": "Q4_K_M", "skip_convert": True,
        "dpo_beta": beta,
    })


def _last_json_lines(stdout):
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def test_self_check_echoes_received_beta():
    proc = subprocess.run(
        [sys.executable, _TRAINER, "--self-check", _spec(0.1)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    msgs = _last_json_lines(proc.stdout)
    betas = [m.get("dpo_beta") for m in msgs if "dpo_beta" in m]
    assert betas, f"no dpo_beta surfaced: {proc.stdout!r}"
    assert all(abs(b - 0.1) < 1e-6 for b in betas), betas


def test_stub_run_reports_beta_in_failure():
    # Non-self-check path is still a stub (non-zero exit) but must now name the
    # beta it received, proving the value reached Python.
    proc = subprocess.run(
        [sys.executable, _TRAINER, _spec(0.25)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    failed = [m for m in _last_json_lines(proc.stdout) if m.get("kind") == "failed"]
    assert failed, proc.stdout
    assert "0.25" in failed[0]["error"], failed[0]["error"]
