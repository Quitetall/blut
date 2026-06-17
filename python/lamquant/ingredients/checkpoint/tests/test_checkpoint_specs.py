"""Tests for the checkpoint ingredient registry (ADR 0050/0051).

The three checkpoint specs are extracted verbatim from the trainers; these tests
pin (a) byte-equal atomic writes vs both a hand-rolled tmp+replace and the snn
trainer's own ``_atomic_torch_save``, (b) crash-safety, (c) the async path,
(d) the CheckpointManager guard wiring + atomic save, and (e) the durable-resume
None-gate + a save/detect/load round-trip.

Importing the spec module directly self-registers the specs (the package
``__init__.py`` does not yet import them), so we do not depend on Phase-4
registration order.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

# Self-register the checkpoint specs without touching ingredients/__init__.py.
import lamquant.ingredients.checkpoint._specs as _ckpt_specs  # noqa: F401
from lamquant.ingredients import build_ingredient, list_ingredients

pytestmark = pytest.mark.l2


# --------------------------------------------------------------------------
# (1) atomic_save
# --------------------------------------------------------------------------

def test_specs_registered():
    names = list_ingredients("checkpoint")
    assert {"atomic_save", "manager", "durable_resume"} <= set(names)


def test_atomic_save_sync_byte_equal_vs_handrolled(tmp_path):
    """Sync atomic_save produces a file byte-identical to a hand-rolled
    tmp+os.replace torch.save of the same payload.

    torch's checkpoint zip embeds the *archive name* (derived from the final
    file basename), so byte-equality only holds when both writes target the same
    final path through the same tmp pattern — which is exactly what the atomic
    helper does (``{path}.tmp.{pid}``). We therefore write the hand-rolled
    reference and the ingredient to the SAME final path and compare bytes."""
    import os
    payload = {"state_dict": {"w": torch.arange(12).float().reshape(3, 4)},
               "epoch": 7, "meta": "x"}
    save = build_ingredient("checkpoint", "atomic_save", {})

    p = tmp_path / "a.ckpt"

    # hand-rolled reference — identical tmp+os.replace pattern to the helper.
    tmp = f"{p}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, p)
    ref_bytes = p.read_bytes()

    # ingredient overwrites the same final path → same archive name → same bytes.
    save(payload, str(p))

    assert p.read_bytes() == ref_bytes
    # no leftover .tmp.<pid> file
    assert not list(tmp_path.glob("a.ckpt.tmp*"))


def test_atomic_save_equality_vs_snn_atomic_torch_save(tmp_path):
    """Byte-equal vs the snn trainer's own _atomic_torch_save (the source of the
    extraction)."""
    from lamquant.snn.train_4state_controller import _atomic_torch_save
    payload = {"state_dict": {"w": torch.randn(2, 3)}, "best_val_r": 0.5}

    # Same final path through both → same archive name + same tmp pattern → the
    # extraction is byte-identical to the snn trainer's own helper.
    p = tmp_path / "same.ckpt"
    _atomic_torch_save(payload, str(p))
    snn_bytes = p.read_bytes()
    build_ingredient("checkpoint", "atomic_save", {})(payload, str(p))

    assert p.read_bytes() == snn_bytes


def test_atomic_save_crash_safety_leaves_original(tmp_path, monkeypatch):
    """A failing torch.save must leave the original file intact (the tmp is the
    only casualty)."""
    p = tmp_path / "c.ckpt"
    good = {"v": torch.ones(3)}
    save = build_ingredient("checkpoint", "atomic_save", {})
    save(good, str(p))
    original = p.read_bytes()

    real_save = torch.save

    def boom(obj, f, *a, **k):
        if isinstance(f, str) and f.startswith(str(p)):
            raise RuntimeError("disk full mid-write")
        return real_save(obj, f, *a, **k)

    # _atomic_torch_save does `import torch` locally → it resolves the same
    # module-global torch, so patching torch.save reaches the helper.
    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        save({"v": torch.zeros(99)}, str(p))

    # original untouched, no leftover tmp
    assert p.read_bytes() == original
    assert not list(tmp_path.glob("c.ckpt.tmp*"))


def test_atomic_save_async_path(tmp_path):
    """async_ hands the write to the module-global executor; after shutdown the
    file exists and loads back to the payload."""
    payload = {"state_dict": {"w": torch.arange(6).float()}, "epoch": 3}
    save = build_ingredient("checkpoint", "atomic_save", {"async_": True})
    p = tmp_path / "async.ckpt"
    save(payload, str(p))

    # drain the single-worker executor deterministically
    import lamquant.ingredients.checkpoint._specs as mod
    assert mod._SAVE_EXECUTOR is not None
    mod._SAVE_EXECUTOR.shutdown(wait=True)
    # reset so the atexit handler + later builds re-create a fresh one
    mod._SAVE_EXECUTOR = None

    assert p.exists()
    loaded = torch.load(p, weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["state_dict"]["w"], payload["state_dict"]["w"])


def test_atomic_save_state_dict_to_cpu(tmp_path):
    """state_dict_to_cpu detaches + moves the state_dict to CPU (values preserved)."""
    payload = {"state_dict": {"w": torch.randn(4, requires_grad=True)}}
    save = build_ingredient("checkpoint", "atomic_save",
                            {"state_dict_to_cpu": True})
    p = tmp_path / "cpu.ckpt"
    save(payload, str(p))
    loaded = torch.load(p, weights_only=False)
    assert loaded["state_dict"]["w"].device.type == "cpu"
    assert not loaded["state_dict"]["w"].requires_grad


# --------------------------------------------------------------------------
# (2) manager
# --------------------------------------------------------------------------

class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def test_manager_returns_checkpoint_manager_with_guard():
    from lamquant.student.checkpoint_manager import (
        CheckpointManager, GuardConfig)
    m = _TinyModel()
    cm = build_ingredient(
        "checkpoint", "manager", {},
        model=m, ckpt_path="/tmp/_unused_cm.ckpt")
    assert type(cm) is CheckpointManager
    # guard field-equal to a hand-built default GuardConfig
    assert cm.guard == GuardConfig()


def test_manager_guard_overrides_field_equal():
    from lamquant.student.checkpoint_manager import GuardConfig
    cfg = {"r_plateau_patience": 10, "alpha_max_safe": 3.0,
           "alpha_min_safe": 2e-4, "smoke_check_tolerance": 5e-3,
           "improvement_eps": 2e-3, "prd_tiebreak_eps": 0.25}
    cm = build_ingredient("checkpoint", "manager", cfg,
                          model=_TinyModel(), ckpt_path="/tmp/_unused_cm2.ckpt")
    assert cm.guard == GuardConfig(**cfg)


def _strip_saved_at(d):
    """Drop the wall-clock timestamp so two saves of identical weights compare
    byte-equal."""
    d = dict(d)
    d.pop("saved_at", None)
    return d


def test_manager_save_atomic_byte_equal(tmp_path):
    """CheckpointManager._save_atomic writes the documented payload; two saves of
    the same model are byte-equal once the saved_at timestamp is dropped."""
    torch.manual_seed(0)
    m = _TinyModel()
    p1 = tmp_path / "m1.ckpt"
    p2 = tmp_path / "m2.ckpt"
    cm = build_ingredient("checkpoint", "manager", {},
                          model=m, ckpt_path=str(p1))
    cm._save_atomic(p1)
    cm._save_atomic(p2)

    a = _strip_saved_at(torch.load(p1, weights_only=False))
    b = _strip_saved_at(torch.load(p2, weights_only=False))
    # weights identical, metadata identical (sans timestamp)
    assert torch.equal(a["state_dict"]["lin.weight"], b["state_dict"]["lin.weight"])
    assert a["best_val_r"] == b["best_val_r"]
    assert a["best_epoch"] == b["best_epoch"]
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------
# (3) durable_resume
# --------------------------------------------------------------------------

def test_durable_resume_none_on_falsy():
    out = build_ingredient("checkpoint", "durable_resume", {}, resume_dir="")
    assert out is None
    out2 = build_ingredient("checkpoint", "durable_resume", {}, resume_dir=None)
    assert out2 is None


def test_durable_resume_round_trip(tmp_path):
    from lamquant.student.durable_resume import DurableResume
    dur = build_ingredient(
        "checkpoint", "durable_resume", {},
        resume_dir=str(tmp_path / "rdir"), run_id="r1", resume_key="k1")
    assert isinstance(dur, DurableResume)
    try:
        # A COMPLETE recovery checkpoint: carries the full REQUIRED_RECOVERY_KEYS
        # set (encoder/decoder/epoch/phase/optimizer; rng + resume_key are added
        # by save_recovery). An incomplete checkpoint is deliberately rejected on
        # load — see test_durable_resume_rejects_incomplete below.
        dur.save_recovery("qat_latest", {
            "epoch": 4,
            "phase": "qat",
            "encoder": {"w": torch.ones(2)},
            "decoder": {"w": torch.zeros(2)},
            "optimizer": {"state": {}},
        })
        assert dur.detect() == "qat_latest"
        ck = dur.load_recovery("qat_latest")
        assert ck is not None
        assert ck["epoch"] == 4
        assert ck["resume_key"] == "k1"
        assert torch.equal(ck["encoder"]["w"], torch.ones(2))
    finally:
        # finish() joins the daemon heartbeat thread so the test doesn't hang.
        dur.finish()


def test_durable_resume_rejects_incomplete(tmp_path):
    """An incomplete recovery checkpoint (missing required keys) must be REFUSED
    on load — never silently cold-started (Phase D required-keys validation)."""
    from lamquant.student.durable_resume import DurableResume
    dur = build_ingredient(
        "checkpoint", "durable_resume", {},
        resume_dir=str(tmp_path / "rdir"), run_id="r1", resume_key="k1")
    assert isinstance(dur, DurableResume)
    try:
        # Missing encoder/decoder/phase/optimizer.
        dur.save_recovery("qat_latest", {"epoch": 4})
        with pytest.raises(RuntimeError):
            dur.load_recovery("qat_latest")
    finally:
        dur.finish()


def test_durable_resume_config_is_empty_frozen_dataclass():
    cfg_cls = _ckpt_specs.DurableResumeConfig
    assert dataclasses.is_dataclass(cfg_cls)
    assert cfg_cls.__dataclass_params__.frozen
    assert not dataclasses.fields(cfg_cls)
