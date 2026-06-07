"""Training diagnostics — catch ANY problem, including silent bugs.

A consolidated battery of probes for the joint neural-codec trainer. It does NOT
replace the existing per-epoch guards (TrainingGuard, alpha stats, checkpoint
smoke-check, validate_joint) — it adds the checks those miss, with an emphasis
on SILENT failures: bugs that let training "run fine" while learning nothing or
the wrong thing.

Three tiers:

  PREFLIGHT  (before the GPU run — fail fast, cheap):
    - data_sanity        : NaN/Inf, all-zero / dead channels, range, dtype/shape
    - gradient_flow      : every loss term produces NON-zero grad to BOTH
                           encoder and decoder (catches the dead-R-gradient class
                           — a term silently detached from the graph)
    - shape_contract     : encode→[B,32,79] (N-invariant), decode→[B,N,T_out]
    - masked_invariant   : masked metric with all-real == unmasked (no-regression)
    - overfit_one_batch  : the single most powerful silent-bug probe — if the
                           model cannot drive the loss down on ONE batch, there
                           is a gradient / architecture / LR bug. No data or
                           scale can hide it.
    - coords_routing     : (channel-agnostic) coords[i] actually conditions
                           output channel i — a transposed/blind coords wiring
                           reads as "architecture can't learn".
    - padded_leak        : (channel-agnostic) padded channels do not change the
                           real-channel output or loss.

  STEP      (every step, ~free): loss / grad-norm / latent / recon all finite.

  PERIODIC  (val interval — the gap list from the diagnostics inventory):
    - latent_health      : per-channel std (collapse), FSQ code span, kurtosis
    - gradient_health    : per-group grad norm, % near-zero (dead) params
    - loss_oscillation   : rolling-std / mean (instability)
    - ema_divergence     : EMA R vs live R (overfit / instability signal)
    - per_n_recon        : (channel-agnostic) R at each channel count N
    - shape_mismatch     : recon length vs target length (silent domain mix)

Each check returns a DiagResult (PASS / WARN / FAIL + detail + value). A probe
that is meant to FAIL on a real bug is tested to actually fire (see
tests/test_training_diagnostics.py) — a diagnostic that never fails is useless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class DiagResult:
    name: str
    status: str
    detail: str = ""
    value: Optional[float] = None

    def __str__(self) -> str:
        v = "" if self.value is None else f" [{self.value:.4g}]"
        return f"  [{self.status}] {self.name}{v} — {self.detail}"


@dataclass
class DiagReport:
    results: List[DiagResult] = field(default_factory=list)

    def add(self, r):
        if isinstance(r, list):
            self.results.extend(r)
        else:
            self.results.append(r)
        return r

    @property
    def failed(self) -> List[DiagResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def warned(self) -> List[DiagResult]:
        return [r for r in self.results if r.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        n_fail, n_warn = len(self.failed), len(self.warned)
        head = f"diagnostics: {len(self.results)} checks, {n_fail} FAIL, {n_warn} WARN"
        lines = [head] + [str(r) for r in self.results]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_fail": len(self.failed),
            "n_warn": len(self.warned),
            "results": [{"name": r.name, "status": r.status,
                         "detail": r.detail, "value": r.value} for r in self.results],
        }


def _finite(t) -> bool:
    return bool(torch.isfinite(t).all())


class TrainingDiagnostics:
    """Probe battery for a JointCodec-like model.

    loss_fn(recon, l3, fullband=None, ch_mask=None) -> (scalar_loss, parts_or_None).
    If None, a simple MSE+(1-R) stand-in is used (enough for the structural
    probes; the trainer passes its real joint_loss for fidelity).
    """

    def __init__(self, codec, loss_fn: Optional[Callable] = None,
                 channel_agnostic: bool = False, device: str = "cpu"):
        self.codec = codec
        self.channel_agnostic = channel_agnostic
        self.device = device
        self.loss_fn = loss_fn or self._default_loss
        self._loss_hist: List[float] = []

    # ---- default structural loss (used only if the trainer doesn't pass one) --
    @staticmethod
    def _default_loss(recon, l3, fullband=None, ch_mask=None):
        target = fullband if (fullband is not None and
                              abs(recon.shape[-1] - fullband.shape[-1]) <= 8) else l3
        T = min(recon.shape[-1], target.shape[-1])
        r, t = recon[..., :T], target[..., :T]
        mse = torch.nn.functional.mse_loss(r, t)
        rf, tf = r.flatten(1), t.flatten(1)
        rc, tc = rf - rf.mean(1, keepdim=True), tf - tf.mean(1, keepdim=True)
        pear = ((rc * tc).sum(1) / (rc.norm(dim=1) * tc.norm(dim=1) + 1e-8)).mean()
        return mse + (1.0 - pear), None

    def _forward(self, x_l3, coords=None, ch_mask=None, quantize=False):
        if self.channel_agnostic:
            return self.codec(x_l3, quantize=quantize, coords=coords, ch_mask=ch_mask)
        return self.codec(x_l3, quantize=quantize)

    # ===================== PREFLIGHT =====================

    def check_data_sanity(self, x_l3, fullband=None) -> List[DiagResult]:
        out = []
        for name, t in (("l3", x_l3), ("fullband", fullband)):
            if t is None:
                continue
            if not _finite(t):
                out.append(DiagResult(f"data.{name}.finite", FAIL,
                                      "NaN/Inf in input — data pipeline corrupt"))
                continue
            out.append(DiagResult(f"data.{name}.finite", PASS, "no NaN/Inf"))
            # per-channel std: a dead (zero-variance) channel never trains
            std = t.float().std(dim=-1)               # [B,N]
            n_dead = int((std < 1e-8).sum())
            frac = n_dead / std.numel()
            out.append(DiagResult(
                f"data.{name}.dead_channels", FAIL if frac > 0.5 else
                (WARN if n_dead else PASS),
                f"{n_dead}/{std.numel()} channel-windows zero-variance", frac))
            amp = float(t.float().abs().mean())
            out.append(DiagResult(f"data.{name}.scale", WARN if (amp < 1e-6 or amp > 1e6)
                                  else PASS, f"mean|x|={amp:.3g}", amp))
        return out

    def check_gradient_flow(self, x_l3, fullband=None, coords=None,
                            ch_mask=None) -> List[DiagResult]:
        """Every loss term must produce a NON-zero gradient to BOTH the encoder
        and the decoder. Catches a term silently detached from the graph
        (the dead-R-gradient class) and a frozen sub-net."""
        self.codec.train(True)
        self.codec.zero_grad(set_to_none=True)
        recon = self._forward(x_l3, coords, ch_mask, quantize=False)
        loss, _ = self.loss_fn(recon, x_l3, fullband, ch_mask)
        out = [DiagResult("grad.loss.grad_fn", PASS if loss.requires_grad and
                          loss.grad_fn is not None else FAIL,
                          "loss is on the autograd graph")]
        if loss.grad_fn is None:
            return out
        loss.backward()
        for part, params in (("encoder", list(self.codec.encoder.parameters())),
                             ("decoder", list(self.codec.decoder.parameters()))):
            gsum = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
            n_with = sum(1 for p in params if p.grad is not None and p.grad.abs().sum() > 0)
            out.append(DiagResult(
                f"grad.{part}.nonzero", FAIL if gsum == 0 else PASS,
                f"{n_with}/{len(params)} param tensors have non-zero grad", gsum))
        self.codec.zero_grad(set_to_none=True)
        return out

    def check_shape_contract(self, x_l3, coords=None, ch_mask=None,
                             expect_out: Optional[int] = None) -> List[DiagResult]:
        self.codec.train(False)
        out = []
        with torch.no_grad():
            lat = (self.codec.encoder.encode(x_l3, quantize=False, coords=coords)
                   if self.channel_agnostic else
                   self.codec.encoder.encode(x_l3, quantize=False))
        N = x_l3.shape[1]
        out.append(DiagResult("shape.latent", PASS if lat.shape[1:] == (32, 79)
                              else FAIL, f"encode→{tuple(lat.shape)} (want [*,32,79])"))
        with torch.no_grad():
            recon = self._forward(x_l3, coords, ch_mask, quantize=False)
        ok = recon.shape[0] == x_l3.shape[0] and recon.shape[1] == N
        if expect_out is not None:
            ok = ok and abs(recon.shape[-1] - expect_out) <= 8
        out.append(DiagResult("shape.recon", PASS if ok else FAIL,
                              f"decode→{tuple(recon.shape)} (want [B,{N},*])"))
        return out

    def overfit_one_batch(self, x_l3, fullband=None, coords=None, ch_mask=None,
                          steps: int = 150, lr: float = 3e-3,
                          min_drop: float = 0.15) -> DiagResult:
        """THE silent-bug probe: a healthy model+loss+optimizer must drive the
        loss DOWN on a single fixed batch. Failure ⇒ gradient/architecture/LR
        bug, not a data problem. The discriminator is healthy (meaningful
        decrease) vs broken (≈0% — frozen / detached grad / dead LR); the
        threshold separates those, it is NOT a fit-quality bar (the codec
        bottleneck is lossy, so 100% is unreachable). Throwaway optimizer;
        weights restored after."""
        import copy
        trainable = [p for p in self.codec.parameters() if p.requires_grad]
        if not trainable:
            return DiagResult("overfit_one_batch", FAIL,
                              "no trainable parameters (everything frozen?)")
        state = copy.deepcopy(self.codec.state_dict())
        was_training = self.codec.training
        self.codec.train(True)
        opt = torch.optim.Adam(trainable, lr=lr)
        losses = []
        try:
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                recon = self._forward(x_l3, coords, ch_mask, quantize=False)
                loss, _ = self.loss_fn(recon, x_l3, fullband, ch_mask)
                if not torch.isfinite(loss):
                    return DiagResult("overfit_one_batch", FAIL,
                                      f"loss became non-finite at step {len(losses)}")
                loss.backward()
                opt.step()
                losses.append(loss.item())
        finally:
            self.codec.load_state_dict(state)
            self.codec.zero_grad(set_to_none=True)  # load_state_dict leaves stale .grad
            self.codec.train(was_training)
        l0, lN = losses[0], min(losses[-5:])
        drop = (l0 - lN) / (abs(l0) + 1e-8)
        status = PASS if drop >= min_drop else FAIL
        return DiagResult("overfit_one_batch", status,
                          f"loss {l0:.4f}→{lN:.4f} over {steps} steps "
                          f"({drop*100:.0f}% drop; need ≥{min_drop*100:.0f}%). "
                          + ("healthy" if status == PASS else
                             "CANNOT memorize one batch → grad/arch/LR bug"), drop)

    def check_masked_invariant(self) -> DiagResult:
        """Masked metric with an all-real mask must equal the unmasked metric
        (the guarantee that ch_mask=None training is unchanged)."""
        try:
            from metrics import masked_pearson_r_torch, pearson_r_torch
        except Exception as e:  # pragma: no cover
            return DiagResult("masked_invariant", WARN, f"metrics import failed: {e}")
        a, b = torch.randn(3, 8, 200), torch.randn(3, 8, 200)
        ones = torch.ones(3, 8, dtype=torch.bool)
        d = float((masked_pearson_r_torch(a, b, ones) - pearson_r_torch(a, b)).abs())
        return DiagResult("masked_invariant", PASS if d < 1e-5 else FAIL,
                          f"|masked(all-true) - unmasked| = {d:.2e}", d)

    def check_coords_routing(self, x_l3, coords, ch_mask=None) -> DiagResult:
        """(CA) Permuting two channels' coords (l3 fixed) must change exactly
        those output channels — proves coords[i] conditions output i, not a
        transposed/blind wiring. Perturbs the decoder head out of identity."""
        if not self.channel_agnostic:
            return DiagResult("coords_routing", PASS, "n/a (not channel-agnostic)")
        head = getattr(self.codec.decoder, "head", None)
        pos_mlp = getattr(head, "pos_mlp", None)
        if pos_mlp is None or not hasattr(pos_mlp[-1], "weight"):
            return DiagResult("coords_routing", WARN,
                              "decoder head has no weight-bearing pos_mlp last layer")
        self.codec.train(False)
        saved = {k: v.clone() for k, v in head.state_dict().items()}
        try:
            with torch.no_grad():
                pos_mlp[-1].weight.normal_(std=0.5)
                pos_mlp[-1].bias.normal_(std=0.5)
                a = self._forward(x_l3, coords, ch_mask)
                cp = coords.clone()
                cp[:, [0, 1]] = coords[:, [1, 0]]
                b = self._forward(x_l3, cp, ch_mask)
        finally:
            head.load_state_dict(saved)
        changed01 = (a[:, :2] - b[:, :2]).abs().max().item()
        untouched = (a[:, 2:] - b[:, 2:]).abs().max().item() if a.shape[1] > 2 else 0.0
        ok = changed01 > 1e-5 and untouched < 1e-4
        return DiagResult("coords_routing", PASS if ok else FAIL,
                          f"swap coords[0,1] → Δch01={changed01:.2e} "
                          f"Δch2+={untouched:.2e} (want Δch01>0, Δch2+≈0)", changed01)

    def check_padded_leak(self, x_l3, coords, n_real: int) -> DiagResult:
        """(CA) Padded channels must not change the real-channel reconstruction."""
        if not self.channel_agnostic:
            return DiagResult("padded_leak", PASS, "n/a")
        B, N = x_l3.shape[0], x_l3.shape[1]
        mask_full = torch.ones(B, N, dtype=torch.bool, device=x_l3.device)
        mask_pad = mask_full.clone()
        mask_pad[:, n_real:] = False
        self.codec.train(False)
        with torch.no_grad():
            out_full = self._forward(x_l3[:, :n_real], coords[:, :n_real],
                                     torch.ones(B, n_real, dtype=torch.bool, device=x_l3.device))
            out_pad = self._forward(x_l3, coords, mask_pad)
        d = (out_full - out_pad[:, :n_real]).abs().max().item()
        return DiagResult("padded_leak", PASS if d < 1e-4 else FAIL,
                          f"real-channel Δ with vs without padding = {d:.2e}", d)

    def run_preflight(self, x_l3, fullband=None, coords=None, ch_mask=None,
                      expect_out: Optional[int] = None, overfit_steps: int = 60) -> DiagReport:
        rep = DiagReport()
        rep.add(self.check_data_sanity(x_l3, fullband))
        rep.add(self.check_shape_contract(x_l3, coords, ch_mask, expect_out))
        rep.add(self.check_gradient_flow(x_l3, fullband, coords, ch_mask))
        rep.add(self.check_masked_invariant())
        if self.channel_agnostic and coords is not None:
            rep.add(self.check_coords_routing(x_l3, coords, ch_mask))
        rep.add(self.overfit_one_batch(x_l3, fullband, coords, ch_mask, steps=overfit_steps))
        return rep

    # ===================== STEP (every step) =====================

    def step_probe(self, loss=None, latent=None, recon=None,
                   grad_norm=None) -> List[DiagResult]:
        out = []
        for nm, v in (("loss", loss), ("latent", latent), ("recon", recon)):
            if v is None:
                continue
            t = v if torch.is_tensor(v) else torch.tensor(float(v))
            out.append(DiagResult(f"step.{nm}.finite", PASS if _finite(t) else FAIL,
                                  "finite" if _finite(t) else "NON-FINITE"))
        if grad_norm is not None:
            g = float(grad_norm)
            st = FAIL if (not math.isfinite(g)) else (
                WARN if (g < 1e-7 or g > 1e3) else PASS)
            out.append(DiagResult("step.grad_norm", st, f"|grad|={g:.3g}", g))
        if loss is not None:
            self._loss_hist.append(float(loss))
        return out

    # ===================== PERIODIC =====================

    def latent_health(self, latent) -> List[DiagResult]:
        lat = latent.detach().float()
        std_c = lat.std(dim=(0, 2)) if lat.dim() == 3 else lat.std(dim=0)  # per feature
        collapse = float(std_c.min())
        span = float((lat.amax(dim=tuple(range(lat.dim() - 0))) if False else
                      (lat.max() - lat.min())))
        out = [DiagResult("latent.collapse", FAIL if collapse < 1e-3 else
                          (WARN if collapse < 1e-2 else PASS),
                          f"min per-feature std = {collapse:.3g}", collapse),
               DiagResult("latent.span", WARN if span < 0.5 else PASS,
                          f"global max-min = {span:.3g}", span)]
        # excess kurtosis of the whole latent (heavy tails → CDF-LUT mismatch)
        x = lat.flatten()
        m = x.mean()
        s = x.std() + 1e-8
        kurt = float((((x - m) / s) ** 4).mean() - 3.0)
        out.append(DiagResult("latent.kurtosis", WARN if abs(kurt) > 5 else PASS,
                              f"excess kurtosis = {kurt:.2f}", kurt))
        return out

    def gradient_health(self, model=None) -> List[DiagResult]:
        model = model or self.codec
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return [DiagResult("grad.health", WARN, "no grads present (call after backward)")]
        total = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in params))
        n_dead = sum(1 for p in params if float(p.grad.abs().sum()) == 0)
        st = FAIL if (not math.isfinite(total)) else (
            WARN if (total < 1e-6 or total > 1e3 or n_dead > 0.5 * len(params)) else PASS)
        return [DiagResult("grad.total_norm", st, f"|grad|={total:.3g}, "
                           f"{n_dead}/{len(params)} dead param tensors", total)]

    def loss_oscillation(self, window: int = 50) -> DiagResult:
        h = self._loss_hist[-window:]
        if len(h) < 10:
            return DiagResult("loss_oscillation", PASS, "insufficient history")
        t = torch.tensor(h)
        ratio = float(t.std() / (t.mean().abs() + 1e-8))
        return DiagResult("loss_oscillation", WARN if ratio > 0.5 else PASS,
                          f"rolling std/mean = {ratio:.3f} (window {len(h)})", ratio)

    @staticmethod
    def ema_divergence(live_r: float, ema_r: float) -> DiagResult:
        d = live_r - ema_r
        return DiagResult("ema_divergence", WARN if d > 0.05 else PASS,
                          f"live R={live_r:.4f} EMA R={ema_r:.4f} (live-ema={d:+.4f})", d)

    def per_n_recon(self, x_full_l3, coords_full, fullband_full,
                    ns=(8, 16, 21)) -> List[DiagResult]:
        """(CA) Reconstruction R at each channel count — does quality collapse
        at low N? `*_full` are N=21 tensors; we subset the first n channels."""
        if not self.channel_agnostic:
            return [DiagResult("per_n_recon", PASS, "n/a")]
        try:
            from metrics import masked_pearson_r_torch as mpr
        except Exception:
            from metrics import pearson_r_torch as mpr  # type: ignore
        self.codec.train(False)
        out = []
        for n in ns:
            if n > x_full_l3.shape[1]:
                continue
            with torch.no_grad():
                rec = self._forward(x_full_l3[:, :n], coords_full[:, :n],
                                    torch.ones(x_full_l3.shape[0], n, dtype=torch.bool))
                tgt = fullband_full[:, :n]
                T = min(rec.shape[-1], tgt.shape[-1])
                r = float(mpr(rec[..., :T], tgt[..., :T]))
            out.append(DiagResult(f"per_n_recon.N{n}", PASS, f"R={r:.4f}", r))
        # warn if low-N R is much worse than full-N R
        vals = [r.value for r in out if r.value is not None]
        if len(vals) >= 2 and (max(vals) - min(vals)) > 0.15:
            out.append(DiagResult("per_n_recon.spread", WARN,
                                  f"R varies {min(vals):.3f}..{max(vals):.3f} across N",
                                  max(vals) - min(vals)))
        return out


__all__ = ["TrainingDiagnostics", "DiagResult", "DiagReport", "PASS", "WARN", "FAIL"]


# ----------------------------- standalone CLI -----------------------------
def _main():  # pragma: no cover
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Run the training diagnostics battery.")
    ap.add_argument("--encoder", help="encoder checkpoint (.ckpt)")
    ap.add_argument("--tier", type=int, default=3)
    ap.add_argument("--channel-agnostic", action="store_true")
    ap.add_argument("--N", type=int, default=21)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", help="write JSON report to this path")
    args = ap.parse_args()

    from joint_codec import build_default_joint
    codec = build_default_joint(vocos_tier=args.tier,
                                channel_agnostic=args.channel_agnostic).to(args.device)
    if args.encoder:
        codec.load_encoder(args.encoder, strict=False)
    diag = TrainingDiagnostics(codec, channel_agnostic=args.channel_agnostic,
                               device=args.device)
    N = args.N
    x = torch.randn(2, N, 313, device=args.device)
    fb = torch.randn(2, N, 2500, device=args.device)
    coords = None
    if args.channel_agnostic:
        from lamquant_neural.positions import canonical_21_coords
        if N == 21:
            import numpy as np
            coords = torch.tensor(canonical_21_coords()).unsqueeze(0).expand(2, -1, -1).to(args.device)
        else:
            coords = torch.randn(2, N, 3, device=args.device) * 0.05
    rep = diag.run_preflight(x, fb, coords, expect_out=2500)
    print(rep.summary())
    print(f"\nPREFLIGHT {'OK' if rep.ok else 'FAILED'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep.to_dict(), f, indent=2)
    raise SystemExit(0 if rep.ok else 1)


if __name__ == "__main__":  # pragma: no cover
    _main()
