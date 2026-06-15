#!/usr/bin/env python3
"""Residual diagnostics — is the reconstruction error structure or noise floor?

Loads a trained neural-codec checkpoint, runs it over a validation split, forms
the fullband reconstruction residual ``r = recon - target`` ([21, 2500] @ 250 Hz),
and runs a battery that separates *unmodeled structure* (still optimizable) from
an *irreducible noise floor* (stop optimizing, declare the LQS tier):

  1. per-band PRD/R         — where the error lives (δ/θ carried by L3 vs γ/HFO dropped)
  2. residual PSD           — flat (white) vs band-concentrated (unmodeled signal)
  3. autocorrelation        — white = δ at lag 0; non-zero ACF = temporally predictable
  4. cross-channel corr     — residuals correlated across the 21 ch ⇒ spatial structure unused
  5. predictability probe   — can a tiny linear model predict r from its own AR + neighbours?
                              R² > 0 ⇒ structure remains; R² ≈ 0 ⇒ at the floor (the clean test)
  6. distribution           — kurtosis/skew (additive Gaussian noise is light-tailed, ~symmetric)
  7. heteroscedasticity     — does |r| scale with |target| (error concentrates on spikes)?
  8. ADC-noise-floor ratio  — residual RMS vs a high-band noise-floor proxy; ≈1 ⇒ irreducible

Pairs with the full-residual experiment (ADR 0049, task #238): run BEFORE (expect
structure in γ/HFO + cross-channel corr ⇒ proves the lever) and AFTER (if still
structured → keep pushing; if it collapses toward the floor → real wall).

Reuses the canonical live eval path (eval_codec_pccp.py / validate_joint), NOT the
deprecated eval_fullband.py. Run via the `lamquant_residual_diag` BLUT recipe.

  PYTHONPATH=blut/python python -m lamquant.student.residual_diagnostics \
      --joint-ckpt <ckpt> --windows 256 --out-dir outputs/residual_diag/<run>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────── checkpoint + model ────────────────────────────
_IN_CH_TO_BANDS = {
    21: ("none", "fold"),
    42: ("l3_detail", "fold"),
    84: ("all", "interp"),
    168: ("all", "fold"),
}


def _strip_compile(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _load_ckpt(path: str) -> dict:
    import torch

    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return _strip_compile(sd)


def _detect_in_channels(enc_sd: dict) -> int:
    """First-conv input width = encoder in_channels (21/42/84/168)."""
    # premix / stem / first conv: a weight [out, in, k] whose `in` is a known stack size.
    for key in sorted(enc_sd):
        w = enc_sd[key]
        if hasattr(w, "ndim") and w.ndim == 3 and int(w.shape[1]) in _IN_CH_TO_BANDS:
            if any(t in key.lower() for t in ("premix", "stem", "0.0", "first")):
                return int(w.shape[1])
    # fallback: smallest in-dim among 3-D weights that is a known stack size
    cands = [int(w.shape[1]) for w in enc_sd.values()
             if hasattr(w, "ndim") and w.ndim == 3 and int(w.shape[1]) in _IN_CH_TO_BANDS]
    return min(cands) if cands else 21


def _build_codec(args, dev):
    """Build JointCodec + load weights. Returns (codec, in_channels)."""
    import torch  # noqa: F401
    from lamquant.student.joint_codec import build_default_joint

    enc_sd = dec_sd = whole_sd = None
    if args.joint_ckpt:
        whole_sd = _load_ckpt(args.joint_ckpt)
        # split prefixes if the joint ckpt stores encoder./decoder.
        enc_sd = {k[len("encoder."):]: v for k, v in whole_sd.items() if k.startswith("encoder.")}
        dec_sd = {k[len("decoder."):]: v for k, v in whole_sd.items() if k.startswith("decoder.")}
        if not enc_sd:  # flat codec state — keep whole for codec.load_state_dict
            enc_sd = None
    if args.encoder_ckpt:
        enc_sd = _load_ckpt(args.encoder_ckpt)
    if args.decoder_ckpt:
        dec_sd = _load_ckpt(args.decoder_ckpt)

    in_ch = args.in_channels or (_detect_in_channels(enc_sd) if enc_sd else 21)
    codec = build_default_joint(
        latent_dim=args.latent_dim,
        vocos_tier=args.vocos_tier,
        in_channels=in_ch,
        decoder_channels=21,
    ).to(dev).eval()

    if whole_sd is not None and enc_sd is None:
        miss, unexp = codec.load_state_dict(whole_sd, strict=False)
        print(f"[load] joint flat: missing={len(miss)} unexpected={len(unexp)}", file=sys.stderr)
    else:
        if enc_sd:
            m, u = codec.encoder.load_state_dict(enc_sd, strict=False)
            print(f"[load] encoder: missing={len(m)} unexpected={len(u)} in_ch={in_ch}", file=sys.stderr)
        if dec_sd:
            m, u = codec.decoder.load_state_dict(dec_sd, strict=False)
            print(f"[load] decoder: missing={len(m)} unexpected={len(u)}", file=sys.stderr)
    return codec, in_ch


def _collect_residuals(codec, args, in_ch, dev):
    """Run val windows → stacked residual + target arrays [N,21,2500] (numpy)."""
    import torch
    from lamquant.student.lma_typed_adapter import LmaTypedL3Dataset

    bands, mode = _IN_CH_TO_BANDS.get(in_ch, ("none", "fold"))
    if bands != "none":
        # the dataset stacks detail channels from these env knobs (must precede construction)
        os.environ["SNN_DETAIL_BANDS"] = bands
        os.environ["SNN_DETAIL_STACK_MODE"] = mode

    ds = LmaTypedL3Dataset(
        lma_root=args.lma_root,
        split="val",
        split_manifest_path=args.split_manifest,
        windows_per_epoch=args.windows,
        return_fullband=True,
        seed=args.seed,
    )
    ds.calibrate_shard_budget(dev)

    res, tgt = [], []
    seen = 0
    with torch.no_grad():
        for batch in ds.prefetch_typed_batches(args.batch_size, dev):
            x_l3 = batch.l3_approx
            fb = batch.fullband_target
            if fb is None:
                continue
            recon = codec(x_l3, quantize=args.quantize)
            T = min(recon.shape[-1], fb.shape[-1])
            r = (recon[..., :T].float() - fb[..., :T].float()).cpu().numpy()
            res.append(r)
            tgt.append(fb[..., :T].float().cpu().numpy())
            seen += r.shape[0]
            if seen >= args.windows:
                break
    if not res:
        raise RuntimeError("no fullband windows collected — check --return_fullband path / manifest")
    return np.concatenate(res, 0), np.concatenate(tgt, 0)


# ─────────────────────────────── analyses ──────────────────────────────────
def _psd_band_fractions(x: np.ndarray, fs: float) -> dict:
    """Welch PSD energy fraction per EEG band + spectral flatness (1=white)."""
    from scipy.signal import welch
    from lamquant.common.metrics import EEG_BANDS

    # x: [N,21,T] → average PSD across windows+channels
    f, pxx = welch(x, fs=fs, nperseg=min(256, x.shape[-1]), axis=-1)
    p = pxx.mean(axis=(0, 1))  # [F]
    total = float(p.sum()) + 1e-20
    out = {}
    for name, (lo, hi) in EEG_BANDS.items():
        m = (f >= lo) & (f < hi)
        out[f"frac_{name}"] = float(p[m].sum() / total)
    # spectral flatness: geomean/mean of PSD (1.0 = perfectly white)
    pl = np.clip(p, 1e-20, None)
    out["spectral_flatness"] = float(np.exp(np.mean(np.log(pl))) / (pl.mean() + 1e-20))
    return out


def _autocorr(x: np.ndarray, max_lag: int = 20) -> dict:
    """Mean residual ACF over channels+windows. White ⇒ |ACF(k>0)| ≈ 0."""
    xc = x - x.mean(axis=-1, keepdims=True)
    var = (xc * xc).mean(axis=-1, keepdims=True) + 1e-20
    acf = []
    for k in range(1, max_lag + 1):
        c = (xc[..., : -k] * xc[..., k:]).mean(axis=-1, keepdims=True) / var
        acf.append(float(c.mean()))
    return {
        "acf_lag1": acf[0],
        "acf_lag2": acf[1] if len(acf) > 1 else 0.0,
        "acf_abs_mean_1to20": float(np.mean(np.abs(acf))),
        "acf_curve": [round(a, 4) for a in acf],
    }


def _cross_channel(x: np.ndarray) -> dict:
    """Mean |off-diagonal| residual cross-channel correlation. White ⇒ ≈0."""
    # x: [N,21,T] → [21, N*T]
    N, C, T = x.shape
    flat = x.transpose(1, 0, 2).reshape(C, N * T)
    corr = np.corrcoef(flat)
    off = corr[~np.eye(C, dtype=bool)]
    return {
        "xchan_mean_abs_corr": float(np.nanmean(np.abs(off))),
        "xchan_max_abs_corr": float(np.nanmax(np.abs(off))),
    }


def _predictability(x: np.ndarray, p_ar: int = 4, n_samples: int = 200_000) -> dict:
    """Ridge-predict r[c,t] from its AR-p history + same-t spatial mean of others.

    R² of explained variance: > ~0.02 ⇒ structure the model could still exploit;
    ≈ 0 ⇒ residual is (locally) unpredictable = noise-floor-like. This is the
    crispest structure-vs-floor test.
    """
    N, C, T = x.shape
    rng = np.random.default_rng(0)
    feats, targs = [], []
    per = max(1, n_samples // (N * C))
    spatial = x.mean(axis=1, keepdims=True)  # [N,1,T] spatial mean
    for _ in range(per):
        ts = rng.integers(p_ar, T, size=1)[0]
        f = [x[:, :, ts - k] for k in range(1, p_ar + 1)]          # AR history [N,C] each
        f.append(np.broadcast_to(spatial[:, :, ts], (N, C)))       # spatial context
        X = np.stack(f, axis=-1).reshape(N * C, -1)
        y = x[:, :, ts].reshape(N * C)
        feats.append(X)
        targs.append(y)
    X = np.concatenate(feats, 0)
    y = np.concatenate(targs, 0)
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # bias
    lam = 1e-3 * X.shape[0]
    A = X.T @ X + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ y)
    pred = X @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-20
    return {"predictability_r2": round(1.0 - ss_res / ss_tot, 5), "ar_order": p_ar}


def _distribution(x: np.ndarray) -> dict:
    from scipy.stats import kurtosis, skew

    flat = x.reshape(-1)
    return {
        "kurtosis_excess": float(kurtosis(flat, fisher=True)),
        "skew": float(skew(flat)),
        "residual_rms": float(np.sqrt((flat * flat).mean())),
    }


def _heteroscedasticity(r: np.ndarray, t: np.ndarray) -> dict:
    """Corr(|residual|, |target|): does error concentrate on high-amplitude events?"""
    ra = np.abs(r).reshape(-1)
    ta = np.abs(t).reshape(-1)
    if ra.size > 500_000:  # subsample for speed
        idx = np.random.default_rng(0).integers(0, ra.size, 500_000)
        ra, ta = ra[idx], ta[idx]
    c = np.corrcoef(ra, ta)[0, 1]
    return {"abs_resid_vs_abs_target_corr": float(c)}


def _noise_floor_ratio(r: np.ndarray, t: np.ndarray, fs: float) -> dict:
    """Residual RMS vs a high-band noise-floor proxy.

    No ADC-noise estimator exists in the codebase; proxy the physical floor with
    the target's RMS in a near-Nyquist band (45–<Nyq Hz) — EEG carries little
    physiological signal there, so it is dominated by amplifier/ADC noise. Ratio
    ≈ 1 ⇒ residual is at the floor; ≫ 1 ⇒ recoverable signal remains.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = fs / 2.0
    lo = min(45.0, 0.8 * nyq)
    sos = butter(4, [lo / nyq, 0.99], btype="band", output="sos")
    hf = sosfiltfilt(sos, t, axis=-1)
    floor = float(np.sqrt((hf * hf).mean()))
    rms = float(np.sqrt((r * r).mean()))
    return {
        "noise_floor_proxy_rms": floor,
        "residual_rms": rms,
        "residual_to_floor_ratio": round(rms / (floor + 1e-20), 4),
        "noise_floor_band_hz": [round(lo, 1), round(0.99 * nyq, 1)],
    }


def _verdict(report: dict) -> dict:
    """Structure-vs-floor call from the battery."""
    pr2 = report["predictability"]["predictability_r2"]
    xch = report["cross_channel"]["xchan_mean_abs_corr"]
    acf = report["autocorr"]["acf_abs_mean_1to20"]
    flat = report["psd"]["spectral_flatness"]
    ratio = report["noise_floor"]["residual_to_floor_ratio"]
    structured = (pr2 > 0.02) or (xch > 0.1) or (acf > 0.05)
    near_floor = (pr2 <= 0.02) and (xch <= 0.1) and (acf <= 0.05) and (flat > 0.5) and (ratio < 1.5)
    # which bands hold the recoverable energy
    frac = {k.replace("frac_", ""): v for k, v in report["psd"].items() if k.startswith("frac_")}
    hot = sorted(frac, key=frac.get, reverse=True)[:2]
    return {
        "structured": bool(structured),
        "near_noise_floor": bool(near_floor),
        "residual_energy_concentrated_in": hot,
        "call": (
            "NOISE FLOOR — residual is unpredictable + near the high-band noise proxy; "
            "further optimization unlikely to help; declare the LQS tier."
            if near_floor else
            "STRUCTURED — recoverable signal remains (predictable / cross-channel / "
            f"band-concentrated in {hot}); keep optimizing (full-residual / entropy model / spatial)."
        ),
    }


# ─────────────────────────────────── main ──────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Residual structure-vs-noise-floor diagnostics")
    ap.add_argument("--joint-ckpt", default=None, help="single ckpt with encoder.+decoder. (or flat codec) state")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--decoder-ckpt", default=None)
    ap.add_argument("--lma-root", default="/mnt/4tb/data/Archive/lma")
    ap.add_argument("--split-manifest", default="/mnt/4tb/data/Training/manifests/split_manifest_codec_v1.json")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--vocos-tier", type=int, default=3)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--in-channels", type=int, default=0, help="0 = autodetect from encoder ckpt")
    ap.add_argument("--quantize", action="store_true", help="run the ternary-quantized encoder path")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--out-dir", default="outputs/residual_diag")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not (args.joint_ckpt or args.encoder_ckpt):
        ap.error("need --joint-ckpt or --encoder-ckpt (+ --decoder-ckpt)")

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    codec, in_ch = _build_codec(args, dev)
    r, t = _collect_residuals(codec, args, in_ch, dev)
    print(f"[diag] collected residual {r.shape} (windows×ch×T) @ {args.fs} Hz, in_ch={in_ch}", file=sys.stderr)

    from lamquant.common.metrics import per_band_prd, per_band_r, prd_numpy, pearson_r_numpy

    report = {
        "config": {"in_channels": in_ch, "vocos_tier": args.vocos_tier, "quantize": args.quantize,
                   "windows": int(r.shape[0]), "fs": args.fs,
                   "ckpt": args.joint_ckpt or f"{args.encoder_ckpt}+{args.decoder_ckpt}"},
        "global": {"prd_pct": round(float(prd_numpy(t, t + r)), 4),
                   "pearson_r": round(float(pearson_r_numpy(t, t + r)), 5)},
        "per_band_prd": {k: round(float(v), 3) for k, v in per_band_prd(t, t + r, fs=args.fs).items()},
        "per_band_r": {k: round(float(v), 4) for k, v in per_band_r(t, t + r, fs=args.fs).items()},
        "psd": _psd_band_fractions(r, args.fs),
        "autocorr": _autocorr(r),
        "cross_channel": _cross_channel(r),
        "predictability": _predictability(r),
        "distribution": _distribution(r),
        "heteroscedasticity": _heteroscedasticity(r, t),
        "noise_floor": _noise_floor_ratio(r, t, args.fs),
    }
    report["verdict"] = _verdict(report)

    (out / "residual_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps(report["per_band_prd"]))
    print(f"[diag] wrote {out/'residual_report.json'}", file=sys.stderr)

    if not args.no_plots:
        _plots(r, t, args.fs, out)
    return 0


def _plots(r, t, fs, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.signal import welch
    except Exception as e:  # plotting is best-effort
        print(f"[diag] plots skipped ({e})", file=sys.stderr)
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    f, pr = welch(r, fs=fs, nperseg=min(256, r.shape[-1]), axis=-1)
    _, pt = welch(t, fs=fs, nperseg=min(256, t.shape[-1]), axis=-1)
    ax[0, 0].semilogy(f, pr.mean((0, 1)), label="residual")
    ax[0, 0].semilogy(f, pt.mean((0, 1)), label="target", alpha=0.6)
    ax[0, 0].set(title="PSD (Welch)", xlabel="Hz"); ax[0, 0].legend()
    xc = r - r.mean(-1, keepdims=True)
    var = (xc * xc).mean(-1, keepdims=True) + 1e-20
    acf = [float(((xc[..., :-k] * xc[..., k:]).mean(-1, keepdims=True) / var).mean()) for k in range(1, 31)]
    ax[0, 1].stem(range(1, 31), acf); ax[0, 1].set(title="residual ACF", xlabel="lag")
    C = r.shape[1]
    corr = np.corrcoef(r.transpose(1, 0, 2).reshape(C, -1))
    im = ax[1, 0].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm"); ax[1, 0].set(title="cross-channel corr")
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 1].hist(r.reshape(-1), bins=120, density=True); ax[1, 1].set(title="residual histogram", yscale="log")
    fig.tight_layout(); fig.savefig(out / "residual_diag.png", dpi=110)
    print(f"[diag] wrote {out/'residual_diag.png'}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
