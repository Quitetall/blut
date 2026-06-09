#!/usr/bin/env python3
"""Full diagnostic suite for LamQuant student model checkpoint.

Metrics:
  1. Per-channel kurtosis histogram (32 latent channels)
  2. Per-channel Val R (21 EEG input channels)
  3. FSQ codebook utilization per latent channel
  4. Per-layer gradient norm
  5. Seizure vs background R
  6. End-to-end R at each quality mode (CLINICAL/MONITORING/ALERTING)
  7. Compression ratio at each quality mode
  8. Alpha distribution per layer
"""

import sys, os, json, glob
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "lamquant", "student"))
# MOVE-B: lamquant_codec is the pip-installed PUBLIC wheel; add common DTOs dir.
sys.path.insert(0, os.path.join(ROOT, "lamquant", "common"))

from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
from lamquant_neural.models.blocks import TernaryConv1d, TernaryConvTranspose1d, INT8Conv1d
from subband_preprocess import preprocess_subband_single, reconstruct_from_subband, lifting_3level_inverse, lpc_synthesize_channel


def load_model(ckpt_path, device, cdf_entries=32):
    model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32, cdf_entries=cdf_entries)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # Filter shape mismatches (e.g. cdf_breakpoints 32→64)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    return model


def recalibrate_cdf(model, val_files, device, max_files=16, max_windows=64):
    """Recompute CDF-LUT breakpoints from current encoder's latent distribution."""
    N_CDF = model.cdf_breakpoints.shape[1]
    # Set wide ramp so encode ≈ identity
    model.cdf_breakpoints.copy_(
        torch.linspace(-100, 100, N_CDF).unsqueeze(0).expand_as(model.cdf_breakpoints).to(device))
    latents = []
    for f in val_files[:max_files]:
        d = np.load(f)
        if 'l3' not in d.files:
            continue
        l3 = torch.from_numpy(d['l3'][:max_windows]).float().to(device)
        with torch.no_grad():
            lat = model.encode(l3, quantize=False)
        latents.append(lat.cpu())
    if not latents:
        print("  [!] No data for CDF recalibration")
        return
    all_lat = torch.cat(latents, dim=0)
    C = all_lat.shape[1]
    quantile_fracs = torch.linspace(0, 1, N_CDF)
    for c in range(C):
        ch_vals = all_lat[:, c, :].flatten().sort().values
        indices = (quantile_fracs * (len(ch_vals) - 1)).long()
        # ×100: encode() returns the POST-CDF latent; the wide linspace(-100,100)
        # ramp makes _cdf_forward linear (uniform = z/100), so the collected
        # quantiles are quantiles(z)/100 and must be scaled back by the ramp
        # half-width to land in raw-latent space. Omitting this leaves the
        # breakpoints 100× too tight and saturates the encoder.
        model.cdf_breakpoints.data[c] = (ch_vals[indices] * 100.0).to(device)
    print(f"  CDF recalibrated: {C} channels × {N_CDF} entries")
    bp = model.cdf_breakpoints
    print(f"  Breakpoint ranges: min=[{bp[:, 0].min():.3f}, {bp[:, 0].max():.3f}], "
          f"max=[{bp[:, -1].min():.3f}, {bp[:, -1].max():.3f}]")


def load_val_files(manifest_path, npz_dir):
    """Load val file list from manifest_v3.json (single source of truth).

    `manifest_path` and `npz_dir` are kept as positional args for API
    compatibility but ignored — the manifest is the canonical source.
    """
    sys.path.insert(0, os.path.join(ROOT, "lamquant"))
    from data_types import DatasetManifest, Split
    manifest = DatasetManifest.load(os.path.join(
        ROOT, "lamquant", "dataset", "manifest_v3.json"))
    return [str(p) for p in manifest.get_files(Split.VAL)]


def metric_1_kurtosis(model, val_files, device, max_files=16, max_windows_per_file=32):
    """Per-channel kurtosis of post-CDF latent distribution."""
    print("\n" + "="*70)
    print("1. PER-CHANNEL KURTOSIS (32 latent channels)")
    print("="*70)

    latents = []
    for f in val_files[:max_files]:
        d = np.load(f)
        if 'l3' not in d.files:
            continue
        l3 = torch.from_numpy(d['l3'][:max_windows_per_file]).float().to(device)
        with torch.no_grad():
            lat = model.encode(l3, quantize=True)
        latents.append(lat.cpu())
    if not latents:
        print("  No data!")
        return

    all_lat = torch.cat(latents, dim=0).numpy()  # [N, 32, 79]
    C = all_lat.shape[1]
    kurtosis = np.zeros(C)
    for c in range(C):
        ch = all_lat[:, c, :].flatten()
        mu, sigma = ch.mean(), ch.std()
        if sigma > 1e-8:
            kurtosis[c] = float(np.mean(((ch - mu) / sigma) ** 4)) - 3.0
        else:
            kurtosis[c] = 999.0

    print(f"  Aggregate kurtosis: {kurtosis.mean():.2f}")
    print(f"  Channels < 5:   {(kurtosis < 5).sum()}/32")
    print(f"  Channels 5-10:  {((kurtosis >= 5) & (kurtosis < 10)).sum()}/32")
    print(f"  Channels 10-50: {((kurtosis >= 10) & (kurtosis < 50)).sum()}/32")
    print(f"  Channels 50-100:{((kurtosis >= 50) & (kurtosis < 100)).sum()}/32")
    print(f"  Channels > 100: {(kurtosis >= 100).sum()}/32")
    print(f"  Target: 28+ channels below 5")
    print()
    for c in range(C):
        bar = "#" * max(1, min(40, int(abs(kurtosis[c]) / 5)))
        flag = " <<<" if kurtosis[c] > 10 else ""
        print(f"    ch{c:>2}: {kurtosis[c]:>8.2f} {bar}{flag}")
    return kurtosis


def metric_2_per_channel_val_r(model, val_files, device, max_files=16, max_windows_per_file=32):
    """Per-channel Pearson R on L3 approximation (21 EEG channels)."""
    print("\n" + "="*70)
    print("2. PER-CHANNEL VAL R (21 EEG input channels)")
    print("="*70)

    per_ch_r = [[] for _ in range(21)]
    for f in val_files[:max_files]:
        d = np.load(f)
        if 'l3' not in d.files:
            continue
        l3 = torch.from_numpy(d['l3'][:max_windows_per_file]).float().to(device)
        with torch.no_grad():
            recon = model(l3, quantize=True)
        l3_np = l3.cpu().numpy()
        recon_np = recon.cpu().numpy()
        for c in range(min(21, l3_np.shape[1])):
            for w in range(l3_np.shape[0]):
                x = l3_np[w, c, :]
                y = recon_np[w, c, :]
                if x.std() > 1e-8 and y.std() > 1e-8:
                    r = np.corrcoef(x, y)[0, 1]
                    if not np.isnan(r):
                        per_ch_r[c].append(r)

    print(f"  {'Ch':>4} {'Mean R':>8} {'Std':>7} {'N':>6}")
    print(f"  {'----':>4} {'------':>8} {'---':>7} {'---':>6}")
    for c in range(21):
        if per_ch_r[c]:
            mean_r = np.mean(per_ch_r[c])
            std_r = np.std(per_ch_r[c])
            bar = "#" * max(1, int(mean_r * 20))
            flag = " <<<" if mean_r < 0.75 else ""
            print(f"    {c:>2}   {mean_r:>8.4f} {std_r:>7.4f} {len(per_ch_r[c]):>6} {bar}{flag}")
    agg = np.mean([np.mean(v) for v in per_ch_r if v])
    print(f"\n  Aggregate: {agg:.4f}")
    return per_ch_r


def metric_3_fsq_utilization(model, val_files, device, max_files=16, max_windows_per_file=32):
    """FSQ codebook utilization per latent channel at L=32 (CLINICAL)."""
    print("\n" + "="*70)
    print("3. FSQ CODEBOOK UTILIZATION (L=32, CLINICAL)")
    print("="*70)

    L = 32  # Clinical mode
    latents = []
    for f in val_files[:max_files]:
        d = np.load(f)
        if 'l3' not in d.files:
            continue
        l3 = torch.from_numpy(d['l3'][:max_windows_per_file]).float().to(device)
        with torch.no_grad():
            lat = model.encode(l3, quantize=True)
        latents.append(lat.cpu())
    if not latents:
        print("  No data!")
        return

    all_lat = torch.cat(latents, dim=0).numpy()  # [N, 32, 79]
    C = all_lat.shape[1]

    total_dead = 0
    for c in range(C):
        ch_vals = all_lat[:, c, :].flatten()
        vmin, vmax = ch_vals.min(), ch_vals.max()
        span = vmax - vmin + 1e-8
        symbols = np.clip(((ch_vals - vmin) / span * L).astype(int), 0, L - 1)
        used = len(np.unique(symbols))
        dead = L - used
        total_dead += dead
        bar = "#" * used + "." * dead
        flag = " <<<" if dead > 4 else ""
        print(f"    ch{c:>2}: {used:>2}/{L} active  {bar}{flag}")

    print(f"\n  Total dead bins: {total_dead}/{C * L} ({total_dead / (C * L) * 100:.1f}%)")
    print(f"  Target: 0 dead bins")


def metric_4_gradient_norm(model, val_files, device, max_windows=64):
    """Per-layer gradient norm from a single validation batch."""
    print("\n" + "="*70)
    print("4. PER-LAYER GRADIENT NORM")
    print("="*70)

    # Load a batch
    l3_list = []
    for f in val_files[:4]:
        d = np.load(f)
        if 'l3' in d.files:
            l3_list.append(d['l3'][:16])
    if not l3_list:
        print("  No data!")
        return
    l3 = torch.from_numpy(np.concatenate(l3_list, axis=0)[:max_windows]).float().to(device)

    model.train()
    model.zero_grad()
    recon = model(l3, quantize=True)
    loss = torch.nn.functional.mse_loss(recon, l3)
    loss.backward()

    print(f"  {'Layer':<35} {'Grad Norm':>10} {'#Params':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*8}")
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.data.norm(2).item()
            norms[name] = norm
            n_params = p.numel()
            bar = "#" * max(1, min(30, int(norm * 10)))
            print(f"    {name:<33} {norm:>10.4f} {n_params:>8,} {bar}")

    model.eval()
    model.zero_grad()
    return norms


def metric_5_seizure_vs_background(model, val_files, device, max_files=None):
    """Seizure-segment R vs background R."""
    print("\n" + "="*70)
    print("5. SEIZURE vs BACKGROUND VAL R")
    print("="*70)

    seizure_r, background_r = [], []
    files = val_files if max_files is None else val_files[:max_files]

    for f in files:
        d = np.load(f)
        if 'l3' not in d.files or 'seizure_mask' not in d.files:
            continue
        mask = d['seizure_mask']  # [total_samples]
        l3 = d['l3']  # [N, 21, 313]
        n_windows = l3.shape[0]
        total_samples = len(mask)
        samples_per_window = total_samples // max(n_windows, 1)

        l3_t = torch.from_numpy(l3).float().to(device)
        with torch.no_grad():
            recon = model(l3_t, quantize=True).cpu().numpy()

        for w in range(n_windows):
            start = w * samples_per_window
            end = min(start + samples_per_window, total_samples)
            window_mask = mask[start:end]
            is_seizure = window_mask.sum() > (end - start) * 0.5

            x = l3[w].flatten()
            y = recon[w].flatten()
            if x.std() > 1e-8 and y.std() > 1e-8:
                r = np.corrcoef(x, y)[0, 1]
                if not np.isnan(r):
                    if is_seizure:
                        seizure_r.append(r)
                    else:
                        background_r.append(r)

    bg_mean = np.mean(background_r) if background_r else 0
    sz_mean = np.mean(seizure_r) if seizure_r else 0
    print(f"  Background R: {bg_mean:.4f} (n={len(background_r)})")
    print(f"  Seizure R:    {sz_mean:.4f} (n={len(seizure_r)})")
    if seizure_r:
        gap = bg_mean - sz_mean
        print(f"  Gap:          {gap:.4f} {'<<<' if gap > 0.05 else '(OK)'}")
    else:
        print(f"  No seizure windows in validation set")


def metric_6_end_to_end_r(val_files, device, model, max_files=8):
    """End-to-end R at each quality mode: full codec pipeline."""
    print("\n" + "="*70)
    print("6. END-TO-END R (full codec: TNN + inverse lifting + LPC)")
    print("="*70)

    from codec import SubbandCodec as LamQuantCodec

    codec = LamQuantCodec(model)

    modes = {
        'ALERTING': LamQuantCodec.QUALITY_ALERTING,
        'MONITORING': LamQuantCodec.QUALITY_MONITORING,
        'CLINICAL': LamQuantCodec.QUALITY_CLINICAL,
    }

    for mode_name, mode_val in modes.items():
        r_scores = []
        for f in val_files[:max_files]:
            d = np.load(f)
            if 'data' not in d.files or 'l3' not in d.files:
                continue
            raw_data = d['data']  # [21, total_samples] int32
            gain = float(d['gain'])
            sr = float(d['sample_rate'])

            # Process window by window
            n_windows = d['l3'].shape[0]
            samples_per_window = raw_data.shape[1] // max(n_windows, 1)

            for w in range(min(n_windows, 8)):
                start = w * samples_per_window
                end = start + samples_per_window
                if end > raw_data.shape[1]:
                    break
                segment = (raw_data[:, start:end].astype(np.float32) / 2147483647.0) * 1000.0

                try:
                    l3, coeffs, subs_per_ch = preprocess_subband_single(segment)
                    l3_t = torch.from_numpy(l3).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        lat = codec.model.encode(l3_t, quantize=True)

                    # FSQ quantize + dequantize at this mode's level
                    L = codec.FSQ_LEVELS_BY_MODE[mode_val]
                    lat_np = lat.cpu().numpy().flatten()
                    vmin, vmax = lat_np.min(), lat_np.max()
                    span = vmax - vmin + 1e-8
                    symbols = np.clip(((lat_np - vmin) / span * L).astype(int), 0, L - 1)
                    # Dequantize: center of each bin
                    lat_deq = (symbols.astype(np.float32) + 0.5) / L * span + vmin
                    lat_deq_t = torch.from_numpy(
                        lat_deq.reshape(lat.shape)).float().to(device)

                    with torch.no_grad():
                        l3_recon = codec.model.decode(lat_deq_t, target_len=313, quantize=True)

                    l3_recon_np = l3_recon[0].cpu().numpy().astype(np.float64)

                    # Inverse lifting with actual detail subbands
                    recon_signal = reconstruct_from_subband(l3_recon_np, coeffs, subs_per_ch)

                    # Pearson R on the full [21, T] reconstruction
                    orig_flat = segment[:, :recon_signal.shape[1]].flatten()
                    recon_flat = recon_signal.flatten()
                    if orig_flat.std() > 1e-8 and recon_flat.std() > 1e-8:
                        r = np.corrcoef(orig_flat, recon_flat)[0, 1]
                        if not np.isnan(r):
                            r_scores.append(r)
                except Exception as e:
                    print(f"    [!] Window {w} error: {e}")
                    continue

        mean_r = np.mean(r_scores) if r_scores else 0
        print(f"  {mode_name:<12} L={codec.FSQ_LEVELS_BY_MODE[mode_val]:>2}  R={mean_r:.4f}  (n={len(r_scores)})")


def metric_7_compression_ratio(val_files, device, model, max_files=8):
    """Compression ratio at each quality mode."""
    print("\n" + "="*70)
    print("7. COMPRESSION RATIO (bytes per 10-second window)")
    print("="*70)

    from codec import SubbandCodec as LamQuantCodec

    codec = LamQuantCodec(model)
    codec.model = codec.model.to(device)

    modes = {
        'ALERTING': LamQuantCodec.QUALITY_ALERTING,
        'MONITORING': LamQuantCodec.QUALITY_MONITORING,
        'CLINICAL': LamQuantCodec.QUALITY_CLINICAL,
    }

    for mode_name, mode_val in modes.items():
        sizes = []
        for f in val_files[:max_files]:
            d = np.load(f)
            if 'data' not in d.files or 'l3' not in d.files:
                continue
            raw_data = d['data']
            gain = float(d['gain'])
            n_windows = d['l3'].shape[0]
            samples_per_window = raw_data.shape[1] // max(n_windows, 1)

            for w in range(min(n_windows, 4)):
                start = w * samples_per_window
                end = start + samples_per_window
                if end > raw_data.shape[1]:
                    break
                segment = (raw_data[:, start:end].astype(np.float32) / 2147483647.0) * 1000.0

                try:
                    l3, coeffs, subs_per_ch = preprocess_subband_single(segment)
                    l3_t = torch.from_numpy(l3).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        lat = codec.model.encode(l3_t, quantize=True)

                    packet = codec.compress(
                        lat[0].cpu(), lpc_coeffs=coeffs,
                        subbands_per_ch=subs_per_ch, quality_mode=mode_val)
                    sizes.append(len(packet))
                except Exception:
                    continue

        if sizes:
            mean_bytes = np.mean(sizes)
            raw_bytes = 21 * samples_per_window * 2  # 21 ch × T × int16
            cr = raw_bytes / mean_bytes if mean_bytes > 0 else 0
            print(f"  {mode_name:<12} {mean_bytes:>7.0f} bytes/window  CR={cr:>6.1f}x  "
                  f"raw={raw_bytes} bytes")
        else:
            print(f"  {mode_name:<12} No data")


def metric_8_alpha_distribution(model):
    """Alpha distribution per ternary layer."""
    print("\n" + "="*70)
    print("8. ALPHA DISTRIBUTION PER LAYER")
    print("="*70)

    print(f"  {'Layer':<35} {'Alpha':>7} {'σ_W':>7} {'Sparsity':>9} {'Ratio':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")
    for name, m in model.named_modules():
        if hasattr(m, 'lsq_alpha') and hasattr(m, 'weight'):
            alpha = m.lsq_alpha.data.abs().mean().item()
            sigma_w = m.weight.data.std().item()
            w = m.weight.data
            a = m.lsq_alpha.data.abs()
            w_scaled = w / a
            w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)
            sparsity = (w_ternary == 0).float().mean().item() * 100
            ratio = alpha / max(sigma_w, 1e-8)
            bar = "#" * max(1, min(20, int(alpha * 10)))
            print(f"    {name:<33} {alpha:>7.4f} {sigma_w:>7.4f} {sparsity:>8.1f}% {ratio:>7.2f} {bar}")


def metric_9_detail_energy(val_files, device, ckpt_path, max_files=8):
    """Detail subband energy contribution vs L3 at each quality mode."""
    print("\n" + "="*70)
    print("9. DETAIL SUBBAND ENERGY CONTRIBUTION")
    print("="*70)

    from codec import SubbandCodec as LamQuantCodec
    from subband_preprocess import lifting_3level_inverse_int

    model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32)
    model.load_state_dict(
        {k: v for k, v in torch.load(ckpt_path, map_location='cpu', weights_only=True).items()
         if k in model.state_dict() and v.shape == model.state_dict()[k].shape},
        strict=False)
    codec = LamQuantCodec(model)
    codec.model = codec.model.to(device)

    modes = {
        'ALERTING': LamQuantCodec.QUALITY_ALERTING,
        'MONITORING': LamQuantCodec.QUALITY_MONITORING,
        'CLINICAL': LamQuantCodec.QUALITY_CLINICAL,
    }

    for mode_name, mode_val in modes.items():
        l3_energies, detail_energies = [], []
        for f in val_files[:max_files]:
            d = np.load(f)
            if 'data' not in d.files or 'l3' not in d.files:
                continue
            raw_data = d['data']
            gain = float(d['gain'])
            n_windows = d['l3'].shape[0]
            spw = raw_data.shape[1] // max(n_windows, 1)

            for w in range(min(n_windows, 4)):
                start = w * spw
                end = start + spw
                if end > raw_data.shape[1]:
                    break
                segment = (raw_data[:, start:end].astype(np.float32) / 2147483647.0) * 1000.0
                try:
                    l3, coeffs, subs_per_ch = preprocess_subband_single(segment)
                    l3_t = torch.from_numpy(l3).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        lat = codec.model.encode(l3_t, quantize=True)
                    L = codec.FSQ_LEVELS_BY_MODE[mode_val]
                    lat_np = lat.cpu().numpy().flatten()
                    vmin, vmax = lat_np.min(), lat_np.max()
                    span = vmax - vmin + 1e-8
                    symbols = np.clip(((lat_np - vmin) / span * L).astype(int), 0, L - 1)
                    lat_deq = (symbols.astype(np.float32) + 0.5) / L * span + vmin
                    lat_deq_t = torch.from_numpy(lat_deq.reshape(lat.shape)).float().to(device)
                    with torch.no_grad():
                        l3_recon = codec.model.decode(lat_deq_t, target_len=313, quantize=True)
                    l3_recon_np = l3_recon[0].cpu().numpy().astype(np.float64)

                    # Compute energy from L3 only (zero details)
                    for c in range(l3_recon_np.shape[0]):
                        subs_zero = {k: np.zeros_like(v) for k, v in subs_per_ch[c].items()}
                        subs_zero['l3_approx'] = np.round(l3_recon_np[c]).astype(np.int64)
                        l3_only = lifting_3level_inverse_int(subs_zero).astype(np.float64)
                        l3_energies.append(np.sum(l3_only ** 2))

                        subs_full = {k: np.round(v).astype(np.int64) for k, v in subs_per_ch[c].items()}
                        subs_full['l3_approx'] = np.round(l3_recon_np[c]).astype(np.int64)
                        full = lifting_3level_inverse_int(subs_full).astype(np.float64)
                        detail_only_energy = np.sum(full ** 2) - np.sum(l3_only ** 2)
                        detail_energies.append(max(0, detail_only_energy))
                except Exception:
                    continue

        if l3_energies:
            total = np.sum(l3_energies) + np.sum(detail_energies)
            l3_pct = np.sum(l3_energies) / max(total, 1e-12) * 100
            det_pct = np.sum(detail_energies) / max(total, 1e-12) * 100
            print(f"  {mode_name:<12} L3={l3_pct:.1f}%  Detail={det_pct:.1f}%")
        else:
            print(f"  {mode_name:<12} No data")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LamQuant checkpoint diagnostics")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(ROOT, "ai_models/student/student_subband_gold.ckpt"))
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--cdf-entries", type=int, default=32)
    parser.add_argument("--recalibrate-cdf", action="store_true",
                        help="Recompute CDF-LUT from checkpoint's latent distribution")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CDF entries: {args.cdf_entries}{' (recalibrating)' if args.recalibrate_cdf else ''}")

    npz_dir = os.path.join(ROOT, "ai_models/dataset_sim/q31_events")
    manifest_path = os.path.join(ROOT, "ai_models/dataset_sim/validation_manifest/validation_manifest.json")

    model = load_model(args.checkpoint, device, cdf_entries=args.cdf_entries)
    val_files = load_val_files(manifest_path, npz_dir)
    print(f"Validation files: {len(val_files)}")

    if args.recalibrate_cdf:
        print("\n[*] Recalibrating CDF-LUT...")
        recalibrate_cdf(model, val_files, device)

    kurtosis = metric_1_kurtosis(model, val_files, device, max_files=args.max_files)
    metric_2_per_channel_val_r(model, val_files, device, max_files=args.max_files)
    metric_3_fsq_utilization(model, val_files, device, max_files=args.max_files)
    metric_4_gradient_norm(model, val_files, device)
    metric_5_seizure_vs_background(model, val_files, device, max_files=None)
    metric_6_end_to_end_r(val_files, device, model, max_files=min(8, args.max_files))
    metric_7_compression_ratio(val_files, device, model, max_files=min(8, args.max_files))
    metric_8_alpha_distribution(model)
    metric_9_detail_energy(val_files, device, args.checkpoint, max_files=min(8, args.max_files))

    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
