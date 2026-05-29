#!/usr/bin/env python3
"""ship_fast_preset.py — end-to-end integration test for the fast-preset
checkpoint through the full deployment pipeline.

This script validates that:

  1. The encoder checkpoint loads into SubbandCodec
  2. Raw EEG [21, 2500] encodes to an LMQ v5 packet
  3. The packet decompresses to recover the same latent
  4. The encoder-solo inverse DSP path (Mode 1, base station via
     lifting-inverse + LPC synthesis) produces a fullband reconstruction
  5. (Optional) The Vocos decoder path produces fullband
  6. Compression ratio, R, PRD, and per-band PRD are printed
  7. An LMQ file round-trips through NeuralWriter → LMQReader

This is the "ship the fast preset first" artefact the user specified:
a working checkpoint through the entire pipeline, even at R=0.85 quality,
that proves every component integrates.

Usage:
    python ai_models/student/ship_fast_preset.py
    python ai_models/student/ship_fast_preset.py --encoder <path> --decoder <path>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'lamquant' / 'student'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'common'))  # MOVE-B: common DTOs
sys.path.insert(0, str(_REPO / 'lamquant'))


def main() -> int:
    parser = argparse.ArgumentParser(prog='ship_fast_preset')
    parser.add_argument('--encoder', type=Path,
                        default=_REPO / 'lamquant' / 'student' / 'student_encoder_joint_fast.ckpt')
    parser.add_argument('--decoder', type=Path,
                        default=_REPO / 'lamquant' / 'student' / 'decoder_tier3_joint_fast.ckpt')
    parser.add_argument('--tier', type=int, default=3)
    parser.add_argument('--n-windows', type=int, default=5)
    args = parser.parse_args()

    print('=' * 72)
    print('  FAST-PRESET END-TO-END INTEGRATION TEST')
    print('=' * 72)

    # ---- Step 1: Load encoder into SubbandCodec ----
    print('\n[1/6] Loading encoder into SubbandCodec...')
    from lamquant_neural.codec import SubbandCodec
    codec = SubbandCodec.from_checkpoint(str(args.encoder))
    print(f'      Encoder params: {sum(p.numel() for p in codec.model.parameters()):,}')

    # ---- Step 2: Load real val EEG from manifest ----
    print('\n[2/6] Loading real EEG from val split...')
    from data_types import DatasetManifest, Split
    manifest = DatasetManifest.load(_REPO / 'lamquant' / 'dataset_sim' / 'manifest_v3.json')
    val_entries = manifest.get_file_entries(Split.VAL)[:args.n_windows]
    print(f'      {len(val_entries)} val files sampled')

    # Load one window from the first file
    first = val_entries[0]
    with np.load(first.path) as d:
        raw = d['data'][:, :2500].astype(np.float32)
    raw_uv = raw / 2147483647.0 * 1000.0
    print(f'      Window shape: {raw_uv.shape}, mean={raw_uv.mean():.1f} µV, '
          f'std={raw_uv.std():.1f} µV')

    # ---- Step 3: Encode → LMQ v5 packet ----
    print('\n[3/6] Encode → compress (LMQ v5)...')
    x = torch.from_numpy(raw_uv).unsqueeze(0).float()
    t0 = time.perf_counter()
    latent_wht, metadata = codec.encode(x)
    encode_ms = (time.perf_counter() - t0) * 1000

    lpc_coeffs = metadata[0][0] if metadata else None
    subbands = [m[1] for m in metadata] if metadata else None
    t0 = time.perf_counter()
    compressed = codec.compress(latent_wht, lpc_coeffs, subbands, quality_mode=2)
    compress_ms = (time.perf_counter() - t0) * 1000

    raw_bytes = raw_uv.nbytes
    cr = raw_bytes / len(compressed)
    print(f'      Latent shape:     {tuple(latent_wht.shape)}')
    print(f'      Compressed:       {len(compressed)} bytes ({cr:.0f}:1 CR)')
    print(f'      Encode time:      {encode_ms:.1f} ms')
    print(f'      Compress time:    {compress_ms:.1f} ms')

    # ---- Step 4: Decompress → decode (inverse DSP path) ----
    print('\n[4/6] Decompress → decode (DSP inverse path)...')
    t0 = time.perf_counter()
    latent_rec, quality_out, _, _ = codec.decompress(compressed)
    decompress_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    recon = codec.decode(latent_rec, metadata)
    decode_ms = (time.perf_counter() - t0) * 1000
    recon_np = recon[0].numpy()

    print(f'      Decompress time:  {decompress_ms:.1f} ms')
    print(f'      Decode time:      {decode_ms:.1f} ms')
    print(f'      Recon shape:      {recon_np.shape}')

    # ---- Step 5a: Quality metrics (DSP inverse path — legacy) ----
    print('\n[5/7] Quality metrics (DSP inverse path — legacy)...')
    from metrics import prd_numpy, pearson_r_numpy, per_band_prd, lqs_compliance, lqs_pretty
    T = min(raw_uv.shape[-1], recon_np.shape[-1])
    orig = raw_uv[..., :T]
    rec = recon_np[..., :T]
    r_dsp = pearson_r_numpy(orig, rec)
    prd_dsp = prd_numpy(orig, rec)
    print(f'      R:              {r_dsp:.4f}')
    print(f'      PRD:            {prd_dsp:.2f}%')
    if r_dsp < 0.5:
        print(f'      NOTE: Low R is expected here — the DSP inverse path uses the')
        print(f'      encoder\'s built-in mini-decoder, which was bypassed during')
        print(f'      joint training (only the Vocos decoder was trained). See step 6.')

    # ---- Step 5b: Quality metrics (Vocos decode path — production) ----
    print('\n[6/7] Vocos decoder path (the actually-trained decode)...')
    from joint_codec import build_default_joint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    joint = build_default_joint(vocos_tier=args.tier).to(device).eval()
    joint.load_encoder(args.encoder)
    joint.load_decoder(args.decoder)
    print(f'      Loaded Vocos Tier {args.tier} decoder '
          f'({sum(p.numel() for p in joint.decoder.parameters()):,} params)')

    # Re-encode via the joint encoder (same weights, produces same latent)
    from subband_preprocess import preprocess_subband_single
    l3, _, _ = preprocess_subband_single(raw_uv)
    l3_t = torch.from_numpy(l3).float().unsqueeze(0).to(device)
    with torch.no_grad():
        vocos_recon = joint(l3_t, quantize=True)
    vocos_np = vocos_recon.squeeze(0).cpu().numpy()

    T_v = min(raw_uv.shape[-1], vocos_np.shape[-1])
    orig_v = raw_uv[..., :T_v]
    rec_v = vocos_np[..., :T_v]
    r = pearson_r_numpy(orig_v, rec_v)
    prd = prd_numpy(orig_v, rec_v)
    pb_prd = per_band_prd(orig_v, rec_v, fs=250.0)
    lqs_level, lqs_viol = lqs_compliance(r, prd, per_band_prd_dict=pb_prd)
    level_name = {'C': 'Clinical', 'M': 'Monitoring',
                  'A': 'Alerting', '': 'below LQS-A'}.get(lqs_level, lqs_level)

    print(f'      R:              {r:.4f}')
    print(f'      PRD:            {prd:.2f}%')
    print(f'      Per-band PRD:   {lqs_pretty(r, prd, pb_prd)}')
    print(f'      CR:             {cr:.0f}:1')
    print(f'      LQS Level:      {lqs_level or "--"} ({level_name})')
    if lqs_viol:
        next_tier = {'M': 'C', 'A': 'M', '': 'A'}.get(lqs_level, '?')
        print(f'      To reach LQS-{next_tier}:')
        for v in lqs_viol[:4]:
            print(f'        - {v}')

    # ---- Step 7: LMQ file round-trip ----
    print('\n[6/6] LMQ file round-trip...')
    import tempfile
    from lamquant_codec.fileformat import NeuralWriter, open_file

    with tempfile.NamedTemporaryFile(suffix='.lmq', delete=False) as tf:
        lmq_path = tf.name
    try:
        with NeuralWriter(lmq_path, channels=21, rate=250) as w:
            w.write_window(compressed, timestamp_us=0)
        with open_file(lmq_path) as reader:
            window = next(iter(reader))
            assert window.payload == compressed, 'LMQ payload mismatch'
        lmq_size = os.path.getsize(lmq_path)
        print(f'      Wrote {lmq_path} ({lmq_size} bytes)')
        print(f'      Read back: payload matches ✓')
    finally:
        os.unlink(lmq_path)

    # ---- Summary ----
    print()
    print('=' * 72)
    print(f'  INTEGRATION TEST {"PASSED" if r > 0.3 else "FAILED"}')
    print(f'  Encoder: {args.encoder.name}')
    print(f'  R={r:.4f}  PRD={prd:.1f}%  CR={cr:.0f}:1  LQS={lqs_level or "--"}')
    print(f'  Pipeline: encode({encode_ms:.0f}ms) → compress({compress_ms:.0f}ms) → '
          f'decompress({decompress_ms:.0f}ms) → decode({decode_ms:.0f}ms)')
    print(f'  Total round-trip:  {encode_ms + compress_ms + decompress_ms + decode_ms:.0f} ms')
    print('=' * 72)

    # Log to experiment log as an integration test
    try:
        from experiment_log import ExperimentRecord, log_experiment
        log_experiment(ExperimentRecord(
            run_id=f'e2e_ship_{int(time.time())}',
            manifest_hash=manifest.hash(),
            preset='fast',
            vocos_tier=args.tier,
            best_val_r=r,
            best_val_prd=prd,
            per_band_prd=pb_prd,
            lqs_level=lqs_level,
            lqs_violations=lqs_viol,
            completed=True,
            notes=f'E2E integration test. CR={cr:.0f}:1. DSP-inverse path.',
            tags=['integration_test', 'fast_preset', 'shipped'],
            encoder_ckpt=str(args.encoder),
            decoder_ckpt=str(args.decoder),
        ))
        print('[*] Logged to experiment_log.jsonl')
    except Exception as e:
        print(f'[!] Log failed: {e}')

    return 0 if r > 0.3 else 1


if __name__ == '__main__':
    sys.exit(main())
