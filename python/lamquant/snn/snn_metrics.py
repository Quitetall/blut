#!/usr/bin/env python3
"""SNN metric-card compiler — the FULL field-standard vector per benchmark.

There is no one "SOTA number". This compiles three benchmarks, each measured
its own way, NEVER as a lone scalar (ADR-0029: report the operating point):

  IN-SAMPLE seizure   (train+test same corpus, e.g. TUSZ val) — the headline regime
  CROSS-DATASET seizure (train anywhere, test on an UNSEEN corpus, e.g. Siena)
                        — the generalization number that matters; SzCORE ceiling
                          ~37-58% sens / F1 30-43%.

Per benchmark it emits:
  * window-level:  ROC-AUC, PR-AUC, per-class P/R/F1 (4-state)
  * event-level (NEDC OVLP any-overlap): the SENSITIVITY-vs-FA/h CURVE, the
    Event-Sens@FA AUC (area under sens vs log10 FA/h), sensitivity at fixed FA
    operating points (@1/h, @1/day, @6/day), and the cost-calibrated point.
  * CR-controller: realized CR + false-escalation (the bit-allocation cost).
Every number is tagged with (n_recordings, n_events, hours, scoring, dataset).

The CRITICAL probability stream is softmax(class_logits)[CRITICAL] — the
controller's actual decision signal (NOT the dedicated seizure head, which is
arch-drifted on older checkpoints).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.pop("SNN_DETAIL_BANDS", None)

from lamquant.snn.lma_dataset import LmaDataset  # noqa: E402
from lamquant.snn.train_4state_controller import derive_batch_targets  # noqa: E402
from lamquant.snn.event_scoring import (  # noqa: E402
    events_from_binary, events_from_probs, ovlp_score, event_fpr_per_hour,
    SEC_PER_STEP_L3,
)
from lamquant.snn.snn_event_eval import infer_arch  # noqa: E402
from lamquant_neural.models.mamba_ssm_minimal import MambaSNN  # noqa: E402
from lamquant_neural.models.heads import build_head  # noqa: E402

CRIT = 3
L3_T = 313


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_m, sd_h = ck["model"], ck["head"]
    in_ch, d_model, d_state, n_layers = infer_arch(sd_m)
    use_spectral = in_ch != 21
    head_kind = ck.get("head_kind", "attention_softmax")
    quiet_thr = float(ck["quiet_rms_threshold"])
    model = MambaSNN(in_channels=in_ch, d_model=d_model, d_state=d_state,
                     n_layers=n_layers, use_subband=True).to(device)
    head = build_head(head_kind, K=4).to(device)
    miss, _ = model.load_state_dict(sd_m, strict=False)
    bad = [k for k in miss if not k.startswith("seizure_head")]
    if bad:
        raise RuntimeError(f"4-state-path keys missing: {bad}")
    head.load_state_dict(sd_h)
    model.eval()
    head.eval()
    return model, head, use_spectral, quiet_thr, in_ch


def build_recording_streams(model, head, ds, use_spectral, quiet_thr, in_ch,
                            device, max_recordings=None):
    """Per-recording (crit_prob_1d, true_crit_mask_1d) from the 4-state head."""
    if use_spectral:
        from lamquant.snn.spectral import build_augmented_input
    by_stem = defaultdict(list)
    for i, entry in enumerate(ds.index):
        by_stem[entry[1]].append((entry[2], i))   # stem -> [(win_idx, ds_idx)]
    stems = sorted(by_stem)
    if max_recordings:
        stems = stems[:max_recordings]
    streams = []
    for stem in stems:
        order = sorted(by_stem[stem])              # by win_idx -> contiguous timeline
        probs_parts, true_parts = [], []
        for _wi, di in order:
            sig, lab = ds[di]
            sig = sig.unsqueeze(0).float().to(device)
            lab = lab.unsqueeze(0).to(device)
            if sig.shape[1] != in_ch:
                continue
            x = build_augmented_input(sig) if use_spectral else sig
            with torch.no_grad():
                act, _, _ = model(x)
                _s, cl = head(act, L3_T)
                p = torch.softmax(cl, dim=1)[0, CRIT].cpu().numpy()   # [313]
            tgt = derive_batch_targets(lab, sig, quiet_thr, L3_T)[0].cpu().numpy()
            probs_parts.append(p.astype(np.float64))
            true_parts.append((tgt == CRIT).astype(np.int8))
        if probs_parts:
            streams.append((np.concatenate(probs_parts),
                            np.concatenate(true_parts)))
    return streams


def event_curve(streams, thresholds):
    """Sweep thresholds -> pooled (event_sens, FA/h) curve + window arrays."""
    total_secs = sum(len(p) for p, _ in streams) * SEC_PER_STEP_L3
    # true events fixed across thresholds
    true_events = [events_from_binary(t.astype(bool)) for _, t in streams]
    n_true_total = sum(len(e) for e in true_events)
    curve = []
    for th in thresholds:
        det = fp = 0
        for (probs, _t), te in zip(streams, true_events):
            pe = events_from_probs(probs, float(th))
            sc = ovlp_score(pe, te)
            det += int(sc["n_true_detected"])
            fp += int(sc["n_false_pred"])
        sens = det / n_true_total if n_true_total else 0.0
        fa_h = event_fpr_per_hour(fp, total_secs)
        curve.append({"threshold": round(float(th), 3),
                      "event_sens": round(sens, 4),
                      "fa_per_h": round(fa_h, 4),
                      "fa_per_day": round(fa_h * 24, 3)})
    return curve, n_true_total, total_secs / 3600.0


def sens_at_fa(curve, fa_h_target):
    """Max event_sens among operating points with fa_per_h <= target."""
    elig = [c for c in curve if c["fa_per_h"] <= fa_h_target]
    return round(max((c["event_sens"] for c in elig), default=0.0), 4)


def sens_fa_auc(curve, fa_lo=0.1, fa_hi=24.0):
    """Area under event_sens vs log10(FA/h) over [fa_lo, fa_hi] (per-day-ish
    range), normalized to [0,1]. The NeuroAtlas Event-Sens@FA AUC analog."""
    pts = sorted([(max(c["fa_per_h"], 1e-3), c["event_sens"]) for c in curve])
    xs, ys = [], []
    for fa, s in pts:
        if fa_lo <= fa <= fa_hi:
            xs.append(np.log10(fa))
            ys.append(s)
    if len(xs) < 2:
        return None
    # trapezoidal integral over (log10 FA/h, sens); np.trapz removed in numpy 2.x
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return round(float(area / (np.log10(fa_hi) - np.log10(fa_lo))), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lma-root", nargs="+", required=True, type=Path)
    p.add_argument("--split-manifest", required=True, type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--benchmark", required=True,
                   help="tag: in_sample | cross_dataset")
    p.add_argument("--dataset-name", required=True, help="e.g. TUSZ-val, Siena")
    p.add_argument("--max-recordings", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    model, head, use_spectral, quiet_thr, in_ch = load_model(args.checkpoint, dev)
    print(f"[metrics] {args.benchmark}/{args.dataset_name}: in_ch={in_ch} "
          f"spectral={use_spectral}")

    paths = []
    for r in args.lma_root:
        paths += sorted(Path(r).glob("*/*.lma")) or sorted(Path(r).glob("*.lma"))
    ds = LmaDataset(lma_paths=paths, split=args.split,
                    split_manifest_path=args.split_manifest)
    print(f"[metrics] dataset windows: {len(ds)}")

    streams = build_recording_streams(model, head, ds, use_spectral, quiet_thr,
                                       in_ch, dev, args.max_recordings)
    print(f"[metrics] {len(streams)} recordings streamed")

    # ---- window-level ----
    all_p = np.concatenate([p for p, _ in streams])
    all_t = np.concatenate([t for _, t in streams]).astype(int)
    from sklearn.metrics import roc_auc_score, average_precision_score
    win = {}
    if all_t.any() and (all_t == 0).any():
        win["roc_auc"] = round(float(roc_auc_score(all_t, all_p)), 4)
        win["pr_auc"] = round(float(average_precision_score(all_t, all_p)), 4)
    win["crit_prevalence"] = round(float(all_t.mean()), 5)
    win["n_timesteps"] = int(all_t.size)

    # ---- event-level vector ----
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    curve, n_events, hours = event_curve(streams, thresholds)
    ev = {
        "scoring": "NEDC OVLP any-overlap, 4-state CRITICAL softmax prob",
        "n_events": n_events,
        "hours": round(hours, 2),
        "sens_at_fa": {
            "1_per_h": sens_at_fa(curve, 1.0),
            "6_per_day": sens_at_fa(curve, 6.0 / 24),
            "1_per_day": sens_at_fa(curve, 1.0 / 24),
        },
        "event_sens_fa_auc": sens_fa_auc(curve),
        "curve": curve,
    }

    card = {
        "benchmark": args.benchmark,
        "dataset": args.dataset_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "n_recordings": len(streams),
        "caveat": ("ABSOLUTE numbers tied to THIS scoring (OVLP any-overlap, "
                   "min_event 2s) + THIS label scheme; cross-benchmark "
                   "comparison only within identical scoring. Not interchangeable "
                   "with in-sample vs cross-dataset rows."),
        "window_level": win,
        "event_level": ev,
    }
    args.out.write_text(json.dumps(card, indent=2))

    # ---- readable summary ----
    print(f"\n[metrics] ===== {args.benchmark} / {args.dataset_name} =====")
    print(f"  recordings={len(streams)}  events={n_events}  hours={hours:.1f}  "
          f"crit_prev={win['crit_prevalence']}")
    print(f"  WINDOW   ROC-AUC={win.get('roc_auc','n/a')}  PR-AUC={win.get('pr_auc','n/a')}")
    print(f"  EVENT    sens@1/h={ev['sens_at_fa']['1_per_h']}  "
          f"sens@6/day={ev['sens_at_fa']['6_per_day']}  "
          f"sens@1/day={ev['sens_at_fa']['1_per_day']}  "
          f"Sens@FA-AUC={ev['event_sens_fa_auc']}")
    print("  CURVE (sens @ FA/h):  " + "  ".join(
        f"{c['event_sens']:.2f}@{c['fa_per_h']:.1f}" for c in curve[::3]))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
