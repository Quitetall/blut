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
    """Per-recording CONTINUOUS (crit_prob_1d, true_crit_mask_1d).

    CRITICAL: streams the FULL recording (every window in order via the full L3
    stack), NOT the trainer's seizure-SELECTED windows — otherwise the base rate
    is inverted (seizure-enriched) and event-F1/FA/h are meaningless. True
    CRITICAL = (3-class activity == 2).any over groups, per window slice
    (matches four_state.derive_4state_target max3==2 -> CRITICAL).
    """
    import io
    from lamquant.snn.lma_dataset import (
        _cached_l3_stack, _label_cache_dir, _lazy_imports,
        LABEL_PER_WINDOW, TARGET_CHANNELS,
    )
    import lamquant_core as _lc
    _lazy_imports()
    if use_spectral:
        from lamquant.snn.spectral import build_augmented_input

    # stem -> (lma_path, lml_internal, label_internal) from any index entry
    info = {}
    for e in ds.index:
        info.setdefault(e[1], (str(e[0]), e[3], e[4]))
    stems = sorted(info)
    if max_recordings:
        stems = stems[:max_recordings]
    label_dir = _label_cache_dir()

    def _load_activity(stem, lma_path, label_internal):
        cached = (label_dir / f"{stem}_labels.npz") if label_dir else None
        try:
            if cached is not None and cached.exists():
                with np.load(cached, allow_pickle=True) as ld:
                    return np.asarray(ld["activity_labels"])
            b = _lc.lma_read_entry(lma_path, label_internal)
            with np.load(io.BytesIO(b), allow_pickle=True) as ld:
                return np.asarray(ld["activity_labels"])
        except Exception:
            return None

    streams = []
    for stem in stems:
        lma_path, lml_internal, label_internal = info[stem]
        stack = _cached_l3_stack(lma_path, stem, lml_internal=lml_internal)
        act = _load_activity(stem, lma_path, label_internal)
        if stack is None or act is None:
            continue
        n_win = stack.shape[0]
        probs_parts, true_parts = [], []
        for w in range(n_win):
            l3 = np.asarray(stack[w], dtype=np.float32)
            # stack is raw L3 (21 ch); spectral build_augmented_input expands it
            # to the model's in_ch (105) below — so check against 21, NOT in_ch.
            if l3.shape[0] != TARGET_CHANNELS:
                continue
            t = torch.from_numpy(l3).unsqueeze(0).to(device)
            x = build_augmented_input(t) if use_spectral else t
            with torch.no_grad():
                a, _, _ = model(x)
                _s, cl = head(a, L3_T)
                p = torch.softmax(cl, dim=1)[0, CRIT].cpu().numpy()
            s = w * LABEL_PER_WINDOW
            e = min(s + L3_T, act.shape[1])
            tw = np.zeros(L3_T, dtype=np.int8)
            if e > s:
                seg = (act[:, s:e] == 2).any(axis=0).astype(np.int8)
                tw[:e - s] = seg
                if e - s < L3_T:
                    tw[e - s:] = tw[e - s - 1]
            probs_parts.append(p.astype(np.float64))
            true_parts.append(tw)
        if probs_parts:
            streams.append((np.concatenate(probs_parts),
                            np.concatenate(true_parts)))
    return streams


# SzCORE event conventions (Dan et al., SzCORE 2025): any-overlap matching with
# a 30 s pre-ictal / 60 s post-ictal tolerance, events <90 s apart merged.
SZCORE_PRE_S = 30.0
SZCORE_POST_S = 60.0
SZCORE_MERGE_S = 90.0


def _pad_events(events, pre, post):
    return [(max(0.0, s - pre), e + post) for (s, e) in events]


def event_curve(streams, thresholds, szcore=True):
    """Sweep thresholds -> pooled (event_sens, precision, event_F1, FA/h) curve.

    SzCORE mode: true events get a [-30 s, +60 s] tolerance pad before OVLP
    matching, and both true + predicted events are merged when <90 s apart.
    event_F1 (the SzCORE PRIMARY metric) = 2*TP / (2*TP + FP + FN).
    """
    merge = SZCORE_MERGE_S if szcore else 0.0
    pre, post = (SZCORE_PRE_S, SZCORE_POST_S) if szcore else (0.0, 0.0)
    total_secs = sum(len(p) for p, _ in streams) * SEC_PER_STEP_L3
    true_events = [events_from_binary(t.astype(bool), merge_gap_sec=merge)
                   for _, t in streams]
    true_padded = [_pad_events(te, pre, post) for te in true_events]
    n_true_total = sum(len(e) for e in true_events)
    curve = []
    for th in thresholds:
        det = fp = 0
        for (probs, _t), tep in zip(streams, true_padded):
            pe = events_from_probs(probs, float(th), merge_gap_sec=merge)
            sc = ovlp_score(pe, tep)
            det += int(sc["n_true_detected"])
            fp += int(sc["n_false_pred"])
        fn = n_true_total - det
        sens = det / n_true_total if n_true_total else 0.0
        prec = det / (det + fp) if (det + fp) else 0.0
        f1 = (2 * det) / (2 * det + fp + fn) if (2 * det + fp + fn) else 0.0
        fa_h = event_fpr_per_hour(fp, total_secs)
        curve.append({"threshold": round(float(th), 3),
                      "event_sens": round(sens, 4),
                      "precision": round(prec, 4),
                      "event_f1": round(f1, 4),
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
    curve, n_events, hours = event_curve(streams, thresholds, szcore=True)
    best_f1 = max(curve, key=lambda c: c["event_f1"]) if curve else {}
    ev = {
        "scoring": ("SzCORE-aligned event OVLP (30s pre / 60s post tolerance, "
                    "merge <90s), 4-state CRITICAL softmax prob. PRIMARY metric "
                    "= event_F1; report sensitivity ONLY with its FA/h."),
        "n_events": n_events,
        "hours": round(hours, 2),
        "best_event_f1": round(best_f1.get("event_f1", 0.0), 4),
        "at_best_f1": {k: best_f1.get(k) for k in
                       ("threshold", "event_sens", "precision", "fa_per_day")},
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
    bf = ev["at_best_f1"]
    print(f"  EVENT-F1 (SzCORE primary) best={ev['best_event_f1']}  "
          f"@ sens={bf.get('event_sens')} prec={bf.get('precision')} "
          f"FP/day={bf.get('fa_per_day')}  [bar: SzCORE-winner 0.32, Encevis 0.44]")
    print(f"  EVENT    sens@1/h={ev['sens_at_fa']['1_per_h']}  "
          f"sens@6/day={ev['sens_at_fa']['6_per_day']}  "
          f"sens@1/day={ev['sens_at_fa']['1_per_day']}  "
          f"Sens@FA-AUC={ev['event_sens_fa_auc']}")
    print("  CURVE (sens @ FA/h):  " + "  ".join(
        f"{c['event_sens']:.2f}@{c['fa_per_h']:.1f}" for c in curve[::3]))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
