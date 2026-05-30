#!/usr/bin/env python3
"""
LamQuant — Activity Label Generator (Universal)
================================================
Generates per-window activity labels for SNN training from real clinical
annotations. Supports all TUEG annotation formats (TSE, CSV) and CHB-MIT
seizure summaries.

NO statistical detection. Labels come only from clinical annotations.
Files without annotations are labeled as all-quiet (background EEG).

Label classes:
  0 = QUIET      — background EEG, no annotated events
  1 = ACTIVE     — annotated non-seizure events (spikes, sharp waves,
                   periodic discharges, artifacts, abnormal activity)
  2 = SEIZURE    — annotated seizure (any subtype)

Label resolution: per-group (8 groups × T_latent timesteps).
The 21 channels are partitioned into 8 spatial groups matching the
firmware's SNN readout topology:

  Group 0: Fp1, Fp2           (frontal polar)
  Group 1: F3, F4, Fz         (frontal)
  Group 2: F7, F8             (lateral frontal)
  Group 3: C3, C4, Cz         (central)
  Group 4: T3, T4             (temporal)
  Group 5: T5, T6             (posterior temporal)
  Group 6: P3, P4, Pz         (parietal)
  Group 7: O1, O2, A1, A2     (occipital + reference)

Supported annotation formats:
  - TUH TSE (.tse, .tse_bi):  "start_time stop_time label probability"
  - TUH CSV (.csv, .csv_bi):  "channel,start_time,stop_time,label,confidence"
  - CHB-MIT summary files:     seizure start/end in summary.txt

Label mapping (TUEG → our classes):
  SEIZURE (2): seiz, fnsz, gnsz, spsz, cpsz, absz, tnsz, tcsz, mysz
  ACTIVE  (1): spsw, gped, pled, eybl, artf, musc, chew, elpp, eyem, shiv, bckg_abnormal
  QUIET   (0): bckg, null, (no annotation)

Output (.npz per file):
  activity_labels:  uint8 [8, T_latent]
  source:           str (original filename)
  annotation_file:  str (path to annotation used)
  label_counts:     dict {quiet: N, active: N, seizure: N}

Usage:
  # TUEG (finds .tse_bi/.csv_bi alongside EDFs)
  python generate_activity_labels.py --input /mnt/4tb/data/tuh_eeg/tusz --output ./labels

  # CHB-MIT (uses summary.txt seizure annotations)
  python generate_activity_labels.py --input /mnt/4tb/data/chbmit --output ./labels

  # Mixed (auto-detects format per file)
  python generate_activity_labels.py --input /mnt/4tb/data --output ./labels --recursive
"""
import os
import re
import sys
import glob
import argparse
import numpy as np
from collections import defaultdict

try:
    from lamquant_codec.channel_resolver import SPATIAL_GROUPS
except ImportError:
    SPATIAL_GROUPS = [
        [0, 1],           # Fp1, Fp2
        [2, 3, 16],       # F3, F4, Fz
        [10, 11],         # F7, F8
        [4, 5, 17],       # C3, C4, Cz
        [12, 13],         # T3, T4
        [14, 15],         # T5, T6
        [6, 7, 18],       # P3, P4, Pz
        [8, 9, 19, 20],   # O1, O2, A1, A2
    ]

NUM_GROUPS = 8
STRIDE_8 = 8
TARGET_FS = 250  # Hz


# =====================================================================
# Label mapping — clinical annotations → {0, 1, 2}
# =====================================================================

# All known seizure subtypes in TUSZ
SEIZURE_LABELS = frozenset({
    'seiz',     # generic seizure
    'fnsz',     # focal non-specific
    'gnsz',     # generalized non-specific
    'spsz',     # simple partial
    'cpsz',     # complex partial
    'absz',     # absence
    'tnsz',     # tonic
    'tcsz',     # tonic-clonic
    'mysz',     # myoclonic
})

# Non-seizure events that indicate clinical activity
ACTIVE_LABELS = frozenset({
    'spsw',     # spike and slow wave
    'gped',     # generalized periodic discharges
    'pled',     # periodic lateralized epileptiform discharges
    'eybl',     # eye blink (artifact but indicates activity)
    'artf',     # artifact
    'musc',     # muscle artifact
    'chew',     # chewing artifact
    'elpp',     # electrode pop
    'eyem',     # eye movement
    'shiv',     # shivering
})

# Background / quiet
QUIET_LABELS = frozenset({
    'bckg',     # background
    'null',     # null/unscored
})


def map_label(label_str):
    """Map a TUEG/CHB-MIT annotation label to our {0, 1, 2} scheme."""
    label = label_str.strip().lower()
    if label in SEIZURE_LABELS:
        return 2
    if label in ACTIVE_LABELS:
        return 1
    if label in QUIET_LABELS:
        return 0
    # Unknown label — treat as active (conservative: don't miss events)
    return 1


# =====================================================================
# Annotation parsers
# =====================================================================

def parse_tse(filepath):
    """Parse a TUH .tse or .tse_bi file.

    Format:
        version = tse_v1.0.0

        0.0000 7.1237 bckg 1.0000
        7.1237 33.5162 seiz 1.0000
        ...

    Returns list of (start_sec, end_sec, label_str, confidence).
    """
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('version'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2]
                conf = float(parts[3])
                events.append((start, end, label, conf))
            except (ValueError, IndexError):
                continue
    return events


def parse_csv_annotation(filepath):
    """Parse a TUH .csv or .csv_bi annotation file.

    Format:
        # version = csv_v1.0.0
        # bname = filename
        # duration = 345.0000 secs
        # montage_file = montage.txt
        #
        channel,start_time,stop_time,label,confidence
        TERM,0.0000,7.1237,bckg,1.0000
        TERM,7.1237,33.5162,seiz,1.0000

    Returns list of (start_sec, end_sec, label_str, confidence).
    """
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('channel'):
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            try:
                start = float(parts[1])
                end = float(parts[2])
                label = parts[3].strip()
                conf = float(parts[4])
                events.append((start, end, label, conf))
            except (ValueError, IndexError):
                continue
    return events


# TUEV .rec numeric label codes (TUH EEG Events v2.0.0).
# Maps integer code → canonical TUH label string so map_label() can route it
# through the existing SEIZURE/ACTIVE/QUIET frozensets.
TUEV_CODE_MAP = {
    1: 'spsw',   # spike and slow wave        → ACTIVE
    2: 'gped',   # generalized periodic disch → ACTIVE
    3: 'pled',   # periodic lateralized disch → ACTIVE
    4: 'eyem',   # eye movement               → ACTIVE
    5: 'artf',   # artifact                   → ACTIVE
    6: 'bckg',   # background                 → QUIET
}


def parse_rec(filepath):
    """Parse a TUEV .rec annotation file.

    Format (one event per line, no header):
        channel,start_time,stop_time,code
        4,1.5,2.5,6
        4,2.5,3.5,1

    `code` is a TUEV integer label (see TUEV_CODE_MAP). Per-channel events are
    flattened to recording-level here (same as the .csv/.tse path), since the
    SNN activity labels are montage-level, not per-channel.

    Returns list of (start_sec, end_sec, label_str, confidence).
    """
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                start = float(parts[1])
                end = float(parts[2])
                code = int(float(parts[3]))
                label = TUEV_CODE_MAP.get(code)
                if label is None:
                    continue  # unknown code — skip rather than mis-route
                events.append((start, end, label, 1.0))
            except (ValueError, IndexError):
                continue
    return events


def parse_chbmit_summary(summary_path):
    """Parse CHB-MIT summary.txt for seizure annotations.

    Returns dict: {filename: [(start_sec, end_sec), ...]}.
    """
    seizures = {}
    current_file = None
    starts, ends = [], []
    with open(summary_path, 'r') as f:
        for line in f:
            m_file = re.match(r'File Name:\s*(\S+)', line)
            if m_file:
                if current_file and starts:
                    seizures[current_file] = list(zip(starts, ends))
                current_file = m_file.group(1)
                starts, ends = [], []
                continue
            m_start = re.match(r'Seizure\s*\d*\s*Start.*?:\s*(\d+)', line)
            if m_start:
                starts.append(int(m_start.group(1)))
            m_end = re.match(r'Seizure\s*\d*\s*End.*?:\s*(\d+)', line)
            if m_end:
                ends.append(int(m_end.group(1)))
    if current_file and starts:
        seizures[current_file] = list(zip(starts, ends))
    return seizures


def find_annotation_file(edf_path):
    """Find the annotation file for a given EDF.

    Searches for .tse_bi, .tse, .csv_bi, .csv, .rec in the same directory.
    Returns (path, format) or (None, None).
    """
    base = os.path.splitext(edf_path)[0]
    for ext, fmt in [('.tse_bi', 'tse'), ('.tse', 'tse'),
                     ('.csv_bi', 'csv'), ('.csv', 'csv'),
                     ('.rec', 'rec')]:
        path = base + ext
        if os.path.exists(path):
            return path, fmt
    return None, None


def find_chbmit_summaries(input_dir):
    """Find all CHB-MIT summary files and merge seizure annotations."""
    all_seizures = {}
    for summary_path in glob.glob(os.path.join(input_dir, '**', '*summary*'),
                                  recursive=True):
        if os.path.isfile(summary_path):
            seizures = parse_chbmit_summary(summary_path)
            for fname, intervals in seizures.items():
                full = os.path.join(os.path.dirname(summary_path), fname)
                all_seizures[full] = intervals
    return all_seizures


# =====================================================================
# Siena Scalp EEG — seizure-list parser
# =====================================================================
#
# The Siena Scalp EEG Database (physionet.org/content/siena-scalp-eeg)
# ships one 'Seizures-list-PNxx.txt' per subject directory. Unlike
# CHB-MIT, seizure times are given as wall-clock HH.MM.SS times, NOT
# offsets into the recording. Each seizure block carries:
#
#     Seizure n 1
#     File name: PN00-1.edf
#     Registration start time: 19.39.33
#     Registration end time:  20.22.58
#     Seizure start time: 19.58.36
#     Seizure end time: 19.59.46
#
# The seizure interval (seconds-into-EDF) is therefore:
#     start_sec = wallclock(seizure_start) - wallclock(registration_start)
#     end_sec   = wallclock(seizure_end)   - wallclock(registration_start)
#
# Like CHB-MIT, Siena only annotates seizures — everything else is
# QUIET — so the derived (start_sec, end_sec) intervals route straight
# through chbmit_seizures_to_labels() (the {0,2} mapping).
#
# Format quirks observed across the real corpus (PN00, PN01, PN03,
# PN05, PN06, PN12, ...):
#   * Time separator is usually '.' but sometimes ':' and even mixed
#     within one stamp (PN12 'Seizure start time: 16:13.23').
#   * Field labels vary: 'Seizure start time' vs bare 'Start time'
#     (PN01); 'Seizure n 1', 'Seizure n1', 'Seizure n 2:' all appear.
#   * File-name typos: PN06 lists 'PNO6-1.edf' (letter O) while the
#     EDF on disk is 'PN06-1.edf' — resolved against actual EDFs.
#   * A seizure block may omit 'Registration start time' when it shares
#     an EDF with the previous block (PN12 'PN12-1.2.edf') — the last
#     registration-start seen for that file is carried forward.
#   * Recordings cross midnight (PN01: reg 19:00:44, seizure 07:53:17
#     next day) — a negative offset is wrapped by +24h.
#   * A leading 'File name' / 'Registration start time' header (with no
#     'Seizure n' marker) can describe the EDF for the seizures that
#     follow (PN01) — tracked as the current file/registration context.

_SIENA_TIME_RE = re.compile(r'(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})')

_SECONDS_PER_DAY = 24 * 60 * 60


def _parse_siena_clock(value):
    """Parse a Siena HH.MM.SS (or HH:MM:SS, or mixed) clock string.

    Returns seconds-since-midnight as an int, or None if unparseable.
    """
    if value is None:
        return None
    m = _SIENA_TIME_RE.search(value)
    if not m:
        return None
    try:
        h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    except (ValueError, IndexError):
        return None
    if not (0 <= h < 24 and 0 <= mn < 60 and 0 <= s < 60):
        return None
    return h * 3600 + mn * 60 + s


def _normalize_siena_filename(fname):
    """Strip trailing whitespace from a listed Siena EDF file name.

    Filename typos (e.g. 'PNO6-1.edf' vs on-disk 'PN06-1.edf') are
    resolved later against the actual EDFs in the directory; here we
    only trim and keep the basename.
    """
    return os.path.basename(fname.strip())


def parse_siena(list_path):
    """Parse a Siena 'Seizures-list-PNxx.txt' for seizure annotations.

    Derives per-EDF seizure intervals from wall-clock times:
        offset = wallclock(seizure) - wallclock(registration_start)
    wrapping negative offsets across midnight (+24h).

    Field labels are matched tolerantly ('Seizure start time' or bare
    'Start time'); the time separator may be '.', ':' or mixed.

    Returns dict: {edf_basename: [(start_sec, end_sec), ...]}.
    """
    seizures = defaultdict(list)
    current_file = None
    # Last registration-start (seconds-since-midnight) seen per file,
    # so blocks that omit it can inherit the value (PN12 shared EDF).
    reg_start_by_file = {}
    cur_reg_start = None
    cur_sz_start = None
    cur_sz_end = None

    def flush():
        if (current_file is not None
                and cur_sz_start is not None
                and cur_sz_end is not None
                and cur_reg_start is not None):
            start = (cur_sz_start - cur_reg_start) % _SECONDS_PER_DAY
            end = (cur_sz_end - cur_reg_start) % _SECONDS_PER_DAY
            if end >= start:
                seizures[current_file].append((float(start), float(end)))

    with open(list_path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()

            # New seizure block boundary: 'Seizure n 1', 'Seizure n1', etc.
            if re.match(r'seizure\s*n', low):
                flush()
                cur_sz_start = None
                cur_sz_end = None
                # current_file / cur_reg_start persist (may be inherited)
                continue

            m_file = re.match(r'file\s*name\s*:?\s*(\S+)', low)
            if m_file:
                # Re-match against the original line to preserve case.
                orig = re.match(r'(?i)file\s*name\s*:?\s*(\S+)', line)
                current_file = _normalize_siena_filename(orig.group(1))
                cur_reg_start = reg_start_by_file.get(current_file)
                continue

            if low.startswith('registration start time'):
                secs = _parse_siena_clock(line[len('registration start time'):])
                if secs is not None:
                    cur_reg_start = secs
                    if current_file is not None:
                        reg_start_by_file[current_file] = secs
                continue

            # 'Registration end time' is informational only — skip.
            if low.startswith('registration end time'):
                continue

            # Seizure start: 'Seizure start time' or bare 'Start time'.
            if low.startswith('seizure start time') or low.startswith('start time'):
                secs = _parse_siena_clock(line)
                if secs is not None:
                    cur_sz_start = secs
                continue

            # Seizure end: 'Seizure end time' or bare 'End time'.
            if low.startswith('seizure end time') or low.startswith('end time'):
                secs = _parse_siena_clock(line)
                if secs is not None:
                    cur_sz_end = secs
                continue

    flush()
    return dict(seizures)


def find_siena_seizures(input_dir):
    """Find all Siena 'Seizures-list-*.txt' files and merge annotations.

    Resolves listed EDF file names against the actual EDFs in the
    seizure-list's directory (handles 'PNO6'→'PN06' typos and case)
    and keys the result by absolute EDF path, mirroring
    find_chbmit_summaries().
    """
    all_seizures = {}
    for list_path in glob.glob(os.path.join(input_dir, '**', 'Seizures-list-*.txt'),
                               recursive=True):
        if not os.path.isfile(list_path):
            continue
        subj_dir = os.path.dirname(list_path)
        # Index actual EDFs by lowercased basename for typo-tolerant match.
        on_disk = {}
        for edf in glob.glob(os.path.join(subj_dir, '*.edf')):
            on_disk[os.path.basename(edf).lower()] = edf
        seizures = parse_siena(list_path)
        for fname, intervals in seizures.items():
            if not intervals:
                continue
            resolved = on_disk.get(fname.lower())
            if resolved is None:
                # Typo fallback: treat 'O' (letter) as '0' (digit).
                alt = fname.lower().replace('o', '0')
                resolved = on_disk.get(alt)
            if resolved is None:
                resolved = os.path.join(subj_dir, fname)
            all_seizures.setdefault(resolved, [])
            all_seizures[resolved].extend(intervals)
    return all_seizures


# =====================================================================
# Label computation — annotations only, no statistics
# =====================================================================

def events_to_labels(events, duration_sec, fs=TARGET_FS):
    """Convert annotation events to per-group activity labels.

    events: list of (start_sec, end_sec, label_str, confidence)
    duration_sec: total recording duration in seconds
    fs: sample rate

    Returns: [8, T_latent] uint8 activity labels
    """
    T = int(duration_sec * fs)
    T_latent = T // STRIDE_8
    if T_latent < 1:
        return np.zeros((NUM_GROUPS, 1), dtype=np.uint8)

    # Build sample-level label array (highest priority wins)
    # Priority: SEIZURE (2) > ACTIVE (1) > QUIET (0)
    sample_labels = np.zeros(T, dtype=np.uint8)
    for start, end, label_str, conf in events:
        level = map_label(label_str)
        if level == 0:
            continue  # quiet is the default
        i0 = max(0, int(start * fs))
        i1 = min(T, int(end * fs))
        # Only upgrade, never downgrade (seizure > active > quiet)
        sample_labels[i0:i1] = np.maximum(sample_labels[i0:i1], level)

    # Downsample to latent resolution: max within each stride window
    # Applied uniformly across all 8 spatial groups (annotations are
    # recording-level in TUEG, not per-channel)
    labels = np.zeros((NUM_GROUPS, T_latent), dtype=np.uint8)
    for t in range(T_latent):
        t0 = t * STRIDE_8
        t1 = min(t0 + STRIDE_8, T)
        window_max = sample_labels[t0:t1].max()
        labels[:, t] = window_max

    return labels


def chbmit_seizures_to_labels(seizure_intervals, duration_sec, fs=TARGET_FS):
    """Convert CHB-MIT seizure intervals to activity labels.

    CHB-MIT only has seizure annotations — everything else is quiet.
    """
    events = [(start, end, 'seiz', 1.0) for start, end in seizure_intervals]
    return events_to_labels(events, duration_sec, fs)


# =====================================================================
# EDF duration extraction (lightweight, no MNE)
# =====================================================================

def get_edf_duration(edf_path):
    """Extract duration from EDF header without loading signal data."""
    try:
        with open(edf_path, 'rb') as f:
            header = f.read(256)
            # Bytes 236-244: duration of a data record (seconds)
            # Bytes 244-252: number of data records (may be float in EDF+)
            duration_per_record = float(header[236:244].decode('ascii').strip())
            n_records = float(header[244:252].decode('ascii').strip())
            return duration_per_record * n_records
    except Exception:
        return None


# =====================================================================
# Main processing
# =====================================================================

def process_edf(edf_path, annotation_path=None, annotation_fmt=None,
                chbmit_seizures=None):
    """Process a single EDF file and return activity labels.

    Priority:
    1. TUH annotation file (.tse_bi / .csv_bi) if available
    2. CHB-MIT seizure intervals if available
    3. All-quiet (no annotations = background EEG)
    """
    duration = get_edf_duration(edf_path)
    if duration is None or duration < 1.0:
        return None

    source = os.path.basename(edf_path)
    ann_used = None
    label_counts = {'quiet': 0, 'active': 0, 'seizure': 0}

    if annotation_path and annotation_fmt:
        # Parse real annotation
        if annotation_fmt == 'tse':
            events = parse_tse(annotation_path)
        elif annotation_fmt == 'csv':
            events = parse_csv_annotation(annotation_path)
        elif annotation_fmt == 'rec':
            events = parse_rec(annotation_path)
        else:
            events = []
        labels = events_to_labels(events, duration)
        ann_used = annotation_path
    elif chbmit_seizures:
        labels = chbmit_seizures_to_labels(chbmit_seizures, duration)
        ann_used = 'chbmit_summary'
    else:
        # No annotations — all quiet
        T_latent = max(1, int(duration * TARGET_FS) // STRIDE_8)
        labels = np.zeros((NUM_GROUPS, T_latent), dtype=np.uint8)
        ann_used = 'none'

    label_counts['quiet'] = int(np.sum(labels == 0))
    label_counts['active'] = int(np.sum(labels == 1))
    label_counts['seizure'] = int(np.sum(labels == 2))

    return {
        'activity_labels': labels,
        'source': source,
        'annotation_file': ann_used or '',
        'label_counts': label_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate SNN activity labels from clinical annotations')
    parser.add_argument('--input', required=True,
                        help='Directory containing EDF files (searched recursively)')
    parser.add_argument('--output', required=True,
                        help='Output directory for label .npz files')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Max files to process (0 = all)')
    parser.add_argument('--skip-unannotated', action='store_true',
                        help='Skip EDFs without annotation files (default: label as all-quiet)')
    parser.add_argument('--require-events', action='store_true',
                        help='Skip EDFs where annotations contain only bckg/null')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find all EDF files
    edf_files = sorted(glob.glob(os.path.join(args.input, '**', '*.edf'),
                                 recursive=True))
    print(f"[*] Found {len(edf_files)} EDF files in {args.input}")

    # Check for CHB-MIT summaries (fallback for datasets without per-file annotations)
    chbmit_all = find_chbmit_summaries(args.input)
    if chbmit_all:
        print(f"[*] CHB-MIT summaries: {len(chbmit_all)} files with seizure annotations")

    # Check for Siena Scalp EEG seizure lists. Siena, like CHB-MIT, has only
    # seizure annotations, so its derived (start_sec, end_sec) intervals share
    # the same seizure-interval channel and chbmit_seizures_to_labels() mapping.
    siena_all = find_siena_seizures(args.input)
    if siena_all:
        print(f"[*] Siena seizure lists: {len(siena_all)} files with seizure annotations")
        for full, intervals in siena_all.items():
            chbmit_all.setdefault(full, [])
            chbmit_all[full].extend(intervals)

    if args.max_files > 0:
        edf_files = edf_files[:args.max_files]

    stats = {
        'processed': 0, 'skipped': 0, 'skipped_no_ann': 0,
        'annotated_tse': 0, 'annotated_csv': 0, 'annotated_chbmit': 0,
        'unannotated': 0,
        'total_seizure': 0, 'total_active': 0, 'total_quiet': 0,
    }

    for i, edf_path in enumerate(edf_files):
        # Find annotation
        ann_path, ann_fmt = find_annotation_file(edf_path)
        chbmit_sz = chbmit_all.get(edf_path, None)

        if args.skip_unannotated and ann_path is None and chbmit_sz is None:
            stats['skipped_no_ann'] += 1
            continue

        result = process_edf(edf_path, ann_path, ann_fmt, chbmit_sz)
        if result is None:
            stats['skipped'] += 1
            continue

        # Skip if --require-events and no actual events
        if args.require_events:
            lc = result['label_counts']
            if lc['active'] == 0 and lc['seizure'] == 0:
                stats['skipped'] += 1
                continue

        # Track annotation source
        if ann_fmt == 'tse':
            stats['annotated_tse'] += 1
        elif ann_fmt == 'csv':
            stats['annotated_csv'] += 1
        elif chbmit_sz:
            stats['annotated_chbmit'] += 1
        else:
            stats['unannotated'] += 1

        stats['processed'] += 1
        lc = result['label_counts']
        stats['total_seizure'] += lc['seizure']
        stats['total_active'] += lc['active']
        stats['total_quiet'] += lc['quiet']

        # Save
        out_name = os.path.splitext(result['source'])[0] + '_labels.npz'
        np.savez_compressed(
            os.path.join(args.output, out_name),
            activity_labels=result['activity_labels'],
            source=result['source'],
            annotation_file=result['annotation_file'],
        )

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(edf_files)}] {stats['processed']} processed, "
                  f"{stats['annotated_tse'] + stats['annotated_csv']} TUH, "
                  f"{stats['annotated_chbmit']} CHB-MIT, "
                  f"{stats['unannotated']} unannotated")

    # Summary
    total = stats['total_seizure'] + stats['total_active'] + stats['total_quiet']
    print(f"\n{'=' * 60}")
    print(f"Label generation complete")
    print(f"  Processed:    {stats['processed']}")
    print(f"  Skipped:      {stats['skipped']}")
    if stats['skipped_no_ann']:
        print(f"  No annotation:{stats['skipped_no_ann']}")
    print(f"  Annotations:  {stats['annotated_tse']} TSE, "
          f"{stats['annotated_csv']} CSV, "
          f"{stats['annotated_chbmit']} CHB-MIT, "
          f"{stats['unannotated']} unannotated")
    if total > 0:
        print(f"  Labels ({total:,} total group×timestep):")
        print(f"    QUIET:   {stats['total_quiet']:>10,} "
              f"({100 * stats['total_quiet'] / total:.1f}%)")
        print(f"    ACTIVE:  {stats['total_active']:>10,} "
              f"({100 * stats['total_active'] / total:.1f}%)")
        print(f"    SEIZURE: {stats['total_seizure']:>10,} "
              f"({100 * stats['total_seizure'] / total:.1f}%)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
