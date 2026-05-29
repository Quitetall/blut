"""ai_models/data_types.py — typed contracts for the LamQuant data pipeline.

Single source of truth for every boundary in the data flow:

  EDF → EEGWindow → L3Window → TrainingBatch → EpochReport → RunSummary

Every dataclass enforces a contract. Every loader uses the same types.
Bugs in one component cannot silently propagate to the next. Contract
violations (subject leakage, split mismatch, validation_only-in-train,
window-count drift) are caught by `DatasetManifest.validate()` at load
time and by `TrainingBatch.assert_no_leakage()` at runtime.

Design decisions (locked in 2026-04-16):

  * File-level split — every window in a file shares the same split.
    Window-level masks were rejected because (a) temporal continuity
    leaks information across the boundary and (b) per-batch enforcement
    is fragile.

  * Subject-disjoint holdout — no patient_id appears in both TRAIN and
    VAL splits within the same dataset. The validate() method checks
    this on every load.

  * Single source of truth — the typed `DatasetManifest` (manifest_v3.json)
    is the only canonical split. Old `validation_manifest.json` is kept
    as a migration input but never re-read by the training pipeline.

  * TUEV provenance — `tuh_spsw_046_a_1.npz` is parsed as event_type='spsw',
    patient_id='tuev_046' (prefixed to avoid collision with tuh_seizure
    patient '046'). The event_type field carries the clinical annotation;
    rejecting these files would throw away useful data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ============================================================
# Enums — closed sets of values
# ============================================================

class Split(Enum):
    """Which split a file belongs to. File-level — never per-window."""
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'        # reserved; no current consumers
    HOLDOUT = 'holdout'  # validation_only datasets (siena/eegmmidb/mental)


class Dataset(Enum):
    """Canonical dataset identifiers.

    Adding a new dataset requires three coordinated edits: adding it
    here, teaching parse_npz_filename() about its naming convention,
    and (if applicable) adding it to VALIDATION_ONLY_DATASETS.
    """
    CHBMIT = 'chbmit'
    TUH_SEIZURE = 'tuh_seizure'
    TUH_ARTIFACT = 'tuh_artifact'
    TUH_EVENTS = 'tuh_events'
    TUH_EPILEPSY = 'tuh_epilepsy'
    TUEG = 'tueg'                       # TUEG base corpus (no specific annotations)
    SIENA = 'siena'
    EEGMMIDB = 'eegmmidb'
    MENTAL_ARITHMETIC = 'mental_arithmetic'
    SLEEP_EDF = 'sleep_edf'


# Datasets that NEVER appear in training. Cross-dataset validation only.
# Siena is validation-only because it's a different hospital — tests generalization.
VALIDATION_ONLY_DATASETS: frozenset[Dataset] = frozenset({
    Dataset.SIENA,
})


# ============================================================
# Filename parser — fallback only; manifest lookup is preferred
# ============================================================
#
# Used by build_manifest.py for files NOT covered by the v2 manifest
# (notably tuh_epilepsy / TUEP, which is preprocessed but never had a
# v2 entry generated). For files IN the v2 manifest, the builder uses
# the manifest's `subject` field directly — this parser is fallback only.

# TUEV event tags. These four are the ones present on disk; extend if
# new ones appear after re-processing the TUH event corpus.
_TUEV_EVENT_TYPES = ('spsw', 'gped', 'pled', 'bckg',
                     'eyem', 'eyeb', 'musc', 'chew', 'shiv', 'elpp', 'elec')

_TUEV_REGEX = re.compile(
    r'^tuh_(' + '|'.join(_TUEV_EVENT_TYPES) + r')_(\d+)_(\w+?)(?:_(\d+))?_*$'
)
_CHBMIT_REGEX = re.compile(r'^chbmit_(chb\d+)([a-z]?)_([\w+]+)$')
_TUH_PATIENT_REGEX = re.compile(r'^tuh_([a-z]+)_(.+)$')
_TUEP_PATIENT_REGEX = re.compile(r'^tuep_([a-z]+)_(.+)$')


def _strip_npz_suffix(name: str) -> str:
    """Strip _q31.npz / .npz from a filename. Returns just the basename stem."""
    bn = Path(name).name
    if bn.endswith('_q31.npz'):
        return bn[:-len('_q31.npz')]
    if bn.endswith('.npz'):
        return bn[:-len('.npz')]
    return bn


@dataclass(frozen=True)
class ParsedFilename:
    """Result of parse_npz_filename. Every field has a deterministic value."""
    dataset: Dataset
    patient_id: str
    session_id: str
    event_type: str = ''
    segment: Optional[int] = None


def parse_npz_filename(name: str) -> ParsedFilename:
    """Extract structured metadata from an NPZ filename.

    This is the FALLBACK parser. The build script prefers the v2
    manifest's subject/dataset for files it covers; this parser handles
    files not yet in any manifest (e.g. tuh_epilepsy / TUEP).

    Examples:
        >>> parse_npz_filename('chbmit_chb01_03_q31.npz')
        ParsedFilename(dataset=Dataset.CHBMIT, patient_id='chb01',
                       session_id='03', event_type='', segment=None)

        >>> parse_npz_filename('tuh_spsw_046_a_1_q31.npz')
        ParsedFilename(dataset=Dataset.TUH_EVENTS, patient_id='tuev_046',
                       session_id='a', event_type='spsw', segment=1)

        >>> parse_npz_filename('tuh_aaaaaaac_s002_t000_q31.npz')
        ParsedFilename(dataset=Dataset.TUEG, patient_id='aaaaaaac',
                       session_id='s002_t000', event_type='', segment=None)

    Raises ValueError if the filename matches no known pattern.
    """
    bn = _strip_npz_suffix(name)

    # CHBMIT — chbmit_<patient>[<recording_letter>]_<session>
    # chb17a/chb17b/chb17c are sub-recordings of subject 17 collected
    # months apart; the v2 manifest collapses them to subject='chb17',
    # so the parser does the same to preserve subject-disjoint holdout.
    # Sessions can have a '+' suffix (e.g. chb02_16+) for split files.
    if bn.startswith('chbmit_'):
        m = _CHBMIT_REGEX.match(bn)
        if not m:
            raise ValueError(f'unrecognised chbmit filename: {bn!r}')
        recording = m.group(2)  # 'a', 'b', 'c' or ''
        session = m.group(3)
        if recording:
            session = f'{recording}_{session}'   # preserve recording in session_id
        return ParsedFilename(
            dataset=Dataset.CHBMIT,
            patient_id=m.group(1),
            session_id=session,
        )

    # TUEV — tuh_<event>_<patient_num>_<session>_<segment?>
    # Check this BEFORE the generic tuh_ pattern, since 'spsw' would
    # otherwise be parsed as a patient code.
    if bn.startswith('tuh_'):
        m = _TUEV_REGEX.match(bn)
        if m:
            seg = int(m.group(4)) if m.group(4) else None
            return ParsedFilename(
                dataset=Dataset.TUH_EVENTS,
                patient_id=f'tuev_{m.group(2)}',
                session_id=m.group(3),
                event_type=m.group(1),
                segment=seg,
            )
        # Generic TUH — tuh_<patient>_<session...>
        # Filename alone can't distinguish tuh_seizure vs tuh_artifact vs
        # tuh_epilepsy (edf_to_events.py flattened them all to 'tuh_*').
        # Default to TUH_SEIZURE; build_manifest.py overrides via v2 lookup.
        m = _TUH_PATIENT_REGEX.match(bn)
        if m:
            return ParsedFilename(
                dataset=Dataset.TUEG,  # TUEG super — builder overrides per annotations
                patient_id=m.group(1),
                session_id=m.group(2),
            )
        raise ValueError(f'unrecognised tuh filename: {bn!r}')

    # TUEP — tuep_<patient>_<session...>. Same naming pattern as TUH-seizure
    # but with a distinct prefix so the manifest builder doesn't conflate
    # them. Emitted by edf_to_events.py --dataset tuep.
    if bn.startswith('tuep_'):
        m = _TUEP_PATIENT_REGEX.match(bn)
        if m:
            return ParsedFilename(
                dataset=Dataset.TUH_EPILEPSY,
                patient_id=m.group(1),
                session_id=m.group(2),
            )
        raise ValueError(f'unrecognised tuep filename: {bn!r}')

    # Single-prefix datasets — siena/eegmmidb/mental_arithmetic
    # eegmmi_ is an alias for eegmmidb_ (produced by edf_to_events --dataset eegmmi)
    _SINGLE_PREFIX = [
        (Dataset.SIENA, 'siena_'),
        (Dataset.EEGMMIDB, 'eegmmidb_'),
        (Dataset.EEGMMIDB, 'eegmmi_'),
        (Dataset.MENTAL_ARITHMETIC, 'mental_arithmetic_'),
        (Dataset.MENTAL_ARITHMETIC, 'generic_'),
        (Dataset.SLEEP_EDF, 'sleep_'),
    ]
    for ds, prefix in _SINGLE_PREFIX:
        if bn.startswith(prefix):
            rest = bn[len(prefix):]
            tokens = rest.split('_')
            patient = tokens[0]
            session = '_'.join(tokens[1:]) if len(tokens) > 1 else ''
            # EEGMMIDB: S001R01 → patient=S001, session=R01
            # Each subject has ~14 recordings; splitting by recording
            # instead of subject causes data leakage.
            if ds == Dataset.EEGMMIDB and 'R' in patient:
                parts = patient.split('R', 1)
                patient = parts[0]
                session = f'R{parts[1]}'
            return ParsedFilename(
                dataset=ds,
                patient_id=patient,
                session_id=session,
            )

    raise ValueError(f'unrecognised filename (no known dataset prefix): {bn!r}')


# ============================================================
# FileEntry — one row per NPZ
# ============================================================

# Clinical category constants — used by FileEntry and ClinicalWeightedSampler.
CLINICAL_CATEGORIES = (
    'seizure', 'spike_event', 'epilepsy_patient', 'sleep',
    'pediatric', 'artifact', 'normal',
)


@dataclass
class FileEntry:
    """One file's metadata. Split is FILE-level — every window in this
    file belongs to the same split. No window-level granularity."""
    path: str                       # absolute or repo-relative NPZ path
    dataset: Dataset
    patient_id: str
    session_id: str
    split: Split
    n_windows: int = 0              # number of L3 windows precomputed in this file
    event_type: str = ''            # TUEV annotation, '' otherwise
    segment: Optional[int] = None   # TUEV segment number, None otherwise
    has_seizure: bool = False
    sample_rate: int = 250
    n_channels: int = 21
    clinical_category: str = 'normal'  # pre-computed during manifest build
    adc_bits: int = 16                  # ADC resolution (16=EDF, 24=BDF)
    noise_bits_median: int = 0          # median estimated noise_bits across windows
    noise_bits_p95: int = 0             # 95th percentile (conservative estimate)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['dataset'] = self.dataset.value
        d['split'] = self.split.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'FileEntry':
        kw = dict(d)
        kw['dataset'] = Dataset(d['dataset'])
        kw['split'] = Split(d['split'])
        return cls(**kw)


# ============================================================
# DatasetEntry — one per source corpus
# ============================================================

@dataclass
class DatasetEntry:
    """One dataset's full inventory inside the manifest."""
    name: str                                # matches Dataset(...).value
    n_files: int = 0
    n_windows: int = 0
    n_patients: int = 0
    n_train_files: int = 0
    n_val_files: int = 0
    n_train_windows: int = 0
    n_val_windows: int = 0
    holdout_patients: List[str] = field(default_factory=list)
    validation_only: bool = False            # siena/eegmmidb/mental → never in training
    files: List[FileEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'n_files': self.n_files,
            'n_windows': self.n_windows,
            'n_patients': self.n_patients,
            'n_train_files': self.n_train_files,
            'n_val_files': self.n_val_files,
            'n_train_windows': self.n_train_windows,
            'n_val_windows': self.n_val_windows,
            'holdout_patients': sorted(self.holdout_patients),
            'validation_only': self.validation_only,
            'files': [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DatasetEntry':
        kw = dict(d)
        kw['files'] = [FileEntry.from_dict(f) for f in d.get('files', [])]
        return cls(**kw)

    def recompute_aggregates(self) -> None:
        """Recompute n_files/n_patients/etc. from the files list."""
        self.n_files = len(self.files)
        self.n_windows = sum(f.n_windows for f in self.files)
        self.n_patients = len({f.patient_id for f in self.files})
        self.n_train_files = sum(1 for f in self.files if f.split == Split.TRAIN)
        self.n_val_files = sum(1 for f in self.files if f.split == Split.VAL)
        self.n_train_windows = sum(f.n_windows for f in self.files if f.split == Split.TRAIN)
        self.n_val_windows = sum(f.n_windows for f in self.files if f.split == Split.VAL)


# ============================================================
# DatasetManifest — single source of truth
# ============================================================

MANIFEST_VERSION = '3.0.0'


@dataclass
class DatasetManifest:
    """The canonical training/validation split.

    Built once by build_manifest.py, then read by every training script.
    Replaces the dual-source (official_split_config.json + validation_manifest.json)
    arrangement that produced four independent bug categories.
    """
    version: str = MANIFEST_VERSION
    created: str = ''
    seed: int = 42
    val_fraction: float = 0.05
    datasets: Dict[str, DatasetEntry] = field(default_factory=dict)

    # Aggregates (recomputed by recompute_aggregates())
    total_files: int = 0
    total_windows: int = 0
    total_patients: int = 0
    train_files: int = 0
    val_files: int = 0
    train_windows: int = 0
    val_windows: int = 0

    # ------------------------------------------------------------
    # I/O — every load runs validate(), every save round-trips
    # ------------------------------------------------------------

    @classmethod
    def load(cls, path) -> 'DatasetManifest':
        """Load and validate. Raises ValueError on any contract violation."""
        with open(path) as f:
            d = json.load(f)
        m = cls(
            version=d.get('version', MANIFEST_VERSION),
            created=d.get('created', ''),
            seed=d.get('seed', 42),
            val_fraction=d.get('val_fraction', 0.05),
            total_files=d.get('total_files', 0),
            total_windows=d.get('total_windows', 0),
            total_patients=d.get('total_patients', 0),
            train_files=d.get('train_files', 0),
            val_files=d.get('val_files', 0),
            train_windows=d.get('train_windows', 0),
            val_windows=d.get('val_windows', 0),
        )
        m.datasets = {
            k: DatasetEntry.from_dict(v) for k, v in d.get('datasets', {}).items()
        }
        issues = m.validate()
        if issues:
            raise ValueError(
                'Manifest failed validation:\n  - ' + '\n  - '.join(issues)
            )
        return m

    def save(self, path) -> Path:
        """Write to disk. Recomputes aggregates first."""
        self.recompute_aggregates()
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat(timespec='seconds')
        out = self._serialisable_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        return path

    # ------------------------------------------------------------
    # Hash + provenance — for reproducibility (refactor #72)
    # ------------------------------------------------------------

    def _serialisable_dict(self) -> dict:
        """One canonical dict shape used by both save() and hash()."""
        return {
            'version': self.version,
            'created': self.created,
            'seed': self.seed,
            'val_fraction': self.val_fraction,
            'total_files': self.total_files,
            'total_windows': self.total_windows,
            'total_patients': self.total_patients,
            'train_files': self.train_files,
            'val_files': self.val_files,
            'train_windows': self.train_windows,
            'val_windows': self.val_windows,
            'datasets': {k: v.to_dict() for k, v in self.datasets.items()},
        }

    def hash(self) -> str:
        """Deterministic content hash of the manifest.

        Excludes the `created` timestamp (which would otherwise make
        every save change the hash). Two manifests built from the same
        seed/config/files produce the same hash. Used as the canonical
        provenance tag in checkpoint metadata — a checkpoint's
        manifest_hash matches one and only one DatasetManifest content.
        """
        import hashlib
        d = self._serialisable_dict()
        d.pop('created', None)
        # `sort_keys=True` makes dict iteration order irrelevant.
        canonical = json.dumps(d, sort_keys=True, separators=(',', ':'))
        return 'sha256:' + hashlib.sha256(canonical.encode()).hexdigest()

    def diff(self, other: 'DatasetManifest') -> Dict[str, Tuple]:
        """Shallow diff of (key → (self_value, other_value)) for top-level fields.

        For inspecting why two manifests don't share a hash. Only compares
        top-level scalars (version/seed/totals); per-file diffs are out
        of scope — use `set(self.all_files()) ^ set(other.all_files())`
        for that.
        """
        a = self._serialisable_dict()
        b = other._serialisable_dict()
        for skip in ('created', 'datasets'):
            a.pop(skip, None)
            b.pop(skip, None)
        out = {}
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                out[k] = (a.get(k), b.get(k))
        return out

    # ------------------------------------------------------------
    # Query — the public API every training script uses
    # ------------------------------------------------------------

    def get_files(self, split: Split,
                  datasets: Optional[List[Dataset]] = None) -> List[Path]:
        """Files for a given split. The ONE function for split-aware loading.

        Replaces the old get_training_files() / get_validation_files() pair.
        """
        return [Path(e.path) for e in self.get_file_entries(split, datasets)]

    def get_file_entries(self, split: Split,
                         datasets: Optional[List[Dataset]] = None) -> List[FileEntry]:
        """Same as get_files() but returns the full FileEntry (with provenance)."""
        out: List[FileEntry] = []
        for ds_name, entry in self.datasets.items():
            if datasets is not None and Dataset(ds_name) not in datasets:
                continue
            for f in entry.files:
                if f.split == split:
                    out.append(f)
        return out

    def all_files(self) -> Iterator[FileEntry]:
        """Iterate every FileEntry across every dataset."""
        for entry in self.datasets.values():
            yield from entry.files

    def lookup_path(self, path: str) -> Optional[FileEntry]:
        """Find a FileEntry by NPZ path. None if not in the manifest."""
        target = str(Path(path).resolve())
        for f in self.all_files():
            if str(Path(f.path).resolve()) == target:
                return f
        return None

    # ------------------------------------------------------------
    # Validation — runs on every load
    # ------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of contract violations. Empty list = valid manifest."""
        issues: List[str] = []

        # 1. Subject-disjoint check (per dataset)
        for name, entry in self.datasets.items():
            train_pids = {f.patient_id for f in entry.files if f.split == Split.TRAIN}
            val_pids = {f.patient_id for f in entry.files if f.split == Split.VAL}
            overlap = train_pids & val_pids
            if overlap:
                sample = sorted(overlap)[:5]
                issues.append(
                    f'DATA LEAKAGE in {name}: {len(overlap)} patient(s) appear in '
                    f'both train and val (sample: {sample})'
                )

        # 2. validation_only datasets must not appear in TRAIN
        for name, entry in self.datasets.items():
            if entry.validation_only:
                bad = [f for f in entry.files if f.split == Split.TRAIN]
                if bad:
                    issues.append(
                        f'validation_only dataset {name} has {len(bad)} train files '
                        f'(must be all val/holdout)'
                    )

        # 3. Per-dataset window totals
        for name, entry in self.datasets.items():
            actual = sum(f.n_windows for f in entry.files)
            if entry.n_windows and actual != entry.n_windows:
                issues.append(
                    f'{name}: declared n_windows={entry.n_windows}, '
                    f'sum across files={actual}'
                )

        # 4. Global aggregates
        train_w = sum(
            f.n_windows for e in self.datasets.values()
            for f in e.files if f.split == Split.TRAIN
        )
        val_w = sum(
            f.n_windows for e in self.datasets.values()
            for f in e.files if f.split == Split.VAL
        )
        if self.train_windows and train_w != self.train_windows:
            issues.append(
                f'train_windows mismatch: declared {self.train_windows}, '
                f'sum {train_w}'
            )
        if self.val_windows and val_w != self.val_windows:
            issues.append(
                f'val_windows mismatch: declared {self.val_windows}, '
                f'sum {val_w}'
            )

        # 5. Holdout patients must actually exist in the dataset
        for name, entry in self.datasets.items():
            actual_patients = {f.patient_id for f in entry.files}
            phantoms = set(entry.holdout_patients) - actual_patients
            if phantoms:
                issues.append(
                    f'{name}: holdout_patients lists {len(phantoms)} unknown '
                    f'patients (sample: {sorted(phantoms)[:5]})'
                )

        return issues

    def recompute_aggregates(self) -> None:
        """Recompute global aggregates from the per-dataset entries."""
        for entry in self.datasets.values():
            entry.recompute_aggregates()
        self.total_files = sum(e.n_files for e in self.datasets.values())
        self.total_windows = sum(e.n_windows for e in self.datasets.values())
        self.total_patients = sum(e.n_patients for e in self.datasets.values())
        self.train_files = sum(e.n_train_files for e in self.datasets.values())
        self.val_files = sum(e.n_val_files for e in self.datasets.values())
        self.train_windows = sum(e.n_train_windows for e in self.datasets.values())
        self.val_windows = sum(e.n_val_windows for e in self.datasets.values())

    def summary_str(self) -> str:
        """Human-readable summary suitable for logs and CLI."""
        self.recompute_aggregates()
        lines = [
            f'DatasetManifest v{self.version} (seed={self.seed}, '
            f'val_fraction={self.val_fraction})',
            f'  Total: {self.total_files:,} files, {self.total_windows:,} windows, '
            f'{self.total_patients:,} patients',
            f'  Train: {self.train_files:,} files, {self.train_windows:,} windows '
            f'({100 * self.train_windows / max(self.total_windows, 1):.1f}%)',
            f'  Val:   {self.val_files:,} files, {self.val_windows:,} windows '
            f'({100 * self.val_windows / max(self.total_windows, 1):.1f}%)',
            '',
        ]
        for name, e in self.datasets.items():
            tag = ' [validation_only]' if e.validation_only else ''
            lines.append(
                f'  {name:22}{tag}  files={e.n_files:>5}  patients={e.n_patients:>4}  '
                f'train={e.n_train_files:>5}/{e.n_train_windows:>9,}w  '
                f'val={e.n_val_files:>5}/{e.n_val_windows:>9,}w  '
                f'holdout_patients={len(e.holdout_patients)}'
            )
        return '\n'.join(lines)


# ============================================================
# EEGWindow — full provenance for one 10s segment of raw EEG
# ============================================================

@dataclass
class EEGWindow:
    """A single 10-second EEG window with full provenance."""
    dataset: Dataset
    patient_id: str
    session_id: str
    window_index: int               # which window within the source file
    split: Split

    signal: 'np.ndarray'            # [C, 2500] int16 or float64
    sample_rate: int = 250
    n_channels: int = 21
    channel_labels: List[str] = field(default_factory=list)

    has_seizure: bool = False
    event_type: str = ''

    source_npz: str = ''
    sha256: str = ''                # 16-char prefix of signal hash

    def compute_sha(self) -> str:
        """Hash the signal bytes — used as parent reference in L3Window."""
        import numpy as np
        b = np.ascontiguousarray(self.signal).tobytes()
        return hashlib.sha256(b).hexdigest()[:16]


# ============================================================
# L3Window — precomputed L3 subband with parent reference
# ============================================================

@dataclass
class L3Window:
    """A single L3 subband window with provenance and parent hash."""
    dataset: Dataset
    patient_id: str
    session_id: str
    window_index: int
    split: Split

    l3_approx: 'np.ndarray'         # [C, 313]

    parent_sha256: str = ''         # traces back to EEGWindow
    source_npz: str = ''

    has_seizure: bool = False
    event_type: str = ''


# ============================================================
# TrainingBatch — what the dataloader produces
# ============================================================

@dataclass
class TrainingBatch:
    """A batch of L3 windows with per-sample provenance.

    `assert_no_leakage(expected_split)` is the runtime safety net —
    if a validation window ever leaks into a training batch (or vice
    versa), the assertion fires immediately, not 22 hours later.

    `fullband_target` is the raw EEG window aligned to the L3 sample,
    when the dataset was built with `with_fullband=True`. Tier 3+
    decoders (`output: 'istft'`) emit fullband [B, 21, 2500] directly,
    so the loss can compare decoder output to fullband target without
    any inverse-pipeline-in-gradient-path. Tier 1-2 (`output:
    'direct'`) emit L3-scale [B, 21, 313] and ignore this field.
    """
    l3_approx: 'torch.Tensor'                  # [B, C, 313]
    fullband_target: 'torch.Tensor' = None     # [B, C, 2500] or None

    # Per-sample provenance, parallel arrays of length batch_size.
    datasets: List[str] = field(default_factory=list)
    patient_ids: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=list)
    has_seizure: List[bool] = field(default_factory=list)
    event_types: List[str] = field(default_factory=list)
    clinical_categories: List[str] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.l3_approx.shape[0]

    def assert_no_leakage(self, expected: Split) -> None:
        """Every sample in this batch must be from the expected split."""
        for i, s in enumerate(self.splits):
            if s != expected.value:
                raise AssertionError(
                    f'Data leakage at sample {i}: expected split={expected.value}, '
                    f'got {s} '
                    f'(patient={self.patient_ids[i] if i < len(self.patient_ids) else "?"}, '
                    f'dataset={self.datasets[i] if i < len(self.datasets) else "?"})'
                )

    def assert_subject_disjoint_from(self, other: 'TrainingBatch') -> None:
        """No (dataset, patient_id) pair may appear in both batches."""
        a = set(zip(self.datasets, self.patient_ids))
        b = set(zip(other.datasets, other.patient_ids))
        overlap = a & b
        if overlap:
            raise AssertionError(
                f'Subject overlap: {len(overlap)} (dataset, patient) pairs '
                f'appear in both batches: {sorted(overlap)[:5]}'
            )


__all__ = [
    'Split', 'Dataset', 'VALIDATION_ONLY_DATASETS', 'CLINICAL_CATEGORIES',
    'parse_npz_filename', 'ParsedFilename',
    'FileEntry', 'DatasetEntry', 'DatasetManifest', 'MANIFEST_VERSION',
    'EEGWindow', 'L3Window', 'TrainingBatch',
]
