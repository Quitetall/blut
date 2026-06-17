"""Data ingredient specs (ADR 0050/0051). Importing this registers them.

Three ``kind="data"`` specs, each ``cache_relevant=True`` (the dataset a stage
trains on is part of the trained artifact's identity — two materially different
corpora must never collide on one stage cache key):

  * ``lma_snn``       — the seizure-aware ``snn.lma_dataset.LmaDataset``, shared
                        by ``snn/train_4state_controller.py`` (train + val) and
                        ``snn/pretrain_ssl_tueg.py`` (single split). The
                        byte-identical ``expand_lma_roots`` helper that BOTH
                        trainers copy-pasted is moved here as the single source
                        of truth, alongside the identical num_workers/DataLoader
                        preamble (``build_lma_dataloader_kwargs``).
  * ``lma_l3``        — ``lamquant_codec.training.LmaL3Dataset``, shared by
                        ``student/pretrain_mae.py`` + ``oracle/train_l3_teacher.py``.
  * ``lma_typed_l3``  — ``student/lma_typed_adapter.py``'s ``LmaTypedL3Dataset``
                        (the typed-batch joint-codec adapter), built exactly as
                        ``student/train_joint.py`` builds its train + val pair —
                        INCLUDING the mandatory decode-cache priming
                        (``cache_paths.apply_env`` + ``LMA_NUM_WORKERS`` default)
                        that MUST run before the datasets are constructed so the
                        fork-workers inherit the cache dirs (the documented
                        epoch-2-OOM footgun).

The DATASET CONSTRUCTION is the ingredient output. The per-epoch iteration
(``prefetch_typed_batches`` / ``prefetch_batches`` / the final
sampler-vs-shuffle ``DataLoader``) STAYS in each trainer — the sampler differs
per trainer, so each builds its own final loader from the dataset(s) + the
exported ``build_lma_dataloader_kwargs`` helper.

Heavy deps (the codec/neural wheels, torch) are imported lazily inside ``build``
so importing this module to register the specs stays cheap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec

# The seizure-aware SNN dataset's per-file window cap default. Imported lazily
# in build() (it lives in the snn area, which self-inserts its sys.path); the
# cfg default below mirrors its value so a recipe need not pass it. Kept as a
# module constant so the dataclass default is a plain int (hashable / cacheable).
_MAX_WINDOWS_PER_FILE_DEFAULT = 5


# ===========================================================================
# Shared helpers (the REAL dedup) — were byte-identical copies in
# snn/train_4state_controller.py and snn/pretrain_ssl_tueg.py.
# ===========================================================================

def expand_lma_roots(roots) -> list[Path]:
    """Expand ``--lma-root`` args into a sorted, de-duplicated list of ``.lma``.

    Byte-identical to the copies in ``snn/train_4state_controller.py`` and
    ``snn/pretrain_ssl_tueg.py`` (moved here as the single source of truth).
    Each root may be a single ``.lma`` file or a directory globbed one then two
    levels deep (``<src>/<corpus>.lma`` then ``<corpus>.lma``). Order is glob-
    sorted within each root; the final de-dup is FIRST-SEEN-WINS (stable), not a
    re-sort across roots, so two roots that share an archive keep the first
    root's ordering.
    """
    lma_paths: list[Path] = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix == ".lma":
            lma_paths.append(r)
            continue
        if not r.is_dir():
            raise FileNotFoundError(f"--lma-root not found: {r}")
        found = sorted(r.glob("*/*.lma")) or sorted(r.glob("*.lma"))
        if not found:
            raise RuntimeError(f"no .lma archives under {r}")
        lma_paths.extend(found)
    seen: set = set()
    return [p for p in lma_paths if not (str(p) in seen or seen.add(str(p)))]


def build_lma_dataloader_kwargs(num_workers_arg, device):
    """The identical DataLoader preamble both SNN trainers compute inline.

    Returns ``(num_workers, dl_kwargs, pin)``:

      * ``num_workers`` — the explicit ``--num-workers`` arg when not None, else
        ``LMA_NUM_WORKERS`` from the env, else an L3_CACHE_DIR-CONDITIONED
        default (4 when an L3 decode cache is configured, 2 otherwise). The
        cache-conditioned default is LOAD-BEARING and preserved verbatim — a
        warm on-disk L3 cache makes more decode workers a win; a cold one makes
        them a RAM liability.
      * ``dl_kwargs`` — ``persistent_workers`` + ``prefetch_factor`` (from
        ``LMA_PREFETCH_FACTOR``, default 4) ONLY when ``num_workers > 0``.
      * ``pin`` — ``device.type == "cuda" and num_workers > 0``.

    The trainer builds its own final ``DataLoader`` (sampler vs shuffle differs)
    from these; this helper owns only the shared resolution.
    """
    _default_workers = 4 if os.environ.get("L3_CACHE_DIR") else 2
    num_workers = num_workers_arg if num_workers_arg is not None else \
        int(os.environ.get("LMA_NUM_WORKERS", str(_default_workers)))
    _dl_kwargs = {}
    if num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = int(
            os.environ.get("LMA_PREFETCH_FACTOR", "4"))
    pin = device.type == "cuda" and num_workers > 0
    return num_workers, _dl_kwargs, pin


# ===========================================================================
# (1) lma_snn — the seizure-aware LmaDataset (SNN trainers).
# ===========================================================================

@dataclass(frozen=True)
class LmaSnnConfig:
    """Mirrors the ``LmaDataset`` construction kwargs both SNN trainers pin.

    ``lma_root`` accepts the same shapes ``--lma-root`` does (a single path or a
    list, each a ``.lma`` file or a directory). ``split`` selects which split to
    materialise; when ``split == "train"`` a paired ``val`` dataset is also
    built (the controller wants both), otherwise only the requested split (the
    SSL pretrain wants a single split). ``batch_size`` is carried for cache
    identity only (the final DataLoader is the trainer's).
    """
    lma_root: Union[str, Path, list] = ""
    split_manifest: Union[str, Path] = ""
    split: str = "train"
    max_windows_per_file: int = _MAX_WINDOWS_PER_FILE_DEFAULT
    seq_windows: int = 1
    batch_size: int = 128


def _as_root_list(lma_root):
    """Normalise ``lma_root`` (a single path or a list) into a list for
    ``expand_lma_roots`` — both SNN trainers pass an ``nargs='+'`` list, but a
    recipe may pin a single path."""
    if isinstance(lma_root, (str, Path)):
        return [lma_root]
    return list(lma_root)


def _build_lma_snn(cfg, *, device=None):  # noqa: ARG001 — device unused (kept for kind-uniform extra)
    """Construct the seizure-aware LmaDataset(s).

    Returns ``{"train_ds": ..., "val_ds": ...}``; ``val_ds`` is None unless
    ``cfg.split == "train"`` (the controller builds train+val from one call; the
    SSL pretrain builds one split). The construction is byte-identical to the
    trainers' inline ``LmaDataset(...)`` calls with the SAME kwargs; the final
    sampler-vs-shuffle DataLoader stays in each trainer (use the exported
    ``build_lma_dataloader_kwargs`` to resolve its worker plumbing).
    """
    from lamquant.snn.lma_dataset import LmaDataset

    lma_paths = expand_lma_roots(_as_root_list(cfg.lma_root))
    train_ds = LmaDataset(
        lma_paths=lma_paths, split=cfg.split,
        split_manifest_path=cfg.split_manifest,
        max_windows_per_file=cfg.max_windows_per_file,
        seq_windows=cfg.seq_windows)
    val_ds = None
    if cfg.split == "train":
        val_ds = LmaDataset(
            lma_paths=lma_paths, split="val",
            split_manifest_path=cfg.split_manifest,
            max_windows_per_file=cfg.max_windows_per_file,
            seq_windows=cfg.seq_windows)
    return {"train_ds": train_ds, "val_ds": val_ds}


@register_ingredient
def _lma_snn_data():
    return IngredientSpec(
        name="lma_snn", kind="data", config_cls=LmaSnnConfig,
        cache_relevant=True,
        build=_build_lma_snn,
    )


# ===========================================================================
# (2) lma_l3 — the codec LmaL3Dataset (MAE pretrain + L3 teacher).
# ===========================================================================

@dataclass(frozen=True)
class LmaL3Config:
    """Mirrors the ``LmaL3Dataset`` construction in ``pretrain_mae`` + the L3
    teacher. Both resolve the TRAIN split stems then build one dataset.

    ``seed`` defaults to 0 — the teacher OMITS ``seed=`` (so ``LmaL3Dataset``'s
    own default 0 applies), and the MAE pretrain passes its run seed in. The
    cfg default 0 therefore keeps the teacher BYTE-IDENTICAL while a recipe (or
    the MAE wiring) can override it.
    """
    lma_root: Union[str, Path] = ""
    split_manifest: Union[str, Path] = ""
    windows_per_epoch: int = 50000
    max_windows: Optional[int] = None
    seed: int = 0


def _build_lma_l3(cfg):
    """Build the train-split ``LmaL3Dataset`` exactly as both trainers do:
    resolve the train stems from the manifest, then construct.

    Byte-identical to::

        stems, _ = load_split_stems(split_manifest, "train")
        LmaL3Dataset(lma_root=..., file_stems=stems,
                     windows_per_epoch=..., max_windows=..., seed=...)
    """
    from lamquant_codec.training import LmaL3Dataset, load_split_stems

    train_stems, _ = load_split_stems(cfg.split_manifest, "train")
    return LmaL3Dataset(
        lma_root=cfg.lma_root, file_stems=train_stems,
        windows_per_epoch=cfg.windows_per_epoch,
        max_windows=cfg.max_windows,
        seed=cfg.seed,
    )


@register_ingredient
def _lma_l3_data():
    return IngredientSpec(
        name="lma_l3", kind="data", config_cls=LmaL3Config,
        cache_relevant=True,
        build=_build_lma_l3,
    )


# ===========================================================================
# (3) lma_typed_l3 — train_joint's typed-batch adapter (HIGH RISK).
# ===========================================================================

@dataclass(frozen=True)
class LmaTypedL3Config:
    """Mirrors ``student/train_joint.py``'s LMA-direct train+val construction.

    ``return_fullband`` is a RESOLVED bool: ``train_joint`` resolves its
    ``'auto'`` fullband mode to ram/memmap/off UPSTREAM and passes the boolean
    ``(use_fullband_mode != 'off')`` — this spec takes that already-resolved
    bool, NOT the mode string. ``max_windows_per_file`` is optional: when None
    the trainer omits the kwarg entirely (the adapter's own default applies);
    when set it is forwarded. The val dataset uses ``seed + 1`` (verbatim).
    """
    lma_root: Union[str, Path] = ""
    split_manifest: Union[str, Path] = ""
    windows_per_epoch: int = 50000
    val_windows: int = 50000
    return_fullband: bool = False
    seed: int = 0
    max_windows_per_file: Optional[int] = None


def _build_lma_typed_l3(cfg):
    """Build the typed-adapter ``(train_ds, val_ds)`` pair, byte-identically to
    ``student/train_joint.py`` lines 633-698 of the LMA-direct branch.

    CRITICAL ORDER (the documented epoch-2-OOM footgun): the MANDATORY decode-
    cache priming MUST run BEFORE the datasets are constructed. The DataLoader
    fork-workers inherit ``L3_CACHE_DIR`` / ``FB_CACHE_DIR`` / ``MEMMAP_DIR``
    (set by ``cache_paths.apply_env``) and ``LMA_NUM_WORKERS`` from the
    environment AT FORK TIME — and the adapter itself reads ``LMA_NUM_WORKERS``
    in its ``__init__``. Constructing first would let the workers come up
    without the caches → unbounded RAM growth + OOM at epoch 2. This is why the
    priming lives INSIDE build(), not in the trainer's pre-amble.

    ``os.environ.setdefault('LMA_NUM_WORKERS', '2')`` (NOT a plain assignment)
    preserves an explicit ``LMA_NUM_WORKERS=0`` (serial decode, for debugging).
    """
    # MANDATORY + STANDARDIZED decode caches — set BEFORE the datasets (and the
    # DataLoader fork that follows in the trainer) so workers inherit the dirs.
    from lamquant.common.cache_paths import apply_env
    apply_env()
    os.environ.setdefault("LMA_NUM_WORKERS", "2")

    # Package-form import (NOT the trainer's bare ``from lma_typed_adapter``):
    # train_joint puts ``lamquant/student`` on sys.path before its bare import,
    # but a recipe/framework caller of this ingredient need not have — and the
    # bare form then ModuleNotFoundErrors. The fully-qualified form always
    # resolves and the adapter MODULE self-inserts the sibling area dirs it needs
    # (``snn`` etc.) on import. Same class, same behaviour.
    from lamquant.student.lma_typed_adapter import LmaTypedL3Dataset

    _mwpf = ({} if cfg.max_windows_per_file is None
             else {"max_windows_per_file": cfg.max_windows_per_file})
    train_ds = LmaTypedL3Dataset(
        lma_root=cfg.lma_root,
        split="train",
        split_manifest_path=cfg.split_manifest,
        windows_per_epoch=cfg.windows_per_epoch,
        return_fullband=cfg.return_fullband,
        seed=cfg.seed,
        **_mwpf,
    )
    val_ds = LmaTypedL3Dataset(
        lma_root=cfg.lma_root,
        split="val",
        split_manifest_path=cfg.split_manifest,
        windows_per_epoch=cfg.val_windows,
        return_fullband=cfg.return_fullband,
        seed=cfg.seed + 1,
        **_mwpf,
    )
    return train_ds, val_ds


@register_ingredient
def _lma_typed_l3_data():
    return IngredientSpec(
        name="lma_typed_l3", kind="data", config_cls=LmaTypedL3Config,
        cache_relevant=True,
        build=_build_lma_typed_l3,
    )
