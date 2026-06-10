"""Canonical, un-misconfigurable cache + memmap locations for ALL LamQuant training.

Design goal (owner directive 2026-06-10): there must NOT be a single scenario
where the decode-cache / memmap locations can be messed up. So:

  * ONE knob names the location: ``LAMQUANT_DATA_ROOT`` (env), else the canonical
    default below. That is the only thing a setup ever sets.
  * Every cache / memmap dir is a FIXED subpath under that one root — derived,
    never independently configured. The L3 cache, the fullband-window cache, and
    the memmap dir can therefore NEVER point at three different roots or
    disagree.
  * ``apply_env()`` FORCE-OVERWRITES the per-layer env vars the dataset code
    reads (``L3_CACHE_DIR`` / ``FB_CACHE_DIR`` / ``MEMMAP_DIR``) from the single
    root, so a stale or hand-set value cannot survive. There is no "off" — a
    training entrypoint that calls ``apply_env()`` always gets valid, existing,
    mutually-consistent dirs.
  * The dirs are created on resolve, so "it's there" the moment you name the
    root.

Subpaths intentionally match the EXISTING on-disk layout (``Training/l3_cache``
holds the ~835 GB L3 cache today) so standardising does not orphan the cache
already built.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# The ONE place an absolute location is named. Override per-machine with the
# LAMQUANT_DATA_ROOT env var (the single setup knob); everything else derives.
DEFAULT_DATA_ROOT = "/mnt/4tb/data"
DATA_ROOT_ENV = "LAMQUANT_DATA_ROOT"

# Fixed subpaths under <root>. NOT independently configurable — that is the
# whole point: one root in, three consistent dirs out.
L3_SUBPATH = "Training/l3_cache"      # per-stem L3 stack [n,21,313]
FB_SUBPATH = "Training/fb_cache"      # per-WINDOW fullband [21,2500] fp16
MEMMAP_SUBPATH = "Training/memmap"    # flat fullband .dat memmaps (precompute path)

# Env vars the dataset layers actually read. apply_env() forces these.
L3_ENV = "L3_CACHE_DIR"
FB_ENV = "FB_CACHE_DIR"
MEMMAP_ENV = "MEMMAP_DIR"


@dataclass(frozen=True)
class CacheLayout:
    """The resolved, mutually-consistent location set (all under one root)."""
    data_root: str
    l3_cache_dir: str
    fb_cache_dir: str
    memmap_dir: str


def data_root() -> str:
    """The single canonical data root: LAMQUANT_DATA_ROOT, else the default.

    Fail-CLOSED: if the env var is unset AND the hardcoded default does not
    exist on this machine, raise rather than silently creating cache dirs at a
    machine-specific absolute path that may be wrong (the whole point is that
    caches are never created in the wrong place)."""
    explicit = os.environ.get(DATA_ROOT_ENV, "").strip()
    if explicit:
        return os.path.abspath(explicit)
    root = os.path.abspath(DEFAULT_DATA_ROOT)
    if not os.path.isdir(root):
        raise RuntimeError(
            f"cache data root {root!r} does not exist and {DATA_ROOT_ENV} is "
            f"unset. Set {DATA_ROOT_ENV}=<dir holding Archive/ + Training/> so "
            f"the decode caches are never created in the wrong place."
        )
    return root


def resolve(create: bool = True) -> CacheLayout:
    """Resolve the canonical layout from the one root. Creates the dirs by
    default so they always exist when named."""
    root = data_root()
    lay = CacheLayout(
        data_root=root,
        l3_cache_dir=os.path.join(root, L3_SUBPATH),
        fb_cache_dir=os.path.join(root, FB_SUBPATH),
        memmap_dir=os.path.join(root, MEMMAP_SUBPATH),
    )
    if create:
        for d in (lay.l3_cache_dir, lay.fb_cache_dir, lay.memmap_dir):
            os.makedirs(d, exist_ok=True)
    return lay


def apply_env(create: bool = True) -> CacheLayout:
    """Resolve + FORCE the canonical dirs into the env vars the dataset layers
    read (``L3_CACHE_DIR`` / ``FB_CACHE_DIR`` / ``MEMMAP_DIR``), OVERWRITING any
    pre-set value so the three can never disagree or point off the standard
    root. Call once at the top of a training entrypoint, BEFORE the DataLoader
    forks workers (so they inherit the dirs). Returns the layout for logging."""
    lay = resolve(create=create)
    os.environ[L3_ENV] = lay.l3_cache_dir
    os.environ[FB_ENV] = lay.fb_cache_dir
    os.environ[MEMMAP_ENV] = lay.memmap_dir
    return lay
