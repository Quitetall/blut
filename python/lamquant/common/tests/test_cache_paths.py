"""Pin the un-misconfigurable cache-location contract (owner directive 2026-06-10).

These assert the INVARIANTS, not implementation: one root in → three
mutually-consistent, on-root, created dirs out; apply_env() forces them even
over hostile pre-set values.
"""
import os
import pytest

from lamquant.common import cache_paths as cp


def test_one_root_drives_all_three(tmp_path, monkeypatch):
    monkeypatch.setenv(cp.DATA_ROOT_ENV, str(tmp_path))
    lay = cp.resolve(create=True)
    assert lay.data_root == os.path.abspath(str(tmp_path))
    # every dir is UNDER the single root — none can point elsewhere
    for d in (lay.l3_cache_dir, lay.fb_cache_dir, lay.memmap_dir):
        assert d.startswith(lay.data_root + os.sep)
        assert os.path.isdir(d)  # created on resolve → "it's there"
    # fixed, distinct subpaths
    assert lay.l3_cache_dir.endswith(cp.L3_SUBPATH)
    assert lay.fb_cache_dir.endswith(cp.FB_SUBPATH)
    assert lay.memmap_dir.endswith(cp.MEMMAP_SUBPATH)
    assert len({lay.l3_cache_dir, lay.fb_cache_dir, lay.memmap_dir}) == 3


def test_default_root_when_unset(monkeypatch):
    monkeypatch.delenv(cp.DATA_ROOT_ENV, raising=False)
    assert cp.data_root() == os.path.abspath(cp.DEFAULT_DATA_ROOT)


def test_apply_env_forces_consistency_over_hostile_values(tmp_path, monkeypatch):
    # an operator set the three to inconsistent off-root junk — apply_env must
    # OVERWRITE all three from the single root (no scenario where they survive).
    monkeypatch.setenv(cp.DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(cp.L3_ENV, "/tmp/wrong_l3")
    monkeypatch.setenv(cp.FB_ENV, "")
    monkeypatch.setenv(cp.MEMMAP_ENV, "/somewhere/else")
    lay = cp.apply_env(create=True)
    assert os.environ[cp.L3_ENV] == lay.l3_cache_dir
    assert os.environ[cp.FB_ENV] == lay.fb_cache_dir
    assert os.environ[cp.MEMMAP_ENV] == lay.memmap_dir
    # all three now share the one root
    root = os.path.abspath(str(tmp_path))
    for v in (os.environ[cp.L3_ENV], os.environ[cp.FB_ENV], os.environ[cp.MEMMAP_ENV]):
        assert v.startswith(root + os.sep)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
