#!/usr/bin/env python3
"""Tests for the BLUT 1.0 structural release validator.

Moved here from Quitetall/LamQuant with the validator on 2026-09-07. The
synthetic tree still uses `training/engine/` manifest paths ON PURPOSE: the
prefix is now a CATALOG field, so keeping the fixture on the OLD spelling while
the live catalog uses the new one is what proves the field is read rather than
assumed. A fixture that matched the live shape would pass whether the code read
the field or hard-coded the empty string.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_blut_release_state import validate_release_state


CATALOG = """
schema_version = 1
release_version = "1.0.0"
release_tag = "v1.0.0"
engine_prefix = "training/engine/"
release_workflow = "training/engine/.github/workflows/release.yml"
publish_chain = ["blut-types", "blut"]
binary_artifacts = ["blut-web"]

[toolchain]
cargo_sbom = "0.10.0"
wasm_bindgen_cli = "0.2.126"

[[rust_packages]]
name = "blut-types"
manifest = "training/engine/types/Cargo.toml"
distribution = "crates-io"

[[rust_packages]]
name = "blut"
manifest = "training/engine/Cargo.toml"
distribution = "crates-io"

[[rust_packages]]
name = "blut-web"
manifest = "training/engine/web/Cargo.toml"
distribution = "binary"

[[rust_packages]]
name = "blut-web-ui"
manifest = "training/engine/web/ui/Cargo.toml"
distribution = "embedded"

[[python_packages]]
name = "blut-sdk"
manifest = "training/engine/sdk/pyproject.toml"

[external.blut_backends]
env = "BLUT_BACKENDS_REPO"
name = "blut-cookbook-standard"
manifest = "Cargo.toml"
"""


WORKFLOW = """
crate: [blut-web]
environment: crates-io
CARGO_REGISTRY_TOKEN
scripts/k8s_kind_smoke.sh
cargo install cargo-sbom --version 0.10.0 --locked
cargo install wasm-bindgen-cli --version 0.2.126 --locked
(cd crates/blut-web && bash scripts/build_ui.sh)
if [ ! -s crates/blut-web/ui/dist/blut_web_ui_bg.wasm ]; then
  echo "missing dashboard"
  exit 1
fi
if grep -aFq "blut-web (API-only build)" "$BIN"; then
  echo "fallback dashboard"
  exit 1
fi
cargo sbom --project-directory "$path" --output-format spdx_json_2_3
cargo sbom --project-directory "$path" --output-format cyclone_dx_json_1_6
blut-types|types
blut|.
blut-web|web
blut-web-ui|web/ui
(cd dist && sha256sum -- * > SHA256SUMS)
(cd dist && sha256sum -c SHA256SUMS)
if [ "$QUIET_BENCHMARK_SHA" != "$GITHUB_SHA" ]; then
          publish blut-types
          publish blut
"""


class ReleaseFixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.backends = Path(self.tmp.name) / "backends"
        self.catalog = self.repo / "release.toml"
        self.write("release.toml", CATALOG)
        self.write(
            "training/engine/types/Cargo.toml",
            '[package]\nname="blut-types"\nversion="1.0.0"\npublish=["crates-io"]\n',
        )
        self.write(
            "training/engine/Cargo.toml",
            '[package]\nname="blut"\nversion="1.0.0"\npublish=["crates-io"]\n'
            '[dependencies]\nblut-types={path="types",version="=1.0.0"}\n',
        )
        self.write(
            "training/engine/web/Cargo.toml",
            '[package]\nname="blut-web"\nversion="1.0.0"\npublish=false\n'
            '[dependencies]\nblut={path="..",version="=1.0.0"}\n',
        )
        self.write(
            "training/engine/web/ui/Cargo.toml",
            '[package]\nname="blut-web-ui"\nversion="1.0.0"\npublish=false\n'
            '[dependencies]\nblut={path="../..",version="=1.0.0"}\n'
            'wasm-bindgen={version="=0.2.126"}\n',
        )
        self.write(
            "training/engine/sdk/pyproject.toml",
            '[project]\nname="blut-sdk"\nversion="1.0.0"\n',
        )
        self.write("training/engine/.github/workflows/release.yml", WORKFLOW)
        self.backends.mkdir(parents=True)
        (self.backends / "Cargo.toml").write_text(
            '[package]\nname="blut-cookbook-standard"\nversion="1.0.0"\n',
            encoding="utf-8",
        )

    def write(self, relative: str, text: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def errors(self) -> list[str]:
        return validate_release_state(
            self.repo, self.catalog, {"blut_backends": self.backends}
        )

    def close(self) -> None:
        self.tmp.cleanup()


class ReleaseStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ReleaseFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_valid_release_graph_passes(self) -> None:
        self.assertEqual(self.fixture.errors(), [])

    def test_version_mismatch_blocks(self) -> None:
        self.fixture.write(
            "training/engine/types/Cargo.toml",
            '[package]\nname="blut-types"\nversion="0.2.0-alpha.1"\npublish=["crates-io"]\n',
        )
        self.assertTrue(any("version must equal 1.0.0" in e for e in self.fixture.errors()))

    def test_unpinned_internal_path_dependency_blocks(self) -> None:
        self.fixture.write(
            "training/engine/Cargo.toml",
            '[package]\nname="blut"\nversion="1.0.0"\npublish=["crates-io"]\n'
            '[dependencies]\nblut-types={path="types"}\n',
        )
        self.assertTrue(any("must pin version" in e for e in self.fixture.errors()))

    def test_publish_order_blocks(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace("publish blut-types\n          publish blut", "publish blut\n          publish blut-types"),
        )
        self.assertTrue(any("catalog order" in e for e in self.fixture.errors()))

    def test_missing_binary_blocks(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace("crate: [blut-web]", "crate: [blut-tui]"),
        )
        self.assertTrue(any("binary matrix must equal" in e for e in self.fixture.errors()))

    def test_extra_binary_blocks(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace("crate: [blut-web]", "crate: [blut-web, extra]"),
        )
        self.assertTrue(any("binary matrix must equal" in e for e in self.fixture.errors()))

    def test_quiet_benchmark_must_compare_exact_sha(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace(
                'if [ "$QUIET_BENCHMARK_SHA" != "$GITHUB_SHA" ]; then',
                "quiet_benchmark_sha: documented-only",
            ),
        )
        self.assertTrue(any("QUIET_BENCHMARK_SHA" in e for e in self.fixture.errors()))

    def test_every_engine_workspace_needs_an_sbom(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace("blut-web|web\n", ""),
        )
        self.assertTrue(any("SBOM matrix misses blut-web|web" in e for e in self.fixture.errors()))

    def test_dashboard_fallback_guard_is_required(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace(
                'if grep -aFq "blut-web (API-only build)" "$BIN"; then\n',
                "",
            ),
        )
        self.assertTrue(any("API-only dashboard" in e for e in self.fixture.errors()))

    def test_dashboard_wasm_bundle_guard_is_required(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace(
                "if [ ! -s crates/blut-web/ui/dist/blut_web_ui_bg.wasm ]; then\n",
                "",
            ),
        )
        self.assertTrue(any("missing-or-empty dashboard" in e for e in self.fixture.errors()))

    def test_dashboard_wasm_guard_must_exit_nonzero(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace(
                '  echo "missing dashboard"\n  exit 1\nfi\n',
                '  echo "missing dashboard"\nfi\n',
                1,
            ),
        )
        self.assertTrue(any("missing-or-empty dashboard" in e for e in self.fixture.errors()))

    def test_dashboard_fallback_guard_must_exit_nonzero(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace(
                '  echo "fallback dashboard"\n  exit 1\nfi\n',
                '  echo "fallback dashboard"\nfi\n',
                1,
            ),
        )
        self.assertTrue(any("API-only dashboard" in e for e in self.fixture.errors()))

    def test_dashboard_wasm_bindgen_pin_must_match_cli(self) -> None:
        self.fixture.write(
            "training/engine/web/ui/Cargo.toml",
            '[package]\nname="blut-web-ui"\nversion="1.0.0"\npublish=false\n'
            '[dependencies]\nblut={path="../..",version="=1.0.0"}\n'
            'wasm-bindgen={version="=0.2.125"}\n',
        )
        self.assertTrue(any("match wasm-bindgen-cli" in e for e in self.fixture.errors()))

    def test_checksum_verification_must_run_inside_bundle(self) -> None:
        self.fixture.write(
            "training/engine/.github/workflows/release.yml",
            WORKFLOW.replace(
                "(cd dist && sha256sum -c SHA256SUMS)",
                "sha256sum -c dist/SHA256SUMS",
            ),
        )
        self.assertTrue(any("sha256sum -c SHA256SUMS" in e for e in self.fixture.errors()))

    def test_missing_backends_checkout_blocks(self) -> None:
        errors = validate_release_state(self.fixture.repo, self.fixture.catalog, {})
        self.assertTrue(any("BLUT_BACKENDS_REPO" in e for e in errors))

    def test_a_second_declared_external_is_also_required(self) -> None:
        """Externals are a TABLE, and every entry is checked or reported.

        The migration turned one hard-coded `blut_backends` into a table,
        because BLUT 1.0 ships two components that are not in this repository.
        A loop that stopped after the first would leave the second unchecked and
        still report the release as validated.
        """
        catalog = self.fixture.catalog.read_text(encoding="utf-8")
        self.fixture.write(
            "release.toml",
            catalog
            + "\n[external.blut_cookbook_lamquant]\n"
            + 'env = "BLUT_COOKBOOK_LAMQUANT_REPO"\n'
            + 'name = "blut-cookbook-lamquant"\n'
            + 'manifest = "Cargo.toml"\n',
        )
        errors = validate_release_state(
            self.fixture.repo,
            self.fixture.catalog,
            {"blut_backends": self.fixture.backends},
        )
        self.assertTrue(any("BLUT_COOKBOOK_LAMQUANT_REPO" in e for e in errors))
        self.assertFalse(any("BLUT_BACKENDS_REPO" in e for e in errors))

    def test_the_engine_prefix_is_read_from_the_catalog(self) -> None:
        """A prefix that matches nothing empties the SBOM matrix check.

        The matrix is derived from the manifests carrying the prefix, so a wrong
        prefix does not FAIL the check, it removes it -- which is exactly the
        shape the hard-coded `training/engine/` constant had once the packages
        stopped living there. Planted in both directions: the same broken
        workflow is a finding under the real prefix and invisible under a prefix
        that matches nothing.
        """
        broken = WORKFLOW.replace("blut-web-ui|web/ui\n", "")
        self.fixture.write("training/engine/.github/workflows/release.yml", broken)
        errors = validate_release_state(
            self.fixture.repo,
            self.fixture.catalog,
            {"blut_backends": self.fixture.backends},
        )
        self.assertTrue(
            any("SBOM matrix misses blut-web-ui|web/ui" in e for e in errors),
            f"the real prefix must find the missing SBOM row; got {errors}",
        )

        catalog = self.fixture.catalog.read_text(encoding="utf-8")
        self.fixture.write(
            "release.toml",
            catalog.replace(
                'engine_prefix = "training/engine/"', 'engine_prefix = "nowhere/"'
            ),
        )
        errors = validate_release_state(
            self.fixture.repo,
            self.fixture.catalog,
            {"blut_backends": self.fixture.backends},
        )
        self.assertFalse(
            any("SBOM matrix misses" in e for e in errors),
            "a prefix matching nothing must leave no engine packages to check, "
            "which is what makes the field load-bearing",
        )


if __name__ == "__main__":
    unittest.main()
