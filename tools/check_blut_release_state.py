#!/usr/bin/env python3
"""Fail-closed structural validator for the BLUT 1.0 release graph.

WHERE THIS CAME FROM AND WHY IT IS HERE NOW. It lived in Quitetall/LamQuant as
`tools/check_blut_release_state.py` until 2026-09-07, one of forty-two
gate-shaped commands that repository had accreted. It never measured a fact
about that tree: it validates THIS repository's release catalog against THIS
repository's package manifests and THIS repository's release workflow. Every
manifest path it read was spelled `training/engine/...`, which is only the
submodule path it happened to be reached through. LQ-WAR-0001 M4 named the move
as owed and could not perform it, because that session could not push here.

WHAT CHANGED IN THE MOVE, and nothing else did:

  * `engine_prefix` and `release_workflow` are CATALOG fields rather than
    constants. The prefix was hard-coded `training/engine/`; here it is empty,
    because the packages are in this repository. Declaring it keeps the SBOM
    component paths derivable from the manifest paths in both spellings instead
    of being true only for one checkout shape.
  * externals are a TABLE, not one hard-coded `blut_backends`. Two components
    the BLUT 1.0 release ships are not in this repository -- the standard
    cookbook (Quitetall/blut-backends) and the LamQuant cookbook
    (Quitetall/blut-cookbook-lamquant) -- and each is declared with the
    environment variable that names its checkout. A declared external with no
    checkout supplied is an ERROR, never a skip: an unchecked component of a
    release is exactly what this gate exists to refuse.

Everything the gate asserts is unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


class ReleaseStateError(ValueError):
    pass


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseStateError(f"cannot read TOML {path}: {exc}") from exc


def exact_version(value: object, expected: str) -> bool:
    return value == f"={expected}"


def validate_internal_dependency_versions(
    manifest_path: Path,
    manifest: dict[str, Any],
    internal_names: set[str],
    expected: str,
) -> list[str]:
    errors: list[str] = []
    for table_name in ("dependencies", "dev-dependencies", "build-dependencies"):
        table = manifest.get(table_name, {})
        if not isinstance(table, dict):
            errors.append(f"{manifest_path}: [{table_name}] must be a table")
            continue
        for alias, spec in table.items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            package_name = spec.get("package", alias)
            if package_name not in internal_names:
                continue
            if not exact_version(spec.get("version"), expected):
                errors.append(
                    f"{manifest_path}: path dependency {package_name} must pin "
                    f'version "={expected}"'
                )
    return errors


def validate_workflow(
    workflow_path: Path,
    publish_chain: list[str],
    binary_artifacts: list[str],
    engine_sboms: list[tuple[str, str]],
    cargo_sbom_version: str,
    wasm_bindgen_cli_version: str,
) -> list[str]:
    try:
        text = workflow_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read release workflow {workflow_path}: {exc}"]

    errors: list[str] = []
    offsets: list[int] = []
    for name in publish_chain:
        matches = list(
            re.finditer(rf"(?m)^\s*publish\s+{re.escape(name)}(?:\s|$)", text)
        )
        if len(matches) != 1:
            errors.append(
                f"{workflow_path}: publish command for {name} must occur exactly once"
            )
        else:
            offsets.append(matches[0].start())
    if len(offsets) == len(publish_chain) and offsets != sorted(offsets):
        errors.append(
            f"{workflow_path}: publish commands do not follow catalog order "
            + " -> ".join(publish_chain)
        )

    matrix_match = re.search(r"(?m)^\s*crate:\s*\[([^\]]+)\]", text)
    if not matrix_match:
        errors.append(f"{workflow_path}: sidecar binary matrix is absent")
    else:
        matrix = {item.strip() for item in matrix_match.group(1).split(",")}
        expected_matrix = set(binary_artifacts)
        if matrix != expected_matrix:
            errors.append(
                f"{workflow_path}: binary matrix must equal "
                + ", ".join(sorted(expected_matrix))
            )

    required_fragments = (
        "environment: crates-io",
        "CARGO_REGISTRY_TOKEN",
        "scripts/k8s_kind_smoke.sh",
        "cargo sbom --project-directory \"$path\" --output-format spdx_json_2_3",
        "cargo sbom --project-directory \"$path\" --output-format cyclone_dx_json_1_6",
        f"cargo install cargo-sbom --version {cargo_sbom_version} --locked",
        f"cargo install wasm-bindgen-cli --version {wasm_bindgen_cli_version} --locked",
        "(cd crates/blut-web && bash scripts/build_ui.sh)",
        '(cd dist && sha256sum -- * > SHA256SUMS)',
        '(cd dist && sha256sum -c SHA256SUMS)',
        'if [ "$QUIET_BENCHMARK_SHA" != "$GITHUB_SHA" ]; then',
    )
    for fragment in required_fragments:
        if fragment not in text:
            errors.append(f"{workflow_path}: missing release guard {fragment!r}")
    # These patterns intentionally recognize the workflow's simple, unnested
    # shell guards. Introduce a shell-aware parser before nesting control flow.
    dashboard_guards = (
        (
            "missing-or-empty dashboard WASM",
            r"if\s+\[\s*!\s+-s\s+crates/blut-web/ui/dist/blut_web_ui_bg\.wasm\s*\]"
            r"\s*;\s*then(?:(?!\n\s*fi\b)[\s\S])*?\n\s*exit\s+1\s*\n\s*fi\b",
        ),
        (
            "API-only dashboard fallback",
            r'if\s+grep\s+-aFq\s+"blut-web\s+\(API-only\s+build\)"\s+"\$BIN"'
            r"\s*;\s*then(?:(?!\n\s*fi\b)[\s\S])*?\n\s*exit\s+1\s*\n\s*fi\b",
        ),
    )
    for label, pattern in dashboard_guards:
        if not re.search(pattern, text):
            errors.append(f"{workflow_path}: missing fail-closed {label} guard")
    for name, path in engine_sboms:
        if not re.search(
            rf"(?m)^\s*{re.escape(name)}\|{re.escape(path)}\s*$", text
        ):
            errors.append(f"{workflow_path}: SBOM matrix misses {name}|{path}")
    return errors


def validate_release_state(
    repo: Path,
    catalog_path: Path,
    externals: dict[str, Path] | None = None,
) -> list[str]:
    externals = externals or {}
    catalog = load_toml(catalog_path)
    if catalog.get("schema_version") != 1:
        raise ReleaseStateError("release catalog schema_version must equal 1")
    engine_prefix = catalog.get("engine_prefix", "")
    workflow_relative = catalog.get("release_workflow", ".github/workflows/release.yml")
    if not isinstance(engine_prefix, str) or not isinstance(workflow_relative, str):
        raise ReleaseStateError(
            "engine_prefix and release_workflow must be strings when declared"
        )
    expected = catalog.get("release_version")
    if not isinstance(expected, str) or not re.fullmatch(r"\d+\.\d+\.\d+", expected):
        raise ReleaseStateError("release_version must be an exact stable semver")
    if catalog.get("release_tag") != f"v{expected}":
        raise ReleaseStateError("release_tag must equal v<release_version>")

    packages = catalog.get("rust_packages")
    if not isinstance(packages, list) or not packages:
        raise ReleaseStateError("release catalog must declare rust_packages")
    internal_names = {
        package.get("name")
        for package in packages
        if isinstance(package, dict) and isinstance(package.get("name"), str)
    }
    errors: list[str] = []
    toolchain = catalog.get("toolchain", {})
    if not isinstance(toolchain, dict):
        raise ReleaseStateError("toolchain must be a table")
    cargo_sbom_version = toolchain.get("cargo_sbom")
    wasm_bindgen_cli_version = toolchain.get("wasm_bindgen_cli")
    if not all(
        isinstance(value, str)
        for value in (cargo_sbom_version, wasm_bindgen_cli_version)
    ):
        raise ReleaseStateError(
            "toolchain must pin cargo_sbom and wasm_bindgen_cli"
        )

    for package in packages:
        if not isinstance(package, dict):
            errors.append("release catalog rust_packages entries must be tables")
            continue
        name = package.get("name")
        relative = package.get("manifest")
        distribution = package.get("distribution")
        if not all(isinstance(value, str) for value in (name, relative, distribution)):
            errors.append("release catalog rust package entry has invalid fields")
            continue
        manifest_path = repo / relative
        manifest = load_toml(manifest_path)
        package_table = manifest.get("package", {})
        if package_table.get("name") != name:
            errors.append(f"{manifest_path}: package name must equal {name}")
        if package_table.get("version") != expected:
            errors.append(f"{manifest_path}: version must equal {expected}")
        publish = package_table.get("publish")
        if distribution == "crates-io" and publish != ["crates-io"]:
            errors.append(f"{manifest_path}: publish must equal [\"crates-io\"]")
        if distribution != "crates-io" and publish is not False:
            errors.append(f"{manifest_path}: non-registry package must set publish = false")
        if name == "blut-web-ui":
            wasm_bindgen = manifest.get("dependencies", {}).get("wasm-bindgen", {})
            if not isinstance(wasm_bindgen, dict) or wasm_bindgen.get(
                "version"
            ) != f"={wasm_bindgen_cli_version}":
                errors.append(
                    f"{manifest_path}: wasm-bindgen must pin "
                    f'"={wasm_bindgen_cli_version}" to match wasm-bindgen-cli'
                )
        errors.extend(
            validate_internal_dependency_versions(
                manifest_path, manifest, internal_names, expected
            )
        )

    python_packages = catalog.get("python_packages", [])
    if not isinstance(python_packages, list):
        raise ReleaseStateError("python_packages must be an array of tables")
    for package in python_packages:
        manifest_path = repo / package["manifest"]
        manifest = load_toml(manifest_path)
        project = manifest.get("project", {})
        if project.get("name") != package["name"]:
            errors.append(f"{manifest_path}: Python project name mismatch")
        if project.get("version") != expected:
            errors.append(f"{manifest_path}: version must equal {expected}")

    external_table = catalog.get("external", {})
    if not isinstance(external_table, dict):
        raise ReleaseStateError("external must be a table of component tables")
    for key in sorted(external_table):
        external = external_table[key]
        if not isinstance(external, dict):
            raise ReleaseStateError(f"external.{key} must be a table")
        checkout = externals.get(key)
        if checkout is None:
            # An error, never a skip. A component of the release that nobody
            # looked at is the hole this gate exists to refuse, and "the
            # checkout was not supplied" and "the checkout is correct" must
            # never produce the same verdict.
            errors.append(
                f"{external.get('env', key.upper())} must name the {key} checkout"
            )
            continue
        manifest_path = checkout / external.get("manifest", "Cargo.toml")
        manifest = load_toml(manifest_path)
        package_table = manifest.get("package", {})
        if package_table.get("name") != external.get("name"):
            errors.append(f"{manifest_path}: external {key} package name mismatch")
        if package_table.get("version") != expected:
            errors.append(f"{manifest_path}: version must equal {expected}")

    publish_chain = catalog.get("publish_chain")
    binary_artifacts = catalog.get("binary_artifacts")
    if not isinstance(publish_chain, list) or not all(
        isinstance(item, str) for item in publish_chain
    ):
        raise ReleaseStateError("publish_chain must be an array of strings")
    if not isinstance(binary_artifacts, list) or not all(
        isinstance(item, str) for item in binary_artifacts
    ):
        raise ReleaseStateError("binary_artifacts must be an array of strings")
    registry_names = {
        package["name"]
        for package in packages
        if package.get("distribution") == "crates-io"
    }
    if set(publish_chain) != registry_names or len(publish_chain) != len(registry_names):
        errors.append("publish_chain must contain every crates-io package exactly once")
    expected_binaries = {
        package["name"]
        for package in packages
        if package.get("distribution") == "binary" or package.get("binary") is True
    }
    if set(binary_artifacts) != expected_binaries or len(binary_artifacts) != len(
        expected_binaries
    ):
        errors.append("binary_artifacts must contain every runnable release binary exactly once")

    engine_sboms: list[tuple[str, str]] = []
    for package in packages:
        manifest = package.get("manifest", "")
        name = package.get("name")
        # `package["name"]` here raised KeyError on a catalog entry with a
        # manifest and no name -- a traceback instead of the finding the loop
        # above had already recorded for that same entry. Found by running the
        # migrated gate against a hand-trimmed catalog; a gate that crashes on
        # malformed input is reporting nothing about the input.
        if not isinstance(manifest, str) or not isinstance(name, str) or not name:
            continue
        if not manifest.startswith(engine_prefix):
            continue
        relative = Path(manifest.removeprefix(engine_prefix))
        component_path = relative.parent.as_posix()
        engine_sboms.append((name, "." if component_path == "." else component_path))

    errors.extend(
        validate_workflow(
            repo / workflow_relative,
            publish_chain,
            binary_artifacts,
            engine_sboms,
            cargo_sbom_version,
            wasm_bindgen_cli_version,
        )
    )
    return errors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = Path("docs/release/blut-1.0/release.toml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo", type=Path, default=REPO_ROOT,
        help="the blut repository root (default: this checkout)",
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--external", action="append", default=[], metavar="NAME=PATH",
        help="checkout for a declared [external.NAME] component; repeatable. "
             "Without a flag the component's declared `env` variable is read.",
    )
    return parser.parse_args(argv)


def resolve_externals(catalog: dict[str, Any], flags: list[str]) -> dict[str, Path]:
    """Where each declared external component is checked out.

    A `--external NAME=PATH` flag wins over the component's declared `env`
    variable; an external with neither is left OUT of the map, so
    `validate_release_state` reports it as unchecked rather than passing over
    it. An empty env var is treated as absent for the same reason -- a variable
    set to nothing is not a checkout.
    """
    resolved: dict[str, Path] = {}
    declared = catalog.get("external", {})
    if isinstance(declared, dict):
        for key, entry in declared.items():
            variable = entry.get("env") if isinstance(entry, dict) else None
            value = os.environ.get(variable, "") if isinstance(variable, str) else ""
            if value.strip():
                resolved[key] = Path(value).absolute()
    for flag in flags:
        name, separator, path = flag.partition("=")
        if not separator or not name or not path:
            raise ReleaseStateError(f"--external must be NAME=PATH, got {flag!r}")
        resolved[name] = Path(path).absolute()
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.absolute()
    catalog = (
        args.catalog.absolute()
        if args.catalog
        else repo / DEFAULT_CATALOG
    )
    try:
        errors = validate_release_state(
            repo, catalog, resolve_externals(load_toml(catalog), args.external)
        )
    except ReleaseStateError as exc:
        print(f"BLUT 1.0 release state: INVALID: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"BLOCKED: {error}", file=sys.stderr)
        print(f"BLUT 1.0 release state: BLOCKED ({len(errors)} findings)", file=sys.stderr)
        return 2
    print("BLUT 1.0 release state: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
