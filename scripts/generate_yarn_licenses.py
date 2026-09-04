# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

"""Generate the third-party license report for the vendored ``jupyter_builder/yarn.js``.

``yarn.js`` is a prebuilt upstream artifact: Yarn Berry bundles itself with esbuild, so
the ``JSONLicenseWebpackPlugin`` this project uses for webpack builds cannot see inside
it. Instead this script reconstructs the report from the dependency tree the bundle was
built from -- it checks out ``yarnpkg/berry`` at the tag matching the vendored bundle,
installs the production dependencies of ``@yarnpkg/cli``, and walks ``node_modules``.

Three artifacts are written from that one walk, so they cannot drift apart:

* ``THIRD_PARTY_LICENSES/yarn.js.third-party-licenses.json`` -- the machine-readable
  report. Its schema deliberately matches the one emitted by
  ``JSONLicenseWebpackPlugin``

* ``THIRD_PARTY_LICENSES/yarn.js.LICENSE.txt`` -- the same data rendered in text form.
* the ``license`` field in ``pyproject.toml`` -- the aggregate SPDX expression, computed
  as the union of every ``licenseId`` in the report plus the licenses of the parts of
  this project that are not inside ``yarn.js``.

Note on accuracy: this slightly *over*-includes. esbuild tree-shakes the bundle, so a
few packages listed here may not be fully inlined into ``yarn.js``.

Usage::

    python scripts/generate_yarn_licenses.py 3.5.0
    python scripts/generate_yarn_licenses.py 3.5.0 --berry-checkout /path/to/berry
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# The bundle vendored here is not a stock `@yarnpkg/cli` build: it additionally carries
# the workspace-tools plugin, because `jlpm` exposes `yarn workspaces foreach`. This
# mirrors the patch applied in .github/workflows/verify-yarn-bundle.yml, which verifies
# the vendored bundle is byte-for-byte reproducible; the two must stay in step.
EXTRA_BUNDLE_PLUGINS = ("@yarnpkg/plugin-workspace-tools",)

# Filename prefixes that hold license text, checked case-insensitively.
LICENSE_PREFIXES = ("LICENSE", "LICENCE", "COPYING")

# Licenses covering the parts of the distribution that are not inside yarn.js, and so
# are not discoverable by walking Berry's dependency tree. They still belong in the
# aggregate expression, which has to cover everything in the wheel.
OTHER_LICENSES = (
    "BSD-3-Clause",  # jupyter_builder itself, see LICENSE
    "MIT",  # the vendored jupyter_builder/jupyterlab_semver.py, see semver.LICENSE.txt
)

# npm manifests in the wild still carry SPDX identifiers that have since been
# deprecated. PEP 639 wants a currently-valid expression, so normalise them.
DEPRECATED_SPDX = {
    "GPL-2.0": "GPL-2.0-only",
    "GPL-3.0": "GPL-3.0-only",
    "LGPL-2.1": "LGPL-2.1-only",
    "LGPL-3.0": "LGPL-3.0-only",
}

LICENSE_DIR = Path("THIRD_PARTY_LICENSES")
DEFAULT_JSON_OUTPUT = LICENSE_DIR / "yarn.js.third-party-licenses.json"
DEFAULT_TEXT_OUTPUT = LICENSE_DIR / "yarn.js.LICENSE.txt"
DEFAULT_PYPROJECT = Path("pyproject.toml")

RULE = "-" * 78


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run a command, streaming its output, and raise if it fails."""
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)  # noqa: S603


def capture(cmd: list[str], cwd: Path) -> str:
    """Run a command and return its stripped stdout."""
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def clone_berry(version: str, dest: Path) -> None:
    """Clone yarnpkg/berry at the tag matching ``version``."""
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            berry_tag(version),
            "https://github.com/yarnpkg/berry.git",
            str(dest),
        ],
        cwd=dest.parent,
    )


def berry_tag(version: str) -> str:
    """Return the Berry tag holding the sources for a given Yarn CLI version."""
    return f"@yarnpkg/cli/{version}"


def patch_cli_manifest(checkout: Path) -> None:
    """Add the extra plugins carried by the vendored bundle to the CLI manifest."""
    manifest_path = checkout / "packages" / "yarnpkg-cli" / "package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for plugin in EXTRA_BUNDLE_PLUGINS:
        manifest["dependencies"][plugin] = "workspace:^"
        bundled = manifest["@yarnpkg/builder"]["bundles"]["standard"]
        if plugin not in bundled:
            bundled.append(plugin)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify_checkout(checkout: Path, version: str) -> None:
    """Fail if a reused checkout cannot produce what a fresh clone would.

    ``--berry-checkout`` skips both the clone and the manifest patch, so the tree is
    whatever the caller left there: it may sit at the wrong tag, or lack the extra
    plugins the vendored bundle carries. Either would quietly yield a report that does
    not describe ``jupyter_builder/yarn.js``. Checked before installing, so a checkout
    that cannot give a faithful answer fails in seconds rather than after a full install.
    """
    tag = berry_tag(version)
    try:
        expected = capture(["git", "rev-parse", f"{tag}^{{commit}}"], cwd=checkout)
    except subprocess.CalledProcessError:
        msg = f"{checkout} has no {tag} tag; fetch it, or drop --berry-checkout to clone"
        raise RuntimeError(msg) from None

    head = capture(["git", "rev-parse", "HEAD"], cwd=checkout)
    if head != expected:
        msg = (
            f"{checkout} is at {head[:12]}, not {tag} ({expected[:12]}); "
            f"check out the tag, or drop --berry-checkout to clone"
        )
        raise RuntimeError(msg)

    manifest_path = checkout / "packages" / "yarnpkg-cli" / "package.json"
    dependencies = json.loads(manifest_path.read_text(encoding="utf-8"))["dependencies"]
    absent = [p for p in EXTRA_BUNDLE_PLUGINS if p not in dependencies]
    if absent:
        msg = (
            f"{manifest_path} does not depend on {', '.join(absent)}, which the vendored "
            f"bundle carries, so the report would understate it. Apply the same patch "
            f"patch_cli_manifest() makes, or drop --berry-checkout to clone"
        )
        raise RuntimeError(msg)


def verify_installed_tree(checkout: Path) -> None:
    """Fail if the installed tree lacks the extra plugins the vendored bundle carries.

    A post-condition of the install for every path, not just reused checkouts: a stale
    ``node_modules`` predating the manifest patch looks fine until you walk it.
    """
    node_modules = checkout / "node_modules"
    absent = [p for p in EXTRA_BUNDLE_PLUGINS if not (node_modules / p).is_dir()]
    if absent:
        msg = (
            f"{node_modules} is missing {', '.join(absent)}; the report would understate "
            f"the bundle. Delete node_modules and re-run to reinstall it"
        )
        raise RuntimeError(msg)


def install_production_deps(checkout: Path) -> None:
    """Install only the production dependency tree of ``@yarnpkg/cli``.

    Berry is a zero-install PnP repo, which has no ``node_modules`` to walk, so the
    install is forced through the node-modules linker.
    """
    env = {
        **os.environ,
        "YARN_NODE_LINKER": "node-modules",
        "YARN_ENABLE_IMMUTABLE_INSTALLS": "false",
    }
    run(
        ["node", "scripts/run-yarn.js", "workspaces", "focus", "@yarnpkg/cli", "--production"],
        cwd=checkout,
        env=env,
    )


def iter_package_dirs(node_modules: Path) -> Iterator[Path]:
    """Yield every package directory in a ``node_modules`` tree.

    Handles ``@scope/`` directories, nested ``node_modules``, and the symlinks that the
    node-modules linker creates for workspace packages.
    """
    stack = [node_modules]
    while stack:
        current = stack.pop()
        for entry in sorted(current.iterdir()):
            # is_dir() follows symlinks, which is what we want for workspace links.
            if entry.name == ".bin" or not entry.is_dir():
                continue
            if entry.name.startswith("@"):
                stack.append(entry)
                continue
            if (entry / "package.json").is_file():
                yield entry
            nested = entry / "node_modules"
            if nested.is_dir():
                stack.append(nested)


def license_id(manifest: dict[str, Any]) -> str:
    """Extract an SPDX identifier from a package manifest, or '' if absent."""
    license_field = manifest.get("license")
    if isinstance(license_field, str):
        return license_field
    # Deprecated npm formats, still seen in older transitive dependencies.
    if isinstance(license_field, dict):
        return str(license_field.get("type", ""))
    licenses = manifest.get("licenses")
    if isinstance(licenses, list):
        types = [entry.get("type", "") for entry in licenses if isinstance(entry, dict)]
        return " OR ".join(t for t in types if t)
    return ""


def license_text(package_dir: Path) -> str:
    """Return the verbatim license text shipped in a package, or '' if there is none.

    A package may ship several license files (dual-licensed packages often do); all are
    concatenated so no text is lost.
    """
    candidates = [
        entry
        for entry in package_dir.iterdir()
        if entry.is_file() and entry.name.upper().startswith(LICENSE_PREFIXES)
    ]
    # Shortest name first, so a plain LICENSE wins over LICENSE-MIT.
    candidates.sort(key=lambda p: (len(p.name), p.name))
    texts = [entry.read_text(encoding="utf-8", errors="replace").strip() for entry in candidates]
    return "\n\n".join(text for text in texts if text)


def build_report(checkout: Path) -> tuple[dict[str, Any], list[str]]:
    """Build the license report from an installed Berry checkout.

    Returns the report and the list of ``name@version`` entries with no license text,
    so a human can review them.
    """
    # Berry's own workspaces are symlinked into node_modules and carry no LICENSE file
    # of their own; they are covered by the repository-root license.
    root_license = (checkout / "LICENSE.md").read_text(encoding="utf-8").strip()

    packages: dict[tuple[str, str], dict[str, str]] = {}
    for package_dir in iter_package_dirs(checkout / "node_modules"):
        manifest = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
        name = manifest.get("name")
        if not name:
            continue
        version = str(manifest.get("version", ""))
        if (name, version) in packages:
            continue

        text = license_text(package_dir)
        if not text and "node_modules" not in package_dir.resolve().parts:
            # A workspace package: falls under Berry's root license.
            text = root_license

        packages[name, version] = {
            "name": name,
            "versionInfo": version,
            "licenseId": license_id(manifest),
            "extractedText": text,
        }

    ordered = sorted(packages.values(), key=lambda p: (p["name"], p["versionInfo"]))
    missing = [f"{p['name']}@{p['versionInfo']}" for p in ordered if not p["extractedText"]]
    return {"packages": ordered}, missing


def normalize_spdx(expression: str) -> str:
    """Replace deprecated SPDX identifiers in a single package's license expression."""
    return re.sub(
        r"[A-Za-z0-9.+-]+",
        lambda m: DEPRECATED_SPDX.get(m.group(0), m.group(0)),
        expression,
    )


def aggregate_expression(license_ids: Iterable[str]) -> str:
    """Combine per-package license ids into one SPDX expression covering all of them.

    Simple identifiers are ANDed together in alphabetical order. A package offering a
    choice of licenses contributes a parenthesised ``OR`` group, kept intact and placed
    last so the result stays readable.
    """
    simple: set[str] = set()
    compound: set[str] = set()
    for raw in license_ids:
        value = normalize_spdx(raw.strip().strip("()").strip())
        if not value:
            continue
        if " " in value:
            compound.add(f"({value})")
        else:
            simple.add(value)
    return " AND ".join([*sorted(simple), *sorted(compound)])


def render_text(report: dict[str, Any], version: str, commit: str, expression: str) -> str:
    """Render the human-readable license file from the machine-readable report."""
    packages = report["packages"]
    bundle_expression = aggregate_expression(p["licenseId"] for p in packages)
    lines = [
        "Third-party licenses for jupyter_builder/yarn.js",
        "",
        "jupyter_builder/yarn.js is a vendored copy of the Yarn CLI bundle.",
        "",
        "    Upstream repository: https://github.com/yarnpkg/berry",
        f"    Tag:                 {berry_tag(version)}",
        f"    Commit:              {commit}",
        "    npm package:         @yarnpkg/cli",
        f"    Version:             {version}",
        "",
        "THIS FILE IS GENERATED -- do not edit it by hand. It is rendered from",
        "yarn.js.third-party-licenses.json by scripts/generate_yarn_licenses.py; run",
        "the generator again instead, so the two files cannot disagree.",
        "",
        "The JSON report next to this file carries the same information in a",
        "machine-readable form, including the exact version of every package, for",
        "downstream packagers that need it.",
        "",
        f"Packages: {len(packages)}",
        f"Aggregate license expression for this bundle: {bundle_expression}",
        "",
    ]
    if expression != bundle_expression:
        # The distribution also contains code that is not part of yarn.js.
        lines += [
            "The expression recorded in the project metadata additionally covers the",
            f"rest of the distribution: {expression}",
            "",
        ]

    for package in packages:
        lines += [
            RULE,
            "",
            f"{package['name']} {package['versionInfo']}".rstrip(),
            f"SPDX-License-Identifier: {package['licenseId'] or 'unknown'}",
            "",
        ]
        if package["extractedText"]:
            lines.append(package["extractedText"])
        else:
            lines.append(
                "This package ships no license file; the identifier above is the one "
                "declared\nin its package.json.",
            )
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def update_pyproject(path: Path, expression: str) -> str:
    """Rewrite the ``license`` field in pyproject.toml, returning the previous value."""
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(r'^license = "(?P<value>[^"]*)"$', re.MULTILINE)
    matches = pattern.findall(content)
    if len(matches) != 1:
        msg = f"expected exactly one top-level license field in {path}, found {len(matches)}"
        raise RuntimeError(msg)
    path.write_text(pattern.sub(f'license = "{expression}"', content), encoding="utf-8")
    return str(matches[0])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="bundled Yarn version, e.g. 3.5.0")
    parser.add_argument(
        "--berry-checkout",
        type=Path,
        help="reuse an existing Berry checkout at this path instead of cloning; "
        "it must sit at the matching tag and carry the bundle's extra plugins",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"machine-readable report path (default: {DEFAULT_JSON_OUTPUT})",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=DEFAULT_TEXT_OUTPUT,
        help=f"human-readable report path (default: {DEFAULT_TEXT_OUTPUT})",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help=f"project metadata to update the SPDX expression in (default: {DEFAULT_PYPROJECT})",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the report and write it to disk."""
    args = parse_args()

    temp_dir: str | None = None
    try:
        if args.berry_checkout:
            checkout = args.berry_checkout
            verify_checkout(checkout, args.version)
        else:
            temp_dir = tempfile.mkdtemp(prefix="berry-licenses-")
            checkout = Path(temp_dir) / "berry"
            clone_berry(args.version, checkout)
            patch_cli_manifest(checkout)

        if not (checkout / "node_modules").is_dir():
            install_production_deps(checkout)

        verify_installed_tree(checkout)

        commit = capture(["git", "rev-parse", "HEAD"], cwd=checkout)
        report, missing = build_report(checkout)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    expression = aggregate_expression(
        [*OTHER_LICENSES, *(p["licenseId"] for p in report["packages"])],
    )

    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.text_output.write_text(
        render_text(report, args.version, commit, expression),
        encoding="utf-8",
    )
    previous = update_pyproject(args.pyproject, expression)

    print(f"{len(report['packages'])} packages", file=sys.stderr)
    print(f"wrote {args.json_output}", file=sys.stderr)
    print(f"wrote {args.text_output}", file=sys.stderr)
    print(f"\nSPDX expression\n  before: {previous}\n  after:  {expression}", file=sys.stderr)
    if missing:
        print(
            f"\n{len(missing)} package(s) ship no license text; review these by hand:",
            file=sys.stderr,
        )
        for entry in missing:
            print(f"  - {entry}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
