# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

"""Generate the third-party license report for the vendored ``jupyter_builder/yarn.js``.

``yarn.js`` is a prebuilt upstream artifact: Yarn Berry bundles itself with esbuild, so
the ``JSONLicenseWebpackPlugin`` this project uses for webpack builds cannot see inside
it. Instead this script reconstructs the report from the dependency tree the bundle was
built from -- it checks out ``yarnpkg/berry`` at the tag matching the vendored bundle,
installs the production dependencies of ``@yarnpkg/cli``, and walks ``node_modules``.

The output deliberately matches the schema emitted by ``JSONLicenseWebpackPlugin``
(see ``src/webpack-plugins.ts``) so that downstream consumers can treat this report and
JupyterLab's ``third-party-licenses.json`` identically::

    {"packages": [{"name", "versionInfo", "licenseId", "extractedText"}]}

Note on accuracy: this slightly *over*-includes. esbuild tree-shakes the bundle, so a
few packages listed here may not be fully inlined into ``yarn.js``. That is deliberate
and safe -- for both the aggregate SPDX license expression and for Fedora-style
``Provides: bundled(npm(...))`` CVE tracking, listing a package that turned out not to
be shipped is harmless, while omitting one that was is not.

Usage::

    python scripts/generate_yarn_licenses.py 3.5.0
    python scripts/generate_yarn_licenses.py 3.5.0 --berry-checkout /path/to/berry
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# The bundle vendored here is not a stock `@yarnpkg/cli` build: it additionally carries
# the workspace-tools plugin, because `jlpm` exposes `yarn workspaces foreach`. This
# mirrors the patch applied in .github/workflows/verify-yarn-bundle.yml, which verifies
# the vendored bundle is byte-for-byte reproducible; the two must stay in step.
EXTRA_BUNDLE_PLUGINS = ("@yarnpkg/plugin-workspace-tools",)

# Filename prefixes that hold license text, checked case-insensitively.
LICENSE_PREFIXES = ("LICENSE", "LICENCE", "COPYING")

DEFAULT_OUTPUT = Path("THIRD_PARTY_LICENSES/yarn.js.third-party-licenses.json")


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run a command, streaming its output, and raise if it fails."""
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)  # noqa: S603


def clone_berry(version: str, dest: Path) -> None:
    """Clone yarnpkg/berry at the tag matching ``version``."""
    tag = f"@yarnpkg/cli/{version}"
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            "https://github.com/yarnpkg/berry.git",
            str(dest),
        ],
        cwd=dest.parent,
    )


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="bundled Yarn version, e.g. 3.5.0")
    parser.add_argument(
        "--berry-checkout",
        type=Path,
        help="reuse an existing Berry checkout at this path instead of cloning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"where to write the report (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the report and write it to disk."""
    args = parse_args()

    temp_dir: str | None = None
    try:
        if args.berry_checkout:
            checkout = args.berry_checkout
        else:
            temp_dir = tempfile.mkdtemp(prefix="berry-licenses-")
            checkout = Path(temp_dir) / "berry"
            clone_berry(args.version, checkout)
            patch_cli_manifest(checkout)

        if not (checkout / "node_modules").is_dir():
            install_production_deps(checkout)

        report, missing = build_report(checkout)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output} with {len(report['packages'])} packages", file=sys.stderr)
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
