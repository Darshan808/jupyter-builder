# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""Regression tests for the Module Federation runtime version option.

Module Federation 1 is the default and must stay the default. MF2's runtime
fails hard on shared packages that are not singletons and have no bundled
fallback - which is how JupyterLab consumes core packages missing from
`singletonPackages`, e.g. `@jupyterlab/docregistry` - so an extension built
against one JupyterLab minor version stops loading in the next. That is what
broke in jupyter-builder PR #54 and was reverted in #155. Upstream bug:
https://github.com/module-federation/core/issues/4651

The tests here guard both halves of that: the option threads through every
layer, and an accidental flip of the default is caught immediately.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from jupyter_builder import federated_extensions

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CLI threading (fast: the builder subprocess is stubbed out)
# ---------------------------------------------------------------------------


@pytest.fixture
def recorded_build(tmp_path, monkeypatch):
    """Run `build_labextension`/`watch_labextension` and capture the builder argv.

    Everything below the argument construction is stubbed: the point is which
    flags reach the builder, not what the builder does with them.
    """
    core_package_file = tmp_path / "core.package.json"
    core_package_file.write_text(json.dumps({"jupyterlab": {"singletonPackages": []}}))
    calls: list[list[str]] = []

    def fake_check_call(arguments, cwd=None):  # noqa: ARG001
        calls.append(list(arguments))
        return 0

    monkeypatch.setattr(federated_extensions.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(federated_extensions, "_which_node_js", lambda: "node")
    monkeypatch.setattr(
        federated_extensions,
        "_check_node_version",
        lambda *args, **kwargs: None,  # noqa: ARG005
    )

    def run(marker_pkg="@jupyter/builder", watch=False, **kwargs):
        monkeypatch.setattr(
            federated_extensions,
            "_ensure_builder",
            lambda *args, **kwargs: ("/fake/build-labextension.js", marker_pkg),  # noqa: ARG005
        )
        if watch:
            # Empty: the extension is treated as not yet installed, which routes
            # `watch_labextension` through the stubbed `develop_labextension_py`
            # rather than the symlink-repair branch.
            monkeypatch.setattr(
                federated_extensions,
                "get_federated_extensions",
                lambda *args, **kwargs: {},  # noqa: ARG005
            )
            (tmp_path / "package.json").write_text(
                json.dumps({"name": "fixture", "jupyterlab": {}}),
            )
            monkeypatch.setattr(
                federated_extensions,
                "develop_labextension_py",
                lambda *args, **kwargs: None,  # noqa: ARG005
            )
            federated_extensions.watch_labextension(
                tmp_path,
                labextensions_path=[str(tmp_path)],
                core_package_file=str(core_package_file),
                **kwargs,
            )
        else:
            federated_extensions.build_labextension(
                tmp_path,
                core_package_file=str(core_package_file),
                **kwargs,
            )
        return calls[-1]

    return run


def test_build_omits_the_flag_by_default(recorded_build):
    """No flag unless asked, so package.json still gets its say."""
    argv = recorded_build()
    assert "--module-federation-version" not in argv


@pytest.mark.parametrize("requested", [1, "1", 2, "2"])
def test_build_threads_the_requested_version(recorded_build, requested):
    argv = recorded_build(module_federation_version=requested)
    index = argv.index("--module-federation-version")
    assert argv[index + 1] == str(requested)
    # The flag and its value must precede the positional extension path, which
    # commander expects last.
    assert index + 1 < len(argv) - 1


def test_watch_omits_the_flag_by_default(recorded_build):
    argv = recorded_build(watch=True)
    assert "--module-federation-version" not in argv


@pytest.mark.parametrize("requested", [1, 2])
def test_watch_threads_the_requested_version(recorded_build, requested):
    argv = recorded_build(watch=True, module_federation_version=requested)
    index = argv.index("--module-federation-version")
    assert argv[index + 1] == str(requested)
    assert "--watch" in argv


@pytest.mark.parametrize("requested", [0, 3, "v1", "1.0", "one", -1, True])
def test_invalid_versions_are_rejected(recorded_build, requested):
    with pytest.raises(ValueError, match="expected 1 or 2"):
        recorded_build(module_federation_version=requested)


@pytest.mark.parametrize("unset", [None, ""])
def test_unset_versions_are_not_an_error(recorded_build, unset):
    argv = recorded_build(module_federation_version=unset)
    assert "--module-federation-version" not in argv


def test_invalid_version_is_rejected_before_the_builder_runs(tmp_path, monkeypatch):
    """Validation must not wait on `_ensure_builder`, which can run an install."""

    def fail(*args, **kwargs):  # noqa: ARG001
        msg = "_ensure_builder must not be reached"
        raise AssertionError(msg)

    monkeypatch.setattr(federated_extensions, "_ensure_builder", fail)
    with pytest.raises(ValueError, match="expected 1 or 2"):
        federated_extensions.build_labextension(tmp_path, module_federation_version=9)


def test_legacy_builder_rejects_the_option(recorded_build):
    """@jupyterlab/builder has no such flag; fail loudly instead of ignoring it."""
    with pytest.raises(ValueError, match="only supported by @jupyter/builder"):
        recorded_build(marker_pkg="@jupyterlab/builder", module_federation_version=2)


def test_legacy_builder_still_builds_without_the_option(recorded_build):
    argv = recorded_build(marker_pkg="@jupyterlab/builder")
    assert "--module-federation-version" not in argv
    assert "--core-path" in argv


# ---------------------------------------------------------------------------
# Builder behaviour (slow: compiles lib/ and runs real rspack builds)
# ---------------------------------------------------------------------------

# Runtime behaviour expected of each Module Federation version when the share
# scope offers a version OUTSIDE the extension's `requiredVersion` range, for a
# shared package that is not a singleton and has no bundled fallback (i.e. how
# `@jupyterlab/docregistry` is shared).
#
# LENIENT is what makes an extension built against one JupyterLab minor version
# load in the next, and is why version 1 is the default.
#
# MF2 is recorded as STRICT because that is what it currently does, not because
# it is correct: it raises RUNTIME-012, the bug tracked at
# https://github.com/module-federation/core/issues/4651.
#
# TO RE-EVALUATE MF2: change its entry below to LENIENT and run this file. If it
# passes, MF2 no longer hard-fails on non-singleton shares and has become a
# candidate for the default. If it fails, MF2 must stay opt-in.
LENIENT = "loads-despite-mismatch"
STRICT = "fails-to-load"

MISMATCH_BEHAVIOUR = {
    None: LENIENT,  # the default, i.e. no flag and no package.json key
    1: LENIENT,
    2: STRICT,
}

# The version the share scope offers. `SATISFYING` is inside the extension's
# `^4.5.0` requirement, `MISMATCHED` is deliberately just outside it.
SATISFYING = "4.6.0"
MISMATCHED = "5.0.0"

FIXTURE_PACKAGE_JSON = {
    "name": "mf-fixture-extension",
    "version": "0.1.0",
    "main": "lib/index.js",
    # Shared exactly the way @jupyterlab/docregistry is: present in the core
    # dependencies but absent from `singletonPackages`, so the generated config
    # gets `{requiredVersion: '^4.5.0', import: false}` - no bundled fallback,
    # not a singleton.
    "dependencies": {"@jupyterlab/docregistry": "^4.5.0"},
    "jupyterlab": {
        "extension": True,
        "outputDir": "labextension",
        # Builds for node so the probe can load the emitted chunks with
        # `require` instead of a DOM. Chunk transport is irrelevant to share
        # scope version resolution, which is what these tests exercise.
        "webpackConfig": "rspack.config.js",
    },
}

CORE_PACKAGE_JSON = {
    "name": "@jupyterlab/core-meta",
    "version": "4.5.0",
    "dependencies": {
        "@jupyterlab/docregistry": "~4.5.0",
        "@jupyterlab/application": "~4.5.0",
    },
    "jupyterlab": {"singletonPackages": ["@jupyterlab/application"]},
}

# Probes the built builder. Prints one JSON object per case on stdout:
#   - "config:*"  which Module Federation plugin the generated config uses
#   - "runtime:*" whether a built extension loads against a given share scope
_PROBE_JS = r"""
const path = require('path');
const fs = require('fs');
const vm = require('vm');

const [repoRoot, extPath, corePackageFile] = process.argv.slice(2);
const rspack = require(require.resolve('@rspack/core', { paths: [repoRoot] }));
const builder = require(path.join(repoRoot, 'lib', 'extensionConfig.js'));
const generateConfig = builder.default;

const SATISFYING = '%(satisfying)s';
const MISMATCHED = '%(mismatched)s';
const SHARED_PACKAGE = '@jupyterlab/docregistry';

function makeConfig(moduleFederationVersion) {
  return generateConfig({
    packagePath: extPath,
    corePackageFile,
    mode: 'production',
    moduleFederationVersion
  });
}

function federationPlugin(config) {
  return config[0].plugins.find(p =>
    /^ModuleFederationPlugin/.test(p.constructor.name)
  );
}

// The bundle assigns the container to a script-scoped `var _JUPYTERLAB`, which
// is only a global at script scope - hence vm rather than require().
function loadContainer(staticDir, name) {
  const warnings = [];
  const context = {
    _JUPYTERLAB: undefined,
    console: Object.assign({}, console, { warn: m => warnings.push(String(m)) }),
    require: request => {
      // Production builds append ?v=<contenthash> for cache busting.
      const file = path.join(staticDir, request.split('?')[0]);
      const mod = { exports: {} };
      vm.runInContext(
        '(function (module, exports, require) {' +
          fs.readFileSync(file, 'utf8') +
          '\n})',
        context
      )(mod, mod.exports, context.require);
      return mod.exports;
    }
  };
  vm.createContext(context);
  const entry = fs
    .readdirSync(staticDir)
    .find(f => /^remoteEntry\..*\.js$/.test(f));
  vm.runInContext(fs.readFileSync(path.join(staticDir, entry), 'utf8'), context);
  return { container: context._JUPYTERLAB[name], warnings };
}

function shareScope(offeredVersion) {
  return {
    [SHARED_PACKAGE]: {
      [offeredVersion]: {
        get: () => () => ({ DocumentRegistry: 'host-' + offeredVersion }),
        from: 'host',
        eager: true,
        loaded: 1
      }
    }
  };
}

function emit(name, payload) {
  console.log('__PROBE__' + JSON.stringify(Object.assign({ case: name }, payload)));
}

async function runtimeCase(name, moduleFederationVersion, offeredVersion) {
  const config = makeConfig(moduleFederationVersion);
  const stats = await new Promise((resolve, reject) =>
    rspack(config).run((err, stats) =>
      err ? reject(err) : resolve(stats)
    )
  );
  if (stats.hasErrors()) {
    emit(name, {
      status: 'build-failed',
      detail: JSON.stringify(stats.toJson({}).errors).slice(0, 1000)
    });
    return;
  }
  const staticDir = path.join(extPath, 'labextension', 'static');
  const { container, warnings } = loadContainer(staticDir, 'mf-fixture-extension');
  try {
    await container.init(shareScope(offeredVersion));
    const factory = await container.get('./index');
    const module = factory();
    emit(name, { status: 'loaded', resolved: module.DocumentRegistry, warnings });
  } catch (e) {
    emit(name, { status: 'failed', detail: String((e && e.message) || e), warnings });
  }
}

async function main() {
  emit('constant', { default: builder.DEFAULT_MODULE_FEDERATION_VERSION });

  for (const [name, requested] of [
    ['config:default', undefined],
    ['config:1', 1],
    ['config:2', 2]
  ]) {
    const plugin = federationPlugin(makeConfig(requested));
    emit(name, {
      plugin: plugin.constructor.name,
      shared: plugin._options.shared[SHARED_PACKAGE]
    });
  }

  // package.json opts in; the flag must still win over it.
  const pkgPath = path.join(extPath, 'package.json');
  const original = fs.readFileSync(pkgPath, 'utf8');
  const data = JSON.parse(original);
  data.jupyterlab.moduleFederationVersion = 2;
  fs.writeFileSync(pkgPath, JSON.stringify(data, null, 2));
  delete require.cache[require.resolve(pkgPath)];
  try {
    emit('config:package-json-2', {
      plugin: federationPlugin(makeConfig(undefined)).constructor.name
    });
    delete require.cache[require.resolve(pkgPath)];
    emit('config:package-json-2-flag-1', {
      plugin: federationPlugin(makeConfig(1)).constructor.name
    });
  } finally {
    fs.writeFileSync(pkgPath, original);
    delete require.cache[require.resolve(pkgPath)];
  }

  for (const [name, requested, offered] of [
    ['runtime:default:mismatched', undefined, MISMATCHED],
    ['runtime:1:mismatched', 1, MISMATCHED],
    ['runtime:2:mismatched', 2, MISMATCHED],
    ['runtime:1:satisfying', 1, SATISFYING],
    ['runtime:2:satisfying', 2, SATISFYING]
  ]) {
    await runtimeCase(name, requested, offered);
  }
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
"""


@pytest.fixture(scope="session")
def builder_lib():
    """Compile `lib/` so the probe can require the builder under test.

    Built unconditionally: a stale `lib/` from another branch would happily
    report the wrong default. `tsc` is incremental, so repeats are cheap.
    """
    yarn = REPO_ROOT / "jupyter_builder" / "yarn.js"
    env = {**os.environ, "YARN_ENABLE_IMMUTABLE_INSTALLS": "false"}
    if not (REPO_ROOT / "node_modules").exists():
        subprocess.run(["node", str(yarn)], cwd=REPO_ROOT, check=True, env=env)
    subprocess.run(["node", str(yarn), "build:lib"], cwd=REPO_ROOT, check=True, env=env)
    lib = REPO_ROOT / "lib" / "extensionConfig.js"
    assert lib.exists(), "building lib/ did not produce extensionConfig.js"
    return lib


@pytest.fixture(scope="session")
def probe_results(builder_lib, tmp_path_factory):  # noqa: ARG001
    """Run every builder-level case in one node process and collect the results."""
    workdir = tmp_path_factory.mktemp("mf")
    ext_path = workdir / "ext"
    (ext_path / "lib").mkdir(parents=True)
    (ext_path / "package.json").write_text(json.dumps(FIXTURE_PACKAGE_JSON, indent=2))
    (ext_path / "lib" / "index.js").write_text(
        "const { DocumentRegistry } = require('@jupyterlab/docregistry');\n"
        "module.exports = { DocumentRegistry };\n",
    )
    (ext_path / "rspack.config.js").write_text("module.exports = { target: 'node' };\n")
    core_package_file = workdir / "core.package.json"
    core_package_file.write_text(json.dumps(CORE_PACKAGE_JSON, indent=2))
    probe = workdir / "probe.js"
    probe.write_text(_PROBE_JS % {"satisfying": SATISFYING, "mismatched": MISMATCHED})

    completed = subprocess.run(
        ["node", str(probe), str(REPO_ROOT), str(ext_path), str(core_package_file)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    results = {}
    for line in completed.stdout.splitlines():
        if line.startswith("__PROBE__"):
            payload = json.loads(line.removeprefix("__PROBE__"))
            results[payload.pop("case")] = payload
    assert results, f"probe produced no results:\n{completed.stdout}\n{completed.stderr}"
    return results


@pytest.mark.slow
def test_default_module_federation_version_is_1(probe_results):
    """THE assertion of this file: the default must not drift off version 1.

    An accidental flip is what broke extension loading across JupyterLab minor
    versions in PR #54. Changing this expectation is a deliberate, breaking act.
    """
    assert probe_results["constant"]["default"] == 1


@pytest.mark.slow
def test_default_config_uses_the_v1_plugin(probe_results):
    """With no flag and no package.json key, the webpack-compatible plugin wins."""
    assert probe_results["config:default"]["plugin"] == "ModuleFederationPluginV1"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "expected_plugin"),
    [
        ("config:1", "ModuleFederationPluginV1"),
        ("config:2", "ModuleFederationPlugin"),
        # package.json opts in when the flag is absent...
        ("config:package-json-2", "ModuleFederationPlugin"),
        # ...and the flag wins when both are set.
        ("config:package-json-2-flag-1", "ModuleFederationPluginV1"),
    ],
)
def test_requested_version_selects_the_plugin(probe_results, case, expected_plugin):
    assert probe_results[case]["plugin"] == expected_plugin


@pytest.mark.slow
def test_fixture_shares_the_package_like_docregistry(probe_results):
    """Guard the fixture itself: no fallback, not a singleton, ranged requirement.

    If this drifts, the mismatch cases below stop testing anything.
    """
    shared = probe_results["config:default"]["shared"]
    assert shared == {"requiredVersion": "^4.5.0", "import": False}


@pytest.mark.slow
@pytest.mark.parametrize("requested", [1, 2])
def test_satisfying_version_always_loads(probe_results, requested):
    """Control: both runtimes load the extension when the versions do match."""
    result = probe_results[f"runtime:{requested}:satisfying"]
    assert result["status"] == "loaded", result
    assert result["resolved"] == f"host-{SATISFYING}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, MISMATCH_BEHAVIOUR[None]), (1, MISMATCH_BEHAVIOUR[1]), (2, MISMATCH_BEHAVIOUR[2])],
    ids=["default", "1", "2"],
)
def test_mismatched_version_behaviour(probe_results, requested, expected):
    """Pin what each runtime does when the share scope offers a version outside range.

    See MISMATCH_BEHAVIOUR above for how to re-evaluate MF2 with this test.
    """
    case = "runtime:default:mismatched" if requested is None else f"runtime:{requested}:mismatched"
    result = probe_results[case]

    if expected == LENIENT:
        assert result["status"] == "loaded", result
        # Lenient means: use whatever the host provides, and warn about it.
        assert result["resolved"] == f"host-{MISMATCHED}"
        assert result["warnings"], "expected a warning about the unsatisfied version"
    else:
        assert result["status"] == "failed", result
