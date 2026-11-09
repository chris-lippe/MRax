"""Release archive boundaries with independently constructed synthetic wheels."""

import zipfile

import pytest
from qualify_distribution import inspect_artifact


def wheel(tmp_path, *, extra=None, version="0.4.0", requires="", scripts=None, missing=False):
    path = tmp_path / "mrax-0.4.0-py3-none-any.whl"
    members = {
        "mrax-0.4.0.dist-info/METADATA": "Metadata-Version: 2.3\nName: mrax\nVersion: "
        + version
        + "\n"
        + requires,
        "mrax-0.4.0.dist-info/entry_points.txt": "[console_scripts]\n"
        + (scripts or "mrax-gui = mrax.web.cli:main\n"),
        "mrax/web/static/index.html": '<html><head><link rel="icon" href="/assets/icon.svg"></head>'
        '<body><script src="/assets/main.js"></script></body></html>',
        "mrax/web/static/assets/main.js": "console.log('Independent artifact fixture');",
        "mrax/web/static/assets/icon.svg": '<svg xmlns="http://www.w3.org/2000/svg"/>',
    }
    if missing:
        members.pop("mrax/web/static/assets/icon.svg")
    members.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    return path


def test_release_archive_contains_local_gui_assets_and_one_launcher(tmp_path):
    report = inspect_artifact(wheel(tmp_path), "0.4.0")
    assert report["version"] == "0.4.0" and report["assets"] == 2
    assert report["kind"] == "wheel"


@pytest.mark.parametrize(
    "name",
    [
        "data/private.json",
        "tmp/private.json",
        "mrax/.data/private.json",
        "frontend/test-results/session.json",
        "frontend/node_modules/package/index.js",
        "mrax/web/static/source.npy",
        "mrax/web/static/source.nii.gz",
        "mrax/web/static/fid",
        "mrax/gui/__pycache__/app.pyc",
    ],
)
def test_release_archives_reject_measurements_and_development_artifacts(tmp_path, name):
    with pytest.raises(ValueError, match="data|development"):
        inspect_artifact(wheel(tmp_path, extra={name: "independent synthetic sentinel"}), "0.4.0")


@pytest.mark.parametrize("name", ["mrax/gui/app.py", "mrax/gui/_deps.py", "mrax/gui/__main__.py"])
def test_release_archives_reject_retired_gui_modules(tmp_path, name):
    with pytest.raises(ValueError, match="Retired GUI"):
        inspect_artifact(wheel(tmp_path, extra={name: "independent synthetic sentinel"}), "0.4.0")


@pytest.mark.parametrize("requires", ["PySide6>=6", "PyQt6>=6", "qtpy>=2", "superqt>=1"])
def test_release_archives_reject_retired_widget_dependencies(tmp_path, requires):
    with pytest.raises(ValueError, match="widget"):
        inspect_artifact(wheel(tmp_path, requires="Requires-Dist: " + requires + "\n"), "0.4.0")


def test_release_archive_rejects_missing_assets_and_wrong_version_or_launcher(tmp_path):
    with pytest.raises(ValueError, match="asset"):
        inspect_artifact(wheel(tmp_path, missing=True), "0.4.0")
    with pytest.raises(ValueError, match="identity"):
        inspect_artifact(wheel(tmp_path, version="0.3.2"), "0.4.0")
    with pytest.raises(ValueError, match="launcher"):
        inspect_artifact(wheel(tmp_path, scripts="mrax-old = mrax.gui.app:main\n"), "0.4.0")


def test_release_archive_rejects_paths_outside_the_package(tmp_path):
    with pytest.raises(ValueError, match="Unsafe path"):
        inspect_artifact(wheel(tmp_path, extra={"../outside.txt": "synthetic"}), "0.4.0")
