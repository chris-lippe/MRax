"""Ordinary filesystem workflows, using only independently synthetic data."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
from _synthetic import _write_minimal_bruker_scan
from fastapi.testclient import TestClient

from mrax.gui.services.file_output import export_file_plan, publish_export_files, write_file_atomic
from mrax.web.app import create_app
from mrax.web.filesystem import FileRoots, _mounted_folders, _standard_folders


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def study(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, response):
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(600):
        job = client.get("/api/jobs/" + job["id"]).json()
        if job["status"] not in {"queued", "running", "cancelling"}:
            assert job["status"] == "completed", job
            return job
        time.sleep(0.1)
    pytest.fail("File operation did not finish")


def test_system_paths_browse_parent_unicode_and_mounted_locations(tmp_path):
    other = tmp_path / "Studies and results Ω"
    other.mkdir()
    (other / ".hidden").write_text("hidden")
    (other / "metadata.json").write_text("{}")
    folders = FileRoots(scope="system")
    listing = folders.listing("home", str(other))
    assert listing["path"] == str(other)
    assert listing["parent"] == str(tmp_path)
    assert [entry["name"] for entry in listing["entries"]] == [".hidden", "metadata.json"]
    assert folders.resolve("home", str(other / "..")) == tmp_path
    assert folders.describe("home", str(other / "new.mrax"))["exists"] is False


def test_directory_mount_discovery_handles_escaped_and_hidden_paths(tmp_path):
    hidden = tmp_path / ".mounted"
    spaced = tmp_path / "Mounted studies"
    hidden.mkdir()
    spaced.mkdir()
    regular = tmp_path / "ordinary-file"
    regular.touch()
    mountinfo = tmp_path / "mountinfo"
    entries = [hidden, spaced, spaced, regular, tmp_path / "missing", "/proc", "/sys", "/dev"]
    mountinfo.write_text(
        "\n".join(
            f"{i + 1} 0 8:1 / {str(path).replace(' ', r'\040')} rw shared:1 - ext4 device rw"
            for i, path in enumerate(entries)
        )
        + "\nmalformed row\n1 0 8:1 / /tmp rw\n"
    )
    assert set(_mounted_folders(mountinfo)) == {hidden, spaced}
    assert _mounted_folders(tmp_path / "absent-mountinfo") == []


def test_mount_shortcuts_browse_contents_without_expanding_shared_mode(tmp_path, monkeypatch):
    mounted = tmp_path / ".mounted"
    mounted.mkdir()
    (mounted / ".metadata.json").write_text("{}")
    child = mounted / "Study folder"
    child.mkdir()
    monkeypatch.setattr("mrax.web.filesystem._mounted_folders", lambda: [mounted, Path.cwd()])
    folders = FileRoots(scope="system")
    shortcuts = [row for row in folders.catalog() if row["mounted"]]
    assert len(shortcuts) == 1
    shortcut = shortcuts[0]
    assert shortcut["label"] == ".mounted"
    assert shortcut["path"] == str(mounted)
    assert {entry["name"] for entry in folders.listing(shortcut["id"])["entries"]} == {
        ".metadata.json",
        "Study folder",
    }
    assert folders.checked_tree(shortcut["id"], str(child)) == child
    assert FileRoots([tmp_path], scope="shared").catalog() == [
        {"id": "root0", "label": tmp_path.name, "path": "", "writable": False, "mounted": False}
    ]


def test_standard_folder_shortcuts_use_existing_system_folders_only(tmp_path, monkeypatch):
    home = tmp_path / "Home"
    home.mkdir()
    names = ["Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"]
    for name in names:
        (home / name).mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "No configuration"))
    monkeypatch.setattr("mrax.web.filesystem._windows_user_folders", lambda: {})
    monkeypatch.setattr("mrax.web.filesystem._mounted_folders", lambda: [])
    folders = FileRoots(scope="system")
    catalog = {row["id"]: row for row in folders.catalog()}
    for name in names:
        assert catalog[name.lower()]["label"] == name
        assert folders.resolve(name.lower()) == home / name
        assert folders.listing(name.lower())["entries"] == []
    assert len(FileRoots([tmp_path], scope="shared").catalog()) == 1
    (home / "Music").rmdir()
    assert "music" not in {key for key, _, _ in _standard_folders(home)}
    assert not (home / "Music").exists()


@pytest.mark.skipif(os.name == "nt", reason="XDG folder configuration applies on Unix")
def test_standard_folders_respect_xdg_redirects_and_disabled_folders(tmp_path, monkeypatch):
    home = tmp_path / "Home"
    home.mkdir()
    documents = home / "Research documents Ω"
    documents.mkdir()
    pictures = tmp_path / "Shared pictures"
    pictures.mkdir()
    (home / "Downloads").mkdir()
    config = tmp_path / "Configuration"
    config.mkdir()
    (config / "user-dirs.dirs").write_text(
        'XDG_DOCUMENTS_DIR="$HOME/Research documents Ω"\n'
        f'XDG_PICTURES_DIR="{pictures}"\n'
        'XDG_DOWNLOAD_DIR="${HOME}/"\n'
        'XDG_MUSIC_DIR="$(touch should-not-exist)"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr("mrax.web.filesystem._windows_user_folders", lambda: {})
    folders = {key: path for key, _, path in _standard_folders(home)}
    assert folders == {"documents": documents, "pictures": pictures}
    assert not (home / "Music").exists()
    assert not (tmp_path / "should-not-exist").exists()


def test_standard_folders_use_redirected_windows_locations(tmp_path, monkeypatch):
    home = tmp_path / "Home"
    home.mkdir()
    local = home / "Documents"
    local.mkdir()
    redirected = tmp_path / "Redirected documents Ω"
    redirected.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "No configuration"))
    monkeypatch.setattr(
        "mrax.web.filesystem._windows_user_folders", lambda: {"documents": redirected}
    )
    assert _standard_folders(home) == [("documents", "Documents", redirected)]


def test_unavailable_drives_do_not_block_the_remaining_file_locations(tmp_path, monkeypatch):
    unavailable = tmp_path / "Unavailable drive"
    available = tmp_path / "Available drive"
    available.mkdir()
    original = Path.is_dir

    def is_dir(path):
        if path == unavailable:
            raise PermissionError("Drive is unavailable")
        return original(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    monkeypatch.setattr("mrax.web.filesystem.os.listdrives", lambda: [unavailable, available], raising=False)
    monkeypatch.setattr("mrax.web.filesystem._mounted_folders", lambda: [])
    catalog = {row["id"]: row for row in FileRoots(scope="system").catalog()}
    assert "drive0" not in catalog
    assert catalog["drive1"]["path"] == str(available)
    assert "home" in catalog


def test_workspace_save_open_and_overwrite_are_direct_and_revision_checked(client, tmp_path):
    original = study(client)
    target = tmp_path / "Study Ω.mrax"
    command = {"root": "home", "path": str(target), "revision": original["revision"]}
    saved = client.post("/api/workspaces/" + original["id"] + "/save-file", json=command)
    assert saved.status_code == 200, saved.text
    assert target.is_file()
    assert saved.json()["dirty"] is False
    assert saved.json()["file_location"] == {"root": "home", "path": str(target)}
    before = target.read_bytes()
    assert (
        client.post("/api/workspaces/" + original["id"] + "/save-file", json=command).status_code
        == 422
    )
    assert target.read_bytes() == before
    stale = {**command, "revision": original["revision"] - 1, "overwrite": True}
    assert (
        client.post("/api/workspaces/" + original["id"] + "/save-file", json=stale).status_code
        == 409
    )
    assert target.read_bytes() == before
    assert (
        client.post(
            "/api/workspaces/" + original["id"] + "/save-file", json={**command, "overwrite": True}
        ).status_code
        == 200
    )
    opened = client.post("/api/workspaces/open", json={"root": "home", "path": str(target)})
    assert opened.status_code == 200, opened.text
    restored = opened.json()
    assert restored["shape"] == original["shape"]
    assert restored["experiments"] == original["experiments"]
    assert restored["rois"] == original["rois"]
    np.testing.assert_array_equal(
        np.frombuffer(
            client.get("/api/workspaces/" + original["id"] + "/slice").content, dtype="<f4"
        ),
        np.frombuffer(
            client.get("/api/workspaces/" + restored["id"] + "/slice").content, dtype="<f4"
        ),
    )


def test_direct_bruker_study_folder_import(client, tmp_path):
    source = tmp_path / "Synthetic Bruker study"
    _write_minimal_bruker_scan(source)
    job = finish(
        client,
        client.post(
            "/api/import",
            json={
                "root": "home",
                "path": str(source),
                "options": {"kind": "bruker", "field_strength_mhz": 400, "load_raw_kspace": False},
            },
        ),
    )
    workspace = client.get("/api/workspaces/" + job["workspace_id"]).json()
    assert workspace["loaded"]
    assert workspace["experiments"] == ["1"]
    pixels = np.frombuffer(
        client.get("/api/workspaces/" + workspace["id"] + "/slice").content, dtype="<f4"
    )
    np.testing.assert_array_equal(np.sort(pixels), [10, 12, 14, 16])


def test_direct_object_package_roi_and_text_exports(client, tmp_path):
    workspace = study(client)
    key = workspace["id"]
    inspection = client.get("/api/workspaces/" + key + "/inspect").json()
    obj = next(row for row in inspection["objects"] if row["kind"] == "image")
    target = tmp_path / "Image Ω.npy"
    command = {
        "revision": workspace["revision"],
        "operation": "export",
        "options": {
            "kind": "object",
            "object": obj["id"],
            "format": "npy",
            "root": "home",
            "path": str(target),
        },
    }
    result = finish(client, client.post("/api/workspaces/" + key + "/tasks", json=command))
    assert result["summary"]["path"] == str(target)
    assert not result["summary"].get("download")
    values = np.load(target, allow_pickle=False)
    assert list(values.shape) == obj["shape"]
    before = target.read_bytes()
    response = client.post("/api/workspaces/" + key + "/tasks", json=command)
    job = response.json()
    for _ in range(600):
        job = client.get("/api/jobs/" + job["id"]).json()
        if job["status"] not in {"queued", "running"}:
            break
        time.sleep(0.1)
    assert job["status"] == "failed"
    assert target.read_bytes() == before
    package = tmp_path / "Study package"
    finish(
        client,
        client.post(
            "/api/workspaces/" + key + "/tasks",
            json={
                "revision": workspace["revision"],
                "operation": "export",
                "options": {"kind": "package", "root": "home", "path": str(package)},
            },
        ),
    )
    assert (package / "mrax_package.json").is_file()
    assert (
        client.post("/api/workspaces/open", json={"root": "home", "path": str(package)}).status_code
        == 200
    )
    text = tmp_path / "timecourse.csv"
    content = '"TE (ms)","ROI (ms)"\r\n1,80\r\n'
    written = client.post(
        "/api/files/text",
        json={
            "root": "home",
            "path": str(text),
            "workspace_id": key,
            "revision": workspace["revision"],
            "content": content,
        },
    )
    assert written.status_code == 200, written.text
    assert text.read_bytes() == content.encode()
    assert client.get("/api/files/text", params={"root": "home", "path": str(text)}).json()[
        "content"
    ] == content.replace("\r\n", "\n")


def test_shared_mode_preserves_boundaries_and_read_only_locations(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (tmp_path / "outside.json").write_text("{}")
    with TestClient(
        create_app(read_roots=[shared], filesystem_scope="shared", runtime_dir=tmp_path),
        base_url="http://localhost",
    ) as client:
        client.headers["x-mrax-token"] = client.get("/api/session").json()["token"]
        assert len(client.get("/api/files/roots").json()) == 1
        for path in ["../outside.json", str(tmp_path / "outside.json")]:
            assert (
                client.get("/api/files/text", params={"root": "root0", "path": path}).status_code
                == 422
            )
        assert (
            client.post("/api/files/folder", json={"root": "root0", "path": "new"}).status_code
            == 422
        )
        assert not (shared / "new").exists()


def test_atomic_outputs_keep_original_on_failure_and_never_replace_unrelated_folders(tmp_path):
    target = tmp_path / "result.txt"
    target.write_text("original")

    def fail(path):
        path.write_text("partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError):
        write_file_atomic(target, fail, overwrite=True)
    assert target.read_text() == "original"
    folder = tmp_path / "unrelated"
    folder.mkdir()
    (folder / "keep.txt").write_text("original")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "new.txt").write_text("new")
    files, conflicts = export_file_plan(prepared, folder, policy=lambda *_: (True, ""))
    assert not conflicts
    publish_export_files(prepared, folder, files)
    assert (folder / "keep.txt").read_text() == "original"


@pytest.mark.parametrize("format_name", ["npz", "json", "nifti", "dicom"])
def test_direct_image_exports_keep_all_generated_files(client, tmp_path, format_name):
    workspace = study(client)
    key = workspace["id"]
    inspection = client.get("/api/workspaces/" + key + "/inspect").json()
    obj = next(row for row in inspection["objects"] if row["kind"] == "image")
    folder = format_name in {"nifti", "dicom"}
    target = tmp_path / ("images-" + format_name if folder else "image." + format_name)
    result = finish(
        client,
        client.post(
            "/api/workspaces/" + key + "/tasks",
            json={
                "revision": workspace["revision"],
                "operation": "export",
                "options": {
                    "kind": "object",
                    "object": obj["id"],
                    "format": format_name,
                    "root": "home",
                    "path": str(target),
                },
            },
        ),
    )
    assert not result["summary"].get("download")
    if format_name == "npz":
        with np.load(target, allow_pickle=False) as archive:
            assert list(archive["data"].shape) == obj["shape"]
            assert "metadata" in archive.files
    elif format_name == "json":
        import json

        assert json.loads(target.read_text())["shape"] == obj["shape"]
    else:
        assert not (target / ".mrax-export").exists()
        files = list(target.rglob("*.nii" if format_name == "nifti" else "*.dcm"))
        assert files
        if format_name == "nifti":
            import nibabel as nib

            assert np.isfinite(nib.load(files[0]).get_fdata()).all()
        else:
            import pydicom

            image = pydicom.dcmread(files[0])
            assert np.isfinite(image.pixel_array).all()


@pytest.mark.parametrize("kind", ["state", "rois", "diagnostics"])
def test_direct_state_roi_and_report_exports(client, tmp_path, kind):
    workspace = study(client)
    target = tmp_path / (kind + ".json" if kind == "diagnostics" else kind)
    result = finish(
        client,
        client.post(
            "/api/workspaces/" + workspace["id"] + "/tasks",
            json={
                "revision": workspace["revision"],
                "operation": "export",
                "options": {"kind": kind, "root": "home", "path": str(target)},
            },
        ),
    )
    assert result["summary"]["path"] == str(target)
    if kind == "diagnostics":
        assert target.is_file()
    elif kind == "state":
        assert (
            client.post(
                "/api/workspaces/open", json={"root": "home", "path": str(target)}
            ).status_code
            == 200
        )
    else:
        assert (target / "mrax_rois.npz").is_file()
        assert (target / "mrax_rois.json").is_file()
