"""Multi-file exports modify named files, never replace selected directories."""

import hashlib
import time

import pytest
from fastapi.testclient import TestClient

from mrax.gui.services.file_output import export_file_plan, publish_export_files
from mrax.web.app import create_app


def inventory(folder):
    return {
        path.relative_to(folder).as_posix(): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in [folder, *folder.rglob("*")]
    }


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def workspace(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, response, status="completed"):
    assert response.status_code == 200, response.text
    key = response.json()["id"]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = client.get("/api/jobs/" + key).json()
        if result["status"] not in {"queued", "running", "cancelling"}:
            assert result["status"] == status, result
            return result
        time.sleep(0.02)
    pytest.fail("Synthetic export did not finish")


def export(client, w, destination, *, kind="object", format="nifti", **options):
    obj = next(
        row
        for row in client.get("/api/workspaces/" + w["id"] + "/inspect").json()["objects"]
        if row["kind"] == "image"
    )
    return finish(
        client,
        client.post(
            "/api/workspaces/" + w["id"] + "/tasks",
            json={
                "revision": w["revision"],
                "operation": "export",
                "options": {
                    "kind": kind,
                    "format": format,
                    "object": obj["id"],
                    "root": "home",
                    "path": str(destination),
                    **options,
                },
            },
        ),
    )


def confirm(client, w, pending, *, status="completed", files=None):
    return finish(
        client,
        client.post(
            "/api/workspaces/" + w["id"] + "/tasks",
            json={
                "revision": w["revision"],
                "operation": "export",
                "options": {
                    "prepared_export": pending["summary"]["pending_export"],
                    "replace_files": files
                    if files is not None
                    else [row["path"] for row in pending["summary"]["conflicts"]],
                },
            },
        ),
        status=status,
    )


@pytest.mark.parametrize(
    ("kind", "format"),
    [
        ("object", "nifti"),
        ("object", "dicom"),
        ("rois", "roi_set"),
        ("package", "package"),
        ("state", "state"),
    ],
)
def test_multifile_exports_add_files_without_replacing_existing_contents(
    client, tmp_path, kind, format
):
    w = workspace(client)
    target = tmp_path / "results"
    (target / "unrelated").mkdir(parents=True)
    (target / "keep.txt").write_text("existing file")
    (target / "unrelated" / "keep.txt").write_text("nested file")
    # A legacy export marker must not cause wholesale replacement either.
    (target / ".mrax-export").write_text("MRax export\n")
    sibling = tmp_path / "another result"
    sibling.mkdir()
    (sibling / "keep").write_text("sibling file")
    before, beside = inventory(target), inventory(sibling)
    inode = target.stat().st_ino
    result = export(client, w, target, kind=kind, format=format, overwrite=True)
    assert result["summary"]["saved"] and not result["summary"].get("needs_confirmation")
    after = inventory(target)
    assert all(after[name] == value for name, value in before.items() if name != ".")
    assert target.stat().st_ino == inode
    assert inventory(sibling) == beside
    assert len(after) > len(before)
    assert not list(tmp_path.glob("results.previous-*"))


def test_file_collisions_are_reviewed_before_any_write_and_confirmation_reuses_preparation(
    client, tmp_path, monkeypatch
):
    w = workspace(client)
    target = tmp_path / "results"
    target.mkdir()
    export(client, w, target)
    files = sorted(target.glob("*.nii"))
    expected = files[0].read_bytes()
    files[0].write_bytes(b"previous image")
    (target / "keep").write_text("unrelated")
    before = inventory(target)
    pending = export(client, w, target, overwrite=True)
    assert not pending["summary"]["saved"]
    assert len(pending["summary"]["conflicts"]) == len(files)
    assert all(row["replaceable"] for row in pending["summary"]["conflicts"])
    assert inventory(target) == before
    prepared = client.app.state.jobs.prepared_export(pending["id"], w["id"], w["revision"])
    assert list((prepared / "files").glob("*.nii"))

    def do_not_regenerate(*_args, **_kwargs):
        raise AssertionError("A confirmed export must reuse its prepared files")

    monkeypatch.setattr("mrax.gui.services.export_workflow._write_export_files", do_not_regenerate)
    saved = confirm(client, w, pending)
    assert saved["summary"]["saved"]
    assert files[0].read_bytes() == expected
    assert (target / "keep").read_text() == "unrelated"
    assert not prepared.exists()


def test_cancelling_a_pending_export_leaves_all_destination_files_untouched(client, tmp_path):
    w = workspace(client)
    target = tmp_path / "results"
    target.mkdir()
    export(client, w, target)
    before = inventory(target)
    pending = export(client, w, target)
    prepared = client.app.state.jobs.prepared_export(pending["id"], w["id"], w["revision"])
    response = client.post("/api/jobs/" + pending["id"] + "/cancel")
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    assert not prepared.exists()
    assert inventory(target) == before
    denied = client.post(
        "/api/workspaces/" + w["id"] + "/tasks",
        json={
            "revision": w["revision"],
            "operation": "export",
            "options": {"prepared_export": pending["id"], "replace_files": []},
        },
    )
    assert denied.status_code == 422


@pytest.mark.parametrize("change", ["modified", "new collision", "removed"])
def test_changed_output_files_invalidate_confirmation_without_partial_writes(
    client, tmp_path, change
):
    w = workspace(client)
    target = tmp_path / "results"
    target.mkdir()
    export(client, w, target)
    files = sorted(target.glob("*.nii"))
    if change == "new collision":
        original = files[-1].read_bytes()
        files[-1].unlink()
    pending = export(client, w, target)
    if change == "modified":
        files[0].write_bytes(b"changed after review")
    elif change == "removed":
        files[0].unlink()
    else:
        files[-1].write_bytes(original)
    before = inventory(target)
    failed = confirm(client, w, pending, status="failed")
    assert "changed after review" in failed["detail"]
    assert inventory(target) == before


def test_conflicting_folders_and_unrelated_metadata_are_never_replaceable(client, tmp_path):
    w = workspace(client)
    target = tmp_path / "results"
    target.mkdir()
    export(client, w, target)
    name = next(target.glob("*.nii"))
    name.unlink()
    name.mkdir()
    (name / "keep").write_text("nested original")
    before = inventory(target)
    pending = export(client, w, target)
    assert any(not row["replaceable"] for row in pending["summary"]["conflicts"])
    confirm(client, w, pending, status="failed")
    assert inventory(target) == before
    target = tmp_path / "workspace output"
    target.mkdir()
    (target / "mrax_package.json").write_text('{"external": true}')
    before = inventory(target)
    pending = export(client, w, target, kind="package", format="package")
    assert pending["summary"]["conflicts"][0]["replaceable"] is False
    confirm(client, w, pending, status="failed")
    assert inventory(target) == before


@pytest.mark.parametrize("kind", ["rois", "package", "state"])
def test_native_multifile_exports_replace_only_reviewed_files_and_keep_obsolete_ones(
    client, tmp_path, kind
):
    w = workspace(client)
    target = tmp_path / "native output"
    target.mkdir()
    export(client, w, target, kind=kind)
    (target / "obsolete").mkdir()
    (target / "obsolete" / "keep").write_text("keep older output too")
    inode = target.stat().st_ino
    pending = export(client, w, target, kind=kind)
    assert pending["summary"]["needs_confirmation"]
    assert all(row["replaceable"] for row in pending["summary"]["conflicts"])
    saved = confirm(client, w, pending)
    assert saved["summary"]["saved"]
    assert target.stat().st_ino == inode
    assert (target / "obsolete" / "keep").read_text() == "keep older output too"
    assert not (target / ".mrax-export").exists()


def test_prepared_exports_require_the_same_study_revision_and_exact_file_list(client, tmp_path):
    w = workspace(client)
    target = tmp_path / "results"
    target.mkdir()
    export(client, w, target)
    pending = export(client, w, target)
    before = inventory(target)
    confirm(client, w, pending, files=["not-an-output"], status="failed")
    assert inventory(target) == before
    other = workspace(client)
    response = client.post(
        "/api/workspaces/" + other["id"] + "/tasks",
        json={
            "revision": other["revision"],
            "operation": "export",
            "options": {"prepared_export": pending["id"]},
        },
    )
    assert response.status_code == 422
    response = client.post(
        "/api/workspaces/" + w["id"] + "/actions",
        json={
            "revision": w["revision"],
            "action": "metadata.update",
            "options": {"metadata": {"note": "internal revision"}},
        },
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/workspaces/" + w["id"] + "/tasks",
        json={
            "revision": response.json()["revision"],
            "operation": "export",
            "options": {"prepared_export": pending["id"]},
        },
    )
    assert response.status_code == 422 and "study changed" in response.text
    assert inventory(target) == before


def test_native_roi_sidecars_remain_protected_after_being_imported_as_metadata(client, tmp_path):
    w = workspace(client)
    target = tmp_path / "roi output"
    target.mkdir()
    export(client, w, target, kind="rois")
    response = client.get(
        "/api/files/text",
        params={
            "root": "home",
            "path": str(target / "mrax_rois.json"),
        },
    )
    assert response.status_code == 200
    before = inventory(target)
    pending = export(client, w, target, kind="rois")
    sidecar = next(row for row in pending["summary"]["conflicts"] if row["path"].endswith(".json"))
    assert not sidecar["replaceable"] and "read-only" in sidecar["reason"]
    confirm(client, w, pending, status="failed")
    assert inventory(target) == before


def test_partial_write_failure_restores_files_without_removing_existing_directories(tmp_path):
    target, prepared = tmp_path / "results", tmp_path / "prepared"
    (target / "nested").mkdir(parents=True)
    (prepared / "new").mkdir(parents=True)
    (prepared / "nested").mkdir()
    original = target / "nested" / "replace"
    original.write_text("original")
    (target / "nested" / "unrelated").write_text("keep")
    (prepared / "nested" / "replace").write_text("replacement")
    (prepared / "new" / "array").write_text("new data")
    inode = (target / "nested").stat().st_ino
    before = original.stat().st_mtime_ns
    files, _ = export_file_plan(prepared, target, policy=lambda *_: (True, ""))

    def cancel(current, total, _detail):
        if current == total:
            raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        publish_export_files(
            prepared, target, files, replace_files=["nested/replace"], progress=cancel
        )
    assert original.read_text() == "original"
    assert original.stat().st_mtime_ns == before
    assert (target / "nested").stat().st_ino == inode
    assert (target / "nested" / "unrelated").read_text() == "keep"
    assert not (target / "new").exists()


def test_export_subfolders_cannot_redirect_writes_through_symlinks(tmp_path):
    target, prepared, elsewhere = (
        tmp_path / "results",
        tmp_path / "prepared",
        tmp_path / "elsewhere",
    )
    target.mkdir()
    elsewhere.mkdir()
    (prepared / "nested").mkdir(parents=True)
    (prepared / "nested" / "array").write_text("new data")
    (target / "nested").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        export_file_plan(prepared, target, policy=lambda *_: (True, ""))
    assert not list(elsewhere.iterdir())


@pytest.mark.parametrize("existing", [True, False])
def test_files_created_or_changed_during_publication_are_preserved(tmp_path, existing):
    prepared, target = tmp_path / "prepared", tmp_path / "results"
    prepared.mkdir()
    target.mkdir()
    (prepared / "a").write_text("new a")
    (prepared / "b").write_text("new b")
    if existing:
        (target / "b").write_text("previous b")
    files, _ = export_file_plan(prepared, target, policy=lambda *_: (True, ""))

    def concurrent_change(current, _total, _detail):
        if current == 1:
            (target / "b").write_text("external change")

    with pytest.raises(ValueError):
        publish_export_files(
            prepared,
            target,
            files,
            replace_files=["b"] if existing else [],
            progress=concurrent_change,
        )
    assert not (target / "a").exists()
    assert (target / "b").read_text() == "external change"
