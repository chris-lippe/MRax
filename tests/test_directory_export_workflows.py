"""Multi-file exports modify named files, never replace selected directories."""

import hashlib
import subprocess
import time
from threading import Event

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


def test_multifile_exports_add_files_without_replacing_existing_contents(client, tmp_path):
    w = workspace(client)
    target = tmp_path / "results"
    (target / "unrelated").mkdir(parents=True)
    (target / "keep.txt").write_text("existing file")
    (target / "unrelated" / "keep.txt").write_text("nested file")
    # An existing marker file is ordinary destination content.
    (target / ".mrax-export").write_text("MRax export\n")
    sibling = tmp_path / "another result"
    sibling.mkdir()
    (sibling / "keep").write_text("sibling file")
    before, beside = inventory(target), inventory(sibling)
    inode = target.stat().st_ino
    result = export(client, w, target, kind="object", format="nifti", overwrite=True)
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


def test_export_cancellation_matches_files_on_disk(client, tmp_path, monkeypatch):
    from mrax.gui.services import export_workflow, file_output

    study = workspace(client)

    def exercise(module, function, expected):
        destination = tmp_path / f"{expected}.mrax"
        reached, resume = Event(), Event()
        original = getattr(module, function)

        def paused(*args, **kwargs):
            result = original(*args, **kwargs)
            reached.set()
            assert resume.wait(10)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(module, function, paused)
            job = client.post(
                f"/api/workspaces/{study['id']}/tasks",
                json={
                    "revision": study["revision"],
                    "operation": "export",
                    "options": {"kind": "bundle", "root": "home", "path": str(destination)},
                },
            )
            try:
                assert reached.wait(10)
                client.post(f"/api/jobs/{job.json()['id']}/cancel").raise_for_status()
            finally:
                resume.set()
            finish(client, job, expected)
        assert destination.exists() == (expected == "completed")

    exercise(export_workflow, "save_workspace_bundle", "cancelled")
    exercise(file_output, "publish_new_file", "completed")


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
    link = target / "nested"
    try:
        link.symlink_to(elsewhere, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        # A directory junction exercises a real redirect under a standard Windows account.
        subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/j", "nested", r"..\elsewhere"],
            cwd=target,
            check=True,
            capture_output=True,
        )
        assert link.is_junction()
    assert link.samefile(elsewhere)
    with pytest.raises(ValueError, match="symbolic link"):
        export_file_plan(prepared, target, policy=lambda *_: (True, ""))
    assert not list(elsewhere.iterdir())
