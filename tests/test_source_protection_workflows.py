"""Source immutability and explicit output contracts, with synthetic files only."""

import hashlib
import time
import zipfile

import numpy as np
import pytest
from _synthetic import _write_minimal_bruker_scan
from fastapi.testclient import TestClient

from mrax.data_struct.storage import (
    load_ismrmrd_array,
    load_ismrmrd_kspace,
    save_ismrmrd_array,
    save_ismrmrd_kspace,
    save_study_results,
)
from mrax.gui.services.analysis_workflows import import_source, run_operation, update_metadata
from mrax.gui.services.export_workflow import export_workspace
from mrax.gui.services.file_output import export_file_plan, publish_export_files
from mrax.gui.services.source_protection import check_workspace_output
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import (
    load_workspace_bundle,
    save_workspace_bundle,
    save_workspace_folder,
)
from mrax.gui.session import GuiSession, _original_image_snapshot
from mrax.io.bundle import extract_bundle, write_bundle
from mrax.io.image_export import export_image_files
from mrax.io.nifti import save_nifti
from mrax.web.app import create_app
from mrax.workflows import new_workflow, save_workflow


def inventory(folder):
    """Include mtimes and directory contents, so no-op rewrites are caught."""
    return {
        p.relative_to(folder).as_posix(): (
            p.stat().st_mtime_ns,
            p.stat().st_mode,
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        )
        for p in [folder, *folder.rglob("*")]
    }


def empty_workspace():
    session = GuiSession()
    session.create_empty_study("Independent source protection", field_strength_mhz=100)
    return StudyWorkspace(key="independent", label=session.study_name, session=session)


def wait_job(client, response, *, status="completed"):
    assert response.status_code == 200, response.text
    key = response.json()["id"]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = client.get("/api/jobs/" + key).json()
        if result["status"] not in {"queued", "running", "cancelling"}:
            assert result["status"] == status, result
            return result
        time.sleep(0.02)
    pytest.fail("Synthetic operation did not finish")


@pytest.fixture
def client(tmp_path):
    (tmp_path / "runtime").mkdir()
    with TestClient(
        create_app(runtime_dir=tmp_path / "runtime"), base_url="http://localhost"
    ) as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def test_import_edit_process_fit_and_bundle_leave_all_source_files_unchanged(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    template = synthetic_t2_session()
    values = np.asarray(template.image_array)[:, :6, :7, :1]
    path = source / "image.npy"
    np.save(path, values)
    (source / "image.json").write_text('{"note": "source metadata"}')
    before = inventory(source)
    workspace = import_source(
        empty_workspace(),
        path,
        {
            "kind": "numpy",
            "name": "Echoes",
            "dim_order": ["frame", "x", "y", "z"],
            "exp_type": "t2_se",
            "model_name": "t2_se",
            "observables": {"TE": {"values": [10, 20, 40, 60, 80, 120, 160, 200], "unit": "ms"}},
        },
        lambda *_: None,
    )
    original = np.asarray(workspace.session.original_image_array).copy()
    update_metadata(workspace, {"metadata": {"note": "internal override"}})
    run_operation(
        workspace,
        "process",
        {"method": "gaussian", "input": "image", "axes": ["x", "y"], "parameters": {"sigma": 1}},
        lambda *_: None,
    )
    run_operation(
        workspace, "full_fit", {"use_roi": False, "settings": {"num_steps": 80}}, lambda *_: None
    )
    assert workspace.fit_results_by_experiment
    np.testing.assert_array_equal(workspace.session.original_image_array, original)
    assert inventory(source) == before
    bundle = save_workspace_bundle(workspace, tmp_path / "session.mrax")
    restored = load_workspace_bundle(bundle, tmp_path / "restored", key="restored")
    assert restored.session.current_experiment["extra_metadata"]["note"] == "internal override"
    assert restored.source_files == workspace.source_files
    np.testing.assert_array_equal(restored.session.original_image_array, original)
    assert not np.asarray(restored.session.original_image_array).flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        check_workspace_output(restored, source, directory=True)
    assert inventory(source) == before


def test_bruker_import_and_internal_edits_do_not_add_files_to_the_study(tmp_path):
    source = tmp_path / "vendor"
    _write_minimal_bruker_scan(source)
    before = inventory(source)
    workspace = import_source(empty_workspace(), source, {"kind": "bruker"}, lambda *_: None)
    update_metadata(workspace, {"metadata": {"note": "internal"}})
    run_operation(
        workspace,
        "process",
        {"method": "gaussian", "input": "image", "axes": ["x", "y"], "parameters": {"sigma": 0.5}},
        lambda *_: None,
    )
    save_workspace_bundle(workspace, tmp_path / "vendor-session.mrax")
    assert inventory(source) == before


def test_original_numpy_snapshots_are_frozen_and_detached_from_file_mappings(tmp_path):
    path = tmp_path / "external.npy"
    np.save(path, np.arange(12, dtype=np.float32))
    mapped = np.load(path, mmap_mode="r+")
    original = _original_image_snapshot(mapped[2:8])
    assert not np.shares_memory(original, mapped)
    with pytest.raises(ValueError):
        original[0] = 99
    mapped[2] = 50
    mapped.flush()
    assert original[0] == 2
    assert not _original_image_snapshot(np.arange(3)).flags.writeable
    owner = np.arange(3)
    readonly_view = owner.view()
    readonly_view.flags.writeable = False
    frozen = _original_image_snapshot(readonly_view)
    owner[:] = 99
    np.testing.assert_array_equal(frozen, [0, 1, 2])


def test_ismrmrd_readers_use_read_only_handles_and_never_create_missing_data(tmp_path, monkeypatch):
    import mrax.data_struct.storage as storage

    image, raw = tmp_path / "image.mrd", tmp_path / "raw.mrd"
    save_ismrmrd_array(image, np.arange(12, dtype=np.float32).reshape(3, 4), "image")
    save_ismrmrd_kspace(raw, np.ones((3, 4), dtype=np.complex64))
    image.chmod(0o444)
    raw.chmod(0o444)
    before = inventory(tmp_path)
    dataset = storage.ismrmrd.Dataset
    modes = []

    def open_dataset(*args, **kwargs):
        modes.append(kwargs.get("mode"))
        assert kwargs.get("create_if_needed") is False and kwargs.get("mode") == "r"
        return dataset(*args, **kwargs)

    monkeypatch.setattr(storage.ismrmrd, "Dataset", open_dataset)
    assert load_ismrmrd_array(image, "image", (3, 4)).shape == (3, 4)
    assert load_ismrmrd_kspace(raw).shape == (3, 4)
    with pytest.raises((LookupError, ValueError, RuntimeError)):
        load_ismrmrd_array(image, "missing")
    with pytest.raises((FileNotFoundError, OSError)):
        load_ismrmrd_kspace(tmp_path / "missing.mrd")
    assert modes == ["r"] * 4
    assert inventory(tmp_path) == before


def test_low_level_mrd_exports_require_an_explicit_overwrite_option(tmp_path):
    path = tmp_path / "image.mrd"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    save_ismrmrd_array(path, values, "image")
    before = inventory(tmp_path)
    with pytest.raises(ValueError):
        save_ismrmrd_array(path, values * 2, "image")
    with pytest.raises(ValueError):
        save_ismrmrd_kspace(path, values.astype(np.complex64))
    assert inventory(tmp_path) == before
    with pytest.warns(UserWarning, match="overwrite"):
        save_ismrmrd_array(path, values * 2, "image", overwrite=True)
    np.testing.assert_array_equal(load_ismrmrd_array(path, "image", (3, 4)), values * 2)


def test_bundle_and_folder_writers_refuse_implicit_replacement_or_merging(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "array").write_bytes(b"synthetic")
    archive = write_bundle(tmp_path / "data.mrax", source)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "array").write_bytes(b"keep")
    before = inventory(destination)
    with pytest.raises(ValueError):
        extract_bundle(archive, destination)
    with pytest.raises(ValueError):
        write_bundle(archive, source)
    session = synthetic_t2_session()
    workspace = StudyWorkspace("synthetic", "Independent", session)
    with pytest.raises(ValueError):
        save_workspace_folder(workspace, destination)
    with pytest.raises(ValueError):
        save_study_results(session.study, [], destination, bundle=False)
    assert inventory(destination) == before
    with zipfile.ZipFile(tmp_path / "duplicate.zip", "w") as content:
        content.writestr("sub/../array", "one")
        content.writestr("array", "two")
    with pytest.raises(ValueError, match="Duplicate"):
        extract_bundle(tmp_path / "duplicate.zip", tmp_path / "unpacked")
    assert not (tmp_path / "unpacked").exists()


def test_multifile_image_export_checks_every_collision_before_writing(tmp_path):
    # Obtain the actual generated frame names independently of their spelling.
    initial = tmp_path / "first"
    initial.mkdir()
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    files = export_image_files(initial / "series.nii", values, dims={"frame": 2, "x": 3, "y": 4})
    collision = tmp_path / files[-1].name
    collision.write_bytes(b"keep this file")
    before = inventory(tmp_path)
    with pytest.raises(ValueError):
        export_image_files(tmp_path / "series.nii", values, dims={"frame": 2, "x": 3, "y": 4})
    assert inventory(tmp_path) == before
    with pytest.raises(ValueError):
        save_nifti(collision, values[0], np.eye(4))
    with pytest.warns(UserWarning, match="overwrite"):
        save_nifti(collision, values[0], np.eye(4), overwrite=True)


def test_confirmed_image_export_is_the_only_way_to_replace_an_imported_image(client, tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    path = source / "image.npy"
    values = np.random.default_rng(8).normal(10, 1, (2, 8, 9)).astype(np.float32)
    np.save(path, values)
    sidecar = source / "image.json"
    sidecar.write_text('{"note": "protected"}')
    before = inventory(source)
    result = wait_job(
        client,
        client.post(
            "/api/import",
            json={
                "root": "home",
                "path": str(path),
                "options": {"kind": "numpy", "name": "Image", "dim_order": ["frame", "x", "y"]},
            },
        ),
    )
    key = result["workspace_id"]
    w = client.get("/api/workspaces/" + key).json()
    wait_job(
        client,
        client.post(
            "/api/workspaces/" + key + "/tasks",
            json={
                "revision": w["revision"],
                "operation": "process",
                "options": {
                    "method": "gaussian",
                    "input": "image",
                    "axes": ["x", "y"],
                    "parameters": {"sigma": 1},
                },
            },
        ),
    )
    w = client.get("/api/workspaces/" + key).json()
    obj = next(
        row
        for row in client.get("/api/workspaces/" + key + "/inspect").json()["objects"]
        if row["kind"] == "image"
    )
    command = {
        "revision": w["revision"],
        "operation": "export",
        "options": {
            "kind": "object",
            "object": obj["id"],
            "format": "npy",
            "root": "home",
            "path": str(path),
        },
    }
    wait_job(
        client, client.post("/api/workspaces/" + key + "/tasks", json=command), status="failed"
    )
    assert inventory(source) == before
    command["options"]["overwrite"] = True
    wait_job(client, client.post("/api/workspaces/" + key + "/tasks", json=command))
    current = client.app.state.records[key].workspace
    np.testing.assert_array_equal(np.load(path), np.asarray(current.session.image_array))
    np.testing.assert_array_equal(current.session.original_image_array, values)
    assert not np.array_equal(np.load(path), values)
    after = inventory(source)
    assert after["image.json"] == before["image.json"]
    assert set(after) == set(before)
    denied = client.post(
        "/api/files/text",
        json={
            "root": "home",
            "path": str(sidecar),
            "workspace_id": key,
            "revision": w["revision"],
            "content": "replace metadata",
            "overwrite": True,
        },
    )
    assert denied.status_code == 422
    assert inventory(source) == after


def test_imported_folder_protection_applies_to_other_studies_and_native_saves(client, tmp_path):
    source = tmp_path / "input"
    _write_minimal_bruker_scan(source)
    # Simulate input that happens to have an old export marker and native recipes.
    (source / ".mrax-export").write_text("MRax export\n")
    document = new_workflow(
        "processing",
        [
            {
                "id": "denoise",
                "operation": "process",
                "version": 1,
                "settings": {"method": "gaussian", "axes": ["x", "y"], "parameters": {"sigma": 1}},
            }
        ],
    )
    recipe = source / "existing.mpip"
    save_workflow(document, recipe)
    owned = source / "existing.mrax"
    native = StudyWorkspace("native", "Native", synthetic_t2_session())
    save_workspace_bundle(native, owned)
    before = inventory(source)
    wait_job(
        client,
        client.post(
            "/api/import",
            json={
                "root": "home",
                "path": str(source),
                "options": {"kind": "bruker"},
            },
        ),
    )
    other = client.post("/api/workspaces/synthetic").json()
    key = other["id"]
    response = client.post(
        "/api/workspaces/" + key + "/save-file",
        json={
            "root": "home",
            "path": str(owned),
            "revision": other["revision"],
            "overwrite": True,
        },
    )
    assert response.status_code == 422 and "read-only" in response.text
    library = client.post("/api/workflows", json={"document": document.model_dump()})
    assert library.status_code == 200, library.text
    entry = library.json()
    response = client.post(
        "/api/workflows/" + entry["id"] + "/save-file",
        json={
            "root": "home",
            "path": str(recipe),
            "revision": entry["revision"],
            "overwrite": True,
        },
    )
    assert response.status_code == 422 and "read-only" in response.text
    obj = next(
        row
        for row in client.get("/api/workspaces/" + key + "/inspect").json()["objects"]
        if row["kind"] == "image"
    )
    response = client.post(
        "/api/workspaces/" + key + "/tasks",
        json={
            "revision": other["revision"],
            "operation": "export",
            "options": {
                "kind": "object",
                "object": obj["id"],
                "format": "nifti",
                "root": "home",
                "path": str(source),
                "overwrite": True,
            },
        },
    )
    result = wait_job(client, response)
    assert result["summary"]["saved"]
    after = inventory(source)
    assert all(after[name] == value for name, value in before.items() if name != ".")


def test_native_extensions_do_not_allow_replacing_arbitrary_external_files(client, tmp_path):
    w = client.post("/api/workspaces/synthetic").json()
    path = tmp_path / "not-a-workspace.mrax"
    path.write_bytes(b"external file")
    response = client.post(
        "/api/workspaces/" + w["id"] + "/save-file",
        json={
            "root": "home",
            "path": str(path),
            "revision": w["revision"],
            "overwrite": True,
        },
    )
    assert response.status_code == 422 and path.read_bytes() == b"external file"
    document = new_workflow(
        "processing",
        [
            {
                "id": "denoise",
                "operation": "process",
                "version": 1,
                "settings": {"method": "gaussian"},
            }
        ],
    )
    path = tmp_path / "not-a-workflow.mpip"
    path.write_text("external file")
    with pytest.raises(ValueError):
        save_workflow(document, path, overwrite=True)
    assert path.read_text() == "external file"


def test_text_exports_require_a_new_name_even_for_files_outside_imports(client, tmp_path):
    w = client.post("/api/workspaces/synthetic").json()
    path = tmp_path / "metadata.json"
    path.write_text('{"keep": true}')
    before = inventory(tmp_path)
    response = client.post(
        "/api/files/text",
        json={
            "root": "home",
            "path": str(path),
            "revision": w["revision"],
            "workspace_id": w["id"],
            "content": "overwrite",
            "overwrite": True,
        },
    )
    assert response.status_code == 422
    assert path.read_text() == '{"keep": true}'
    assert inventory(tmp_path) == before
    workspace = client.app.state.records[w["id"]].workspace
    with pytest.raises(ValueError, match="new file name"):
        export_workspace(
            workspace,
            {"kind": "diagnostics", "destination": str(path), "overwrite": True},
            tmp_path / "job",
            lambda *_: None,
        )
    assert path.read_text() == '{"keep": true}'


def test_existing_export_folder_contents_are_retained_when_new_files_are_added(tmp_path):
    destination = tmp_path / "export"
    destination.mkdir()
    (destination / "old").write_bytes(b"old")
    inode = destination.stat().st_ino
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "new").write_bytes(b"new")
    files, conflicts = export_file_plan(prepared, destination, policy=lambda *_: (True, ""))
    assert not conflicts
    publish_export_files(prepared, destination, files)
    assert destination.stat().st_ino == inode
    assert (destination / "old").read_bytes() == b"old"
    assert (destination / "new").read_bytes() == b"new"


def test_explicit_native_bundle_replacement_keeps_optional_recovery_working(tmp_path):
    from mrax.gui.services.persistence import save_workspace_bundle as checkpoint

    workspace = StudyWorkspace("native", "Initial", synthetic_t2_session())
    path = checkpoint(workspace, tmp_path / "recovery.mrax")
    workspace.label = "Updated"
    workspace.session.rename_study("Updated")
    with pytest.raises(ValueError):
        checkpoint(workspace, path)
    checkpoint(workspace, path, overwrite=True)
    assert load_workspace_bundle(path, tmp_path / "restored", key="restored").label == "Updated"


def test_startup_package_folders_are_registered_as_protected_inputs(tmp_path):
    source = tmp_path / "package"
    workspace = StudyWorkspace("native", "Independent", synthetic_t2_session())
    save_workspace_folder(workspace, source)
    (tmp_path / "runtime").mkdir()
    with TestClient(
        create_app(initial_paths=[source], runtime_dir=tmp_path / "runtime"),
        base_url="http://localhost",
    ) as client:
        assert len(client.get("/api/workspaces").json()) == 1
        with pytest.raises(ValueError, match="read-only"):
            client.app.state.folders.sources.check(source, directory=True)
