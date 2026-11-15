"""Source immutability and explicit output contracts, with synthetic files only."""

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.gui.services.analysis_workflows import import_source, run_operation, update_metadata
from mrax.gui.services.source_protection import check_workspace_output
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import (
    load_workspace_bundle,
    save_workspace_bundle,
)
from mrax.gui.session import GuiSession
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
    path = tmp_path / "not-a-workflow.mflow"
    path.write_text("external file")
    with pytest.raises(ValueError):
        save_workflow(document, path, overwrite=True)
    assert path.read_text() == "external file"
