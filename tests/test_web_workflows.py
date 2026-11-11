"""Synthetic browser contracts and numerical integration."""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.session import GuiSession
from mrax.web.app import create_app
from mrax.web.job_runtime import _workflow_worker


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as connection:
        connection.headers["x-mrax-token"] = connection.get("/api/session").json()["token"]
        yield connection


def new_workspace(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def add_roi(client, workspace, **overrides):
    command = {
        "name": "ROI 1",
        "bounds": [4, 3, 8, 6],
        "revision": workspace["revision"],
        "view_revision": workspace["view_revision"],
    }
    command.update(overrides)
    return client.post(f"/api/workspaces/{workspace['id']}/rois", json=command)


def wait_job(client, key):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{key}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"queued", "running", "cancelling"}:
            return job
        time.sleep(0.05)
    pytest.fail("The synthetic fit worker did not finish")


def test_native_roi_and_binary_slice_are_numerically_correct(client):
    workspace = new_workspace(client)
    assert workspace["experiment_catalog"][0]["exp_type"] == "t2_se"
    response = client.get(f"/api/workspaces/{workspace['id']}/slice")
    image = np.frombuffer(response.content, dtype="<f4").reshape((24, 32))
    reference = synthetic_t2_session().frame_image()
    np.testing.assert_array_equal(image, reference)
    assert response.headers["x-dtype"] == "float32-le"
    assert response.headers["x-view-revision"] == "0"
    response = add_roi(client, workspace)
    assert response.status_code == 200
    loaded = response.json()
    expected = np.zeros((24, 32), dtype=int)
    expected[3:9, 4:12] = 1
    np.testing.assert_array_equal(loaded["rois"][0]["mask"], expected)
    assert loaded["rois"][0]["pixels"] == 48
    assert loaded["timecourse"]["x_unit"] == "ms"
    observed = loaded["timecourse"]["observed"]
    assert observed[4] / observed[0] == pytest.approx(np.exp(-70 / 80), rel=1e-6)


def test_revision_checks_prevent_stale_drawing_and_cross_workspace_edits(client):
    first, second = new_workspace(client), new_workspace(client)
    path = f"/api/workspaces/{first['id']}/view"
    assert client.patch(path, json={"revision": 0, "frame": 1, "z": 1}).status_code == 200
    assert add_roi(client, first).status_code == 409
    current = client.get(f"/api/workspaces/{first['id']}").json()
    assert add_roi(client, current).status_code == 200
    assert add_roi(client, current, name="Other ROI").status_code == 409
    assert client.get(f"/api/workspaces/{second['id']}").json()["rois"] == []
    assert client.patch(path, json={"revision": 1, "frame": 99, "z": 0}).status_code == 422


def test_real_spawned_fit_matches_known_t2_and_round_trips(client):
    workspace = new_workspace(client)
    workspace = add_roi(client, workspace).json()
    path = f"/api/workspaces/{workspace['id']}"
    response = client.post(
        path + "/preview-fit",
        json={"revision": workspace["revision"], "roi_name": "ROI 1", "steps": 600},
    )
    assert response.status_code == 200, response.text
    # Serving requests while JAX compiles/fits is part of this contract.
    assert client.get("/api/health").json()["status"] == "ok"
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert job["result"]["parameters"]["t2"] == pytest.approx(80, abs=0.05)
    assert job["result"]["relative_rmse"] < 1e-4
    assert job["result"]["x_unit"] == "ms"
    saved = client.post(path + "/save")
    assert saved.status_code == 200
    with zipfile.ZipFile(io.BytesIO(saved.content)) as archive:
        assert "mrax_gui_state.json" in archive.namelist()
        payload = json.loads(archive.read("mrax_gui_state.json"))
        assert payload["format"] == "mrax.gui.state" and payload["version"] == 3
        assert "theme" not in payload
    response = client.post("/api/workspaces/upload", content=saved.content)
    assert response.status_code == 200, response.text
    restored = response.json()
    assert restored["rois"] == workspace["rois"]
    assert restored["spacing"] == workspace["spacing"]
    assert restored["preview"]["parameters"]["t2"] == job["result"]["parameters"]["t2"]
    assert restored["id"] != workspace["id"]
    assert restored["label"] == workspace["label"]


def test_cancelled_worker_does_not_start_an_operation(tmp_path):
    def cancelled_progress(*_args):
        raise InterruptedError("Cancelled")

    assert _workflow_worker(None, tmp_path, "preview", {}, cancelled_progress) == {
        "cancelled": True
    }
    assert not list(tmp_path.iterdir())


def test_origins_tokens_hosts_and_archives_are_checked(client):
    assert (
        client.post("/api/workspaces/synthetic", headers={"x-mrax-token": "wrong"}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/workspaces/synthetic", headers={"origin": "https://example.invalid"}
        ).status_code
        == 403
    )
    assert client.get("/api/health", headers={"host": "example.invalid"}).status_code == 400
    assert client.post("/api/workspaces/upload", content=b"invalid").status_code == 422
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("../outside.txt", "untrusted")
    assert client.post("/api/workspaces/upload", content=archive.getvalue()).status_code == 422
    assert client.get("/api/workspaces").json() == []
    assert client.get("/api/workspaces/missing").status_code == 404
    assert client.get("/api/jobs/missing").status_code == 404


def test_edited_observables_rebuild_the_study_and_survive_save_restore(client, monkeypatch):
    workspace = new_workspace(client)
    key = workspace["id"]
    original = client.get(f"/api/workspaces/{key}/inspect").json()["metadata"]
    options = json.loads(json.dumps(original))
    options["observables"]["TE"]["values"][0] = 12.5
    options["metadata"]["note"] = "Edited synthetic coordinate"
    image = client.get(f"/api/workspaces/{key}/slice").content
    response = client.post(
        f"/api/workspaces/{key}/actions",
        json={"revision": workspace["revision"], "action": "metadata.update", "options": options},
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    inspected = client.get(f"/api/workspaces/{key}/inspect").json()["metadata"]
    assert inspected["observables"]["TE"]["values"][0] == pytest.approx(12.5)
    assert inspected["observables"]["TE"]["unit"] == original["observables"]["TE"]["unit"]
    assert (
        inspected["observables"]["TE"]["values"][1:] == original["observables"]["TE"]["values"][1:]
    )
    assert client.get(f"/api/workspaces/{key}/slice").content == image

    invalid = json.loads(json.dumps(options))
    invalid["observables"]["TE"]["values"] = [1, 2]
    rejected = client.post(
        f"/api/workspaces/{key}/actions",
        json={"revision": updated["revision"], "action": "metadata.update", "options": invalid},
    )
    assert rejected.status_code == 422
    assert client.get(f"/api/workspaces/{key}/inspect").json()["metadata"] == inspected
    assert client.get(f"/api/workspaces/{key}").json()["revision"] == updated["revision"]

    summary = client.get(f"/api/workspaces/{key}/inspect").json()["summary"]
    invalid["observables"] = options["observables"]
    invalid["exp_type"] = "dwi"
    fit_study = GuiSession.fit_study

    def reject_candidate(session):
        if session.study is None:
            raise ValueError("Invalid synthetic experiment metadata")
        return fit_study(session)

    with monkeypatch.context() as patch:
        patch.setattr(GuiSession, "fit_study", reject_candidate)
        rejected = client.post(
            f"/api/workspaces/{key}/actions",
            json={"revision": updated["revision"], "action": "metadata.update", "options": invalid},
        )
    assert rejected.status_code == 422
    after = client.get(f"/api/workspaces/{key}/inspect").json()
    assert after["metadata"] == inspected
    assert after["summary"] == summary

    saved = client.post(f"/api/workspaces/{key}/save")
    assert saved.status_code == 200
    restored = client.post("/api/workspaces/upload", content=saved.content)
    assert restored.status_code == 200, restored.text
    restored_metadata = client.get(f"/api/workspaces/{restored.json()['id']}/inspect").json()[
        "metadata"
    ]
    assert restored_metadata["observables"] == inspected["observables"]
    assert restored_metadata["metadata"]["note"] == "Edited synthetic coordinate"
