"""Performance invariants and data safety on independently synthetic studies."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def new(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, response):
    assert response.status_code == 200, response.text
    for _ in range(600):
        job = client.get("/api/jobs/" + response.json()["id"]).json()
        if job["status"] not in {"queued", "running", "cancelling"}:
            return job
        time.sleep(0.02)
    pytest.fail("Operation did not finish")


def test_processing_never_serializes_a_workspace_and_releases_finished_futures(client, monkeypatch):
    workspace = new(client)
    original = client.app.state.records[workspace["id"]].workspace
    before = np.asarray(original.session.image_array).copy()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Routine operations must not save or compress a workspace")

    monkeypatch.setattr("mrax.web.app.save_workspace_bundle", forbidden)
    monkeypatch.setattr("mrax.gui.services.workspace_io.save_workspace_bundle", forbidden)
    job = finish(
        client,
        client.post(
            "/api/workspaces/" + workspace["id"] + "/tasks",
            json={
                "revision": workspace["revision"],
                "operation": "process",
                "options": {"method": "gaussian", "parameters": {"sigma": 1.0}, "name": "Filtered"},
            },
        ),
    )
    assert job["status"] == "completed", job
    assert not list(client.app.state.root.rglob("*.mrax"))
    assert all("future" not in row for row in client.app.state.jobs.items.values())
    np.testing.assert_array_equal(original.session.image_array, before)
