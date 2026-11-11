"""Ordinary filesystem workflows, using only independently synthetic data."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def study(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


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
