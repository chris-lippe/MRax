"""Ordinary filesystem workflows, using only independently synthetic data."""

import pytest
from fastapi.testclient import TestClient

from mrax.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


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
