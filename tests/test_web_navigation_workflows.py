"""Atomic navigation and bounded preview work on independent synthetic data."""

import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def decode(response):
    assert response.status_code == 200, response.text[:300] if response.status_code != 200 else ""
    body = response.content
    version, size = struct.unpack_from("<II", body)
    assert version == 1
    header = json.loads(body[8 : 8 + size])
    offset = 8 + (size + 7) // 8 * 8
    planes = {}
    for plane in header["planes"]:
        dtype = np.dtype("<f8" if plane["dtype"] == "float64-le" else "<f4")
        array = np.frombuffer(body, dtype=dtype, offset=offset, count=plane["length"])
        planes[plane["id"]] = array.reshape(plane["shape"])
        offset += (array.nbytes + 7) // 8 * 8
    assert offset == len(body)
    return header, planes


def navigate(client, workspace, indices, *, commit=True):
    return client.post(
        f"/api/workspaces/{workspace['id']}/navigate",
        json={
            "revision": workspace["revision"],
            "view_revision": workspace["view_revision"],
            "indices": indices,
            "commit": commit,
        },
    )


def test_navigation_delivers_a_consistent_cursor_and_plane_without_cloning_or_saving(
    client, monkeypatch
):
    w = client.post("/api/workspaces/synthetic").json()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Navigation must not clone or save the study")

    monkeypatch.setattr("mrax.web.app.clone_workspace", forbidden)
    monkeypatch.setattr(client.app.state, "checkpoint", forbidden)
    response = navigate(client, w, {"frame": 2, "z": 1})
    metadata, planes = decode(response)
    changed = metadata["workspace"]
    assert changed["frame"] == 2 and changed["z"] == 1
    assert changed["view_revision"] == w["view_revision"] + 1
    assert changed["revision"] == w["revision"]
    assert response.headers["cache-control"] == "no-store"
    recorded = client.get(f"/api/workspaces/{w['id']}").json()
    assert recorded["indices"] == changed["indices"]
    pixels = client.get(f"/api/workspaces/{w['id']}/slice")
    dtype = "<f8" if pixels.headers["x-dtype"] == "float64-le" else "<f4"
    np.testing.assert_array_equal(
        planes["image"].ravel(), np.frombuffer(pixels.content, dtype=dtype)
    )
