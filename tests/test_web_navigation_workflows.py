"""Atomic navigation and bounded preview work on independent synthetic data."""

import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.data_struct import Quantity
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


def test_signal_profiles_use_whole_images_and_declared_spectral_coordinates(client):
    w = client.post("/api/workspaces/synthetic").json()
    record = client.app.state.records[w["id"]]
    data = np.asarray(record.workspace.session.image_array).copy()
    assert w["timecourse"]["roi_name"] is None
    np.testing.assert_allclose(w["timecourse"]["observed"], data[..., 0].mean(axis=(1, 2)))
    cached = record.info_cache["curve"]
    header, _ = decode(navigate(client, w, {"frame": 2}))
    w = header["workspace"]
    assert record.info_cache["curve"] is cached
    header, _ = decode(navigate(client, w, {"z": 1}))
    w = header["workspace"]
    np.testing.assert_allclose(w["timecourse"]["observed"], data[..., 1].mean(axis=(1, 2)))
    response = client.post(f"/api/workspaces/{w['id']}/rois", json={
        "revision": w["revision"], "view_revision": w["view_revision"],
        "name": "Region", "bounds": [0, 0, 4, 4],
    })
    assert response.status_code == 200, response.text
    w = response.json()
    assert w["timecourse"]["roi_name"] == "Region"
    np.testing.assert_allclose(w["timecourse"]["observed"], data[:, :4, :4, 1].mean(axis=(1, 2)))
    response = client.post(f"/api/workspaces/{w['id']}/actions", json={
        "revision": w["revision"], "view_revision": w["view_revision"],
        "action": "roi.remove", "options": {"name": "Region"},
    })
    assert response.status_code == 200, response.text
    assert response.json()["timecourse"]["roi_name"] is None
    np.testing.assert_array_equal(record.workspace.session.original_image_array, data)

    spectrum = client.post("/api/workspaces/new").json()
    record = client.app.state.records[spectrum["id"]]
    frequency = np.linspace(-200, 200, 65)
    values = 2 + 10 * np.exp(-((frequency - 35) / 25) ** 2)
    record.workspace.session.add_experiment(
        "Spectrum", np.stack([values, 2 * values])[..., None],
        dims={"frame": 2, "x": 65, "y": 1}, exp_type="spectroscopy",
        non_imaging=True, data_role="spectrum",
        observables={"frequency": Quantity(frequency, "Hz")},
    )
    record.revision += 1
    record.info_cache.clear()
    response = client.get(f"/api/workspaces/{spectrum['id']}")
    assert response.status_code == 200, response.text
    spectrum = response.json()
    assert spectrum["non_imaging"] and spectrum["data_role"] == "spectrum"
    assert spectrum["timecourse"]["dimension"] == "x"
    assert spectrum["timecourse"]["x_name"] == "frequency (Hz)"
    np.testing.assert_allclose(spectrum["timecourse"]["x"], frequency)
    np.testing.assert_allclose(spectrum["timecourse"]["observed"], values)
    header, _ = decode(navigate(client, spectrum, {"frame": 1}))
    np.testing.assert_allclose(header["workspace"]["timecourse"]["observed"], 2 * values)

    record.workspace.session.add_experiment(
        "Uncalibrated spectrum", np.stack([values, 2 * values]),
        dims={"repeat": 2, "frequency": 65}, exp_type="spectroscopy", non_imaging=True,
    )
    record.revision += 1
    record.info_cache.clear()
    spectrum = client.get(f"/api/workspaces/{spectrum['id']}").json()
    assert spectrum["timecourse"]["dimension"] == "frequency"
    assert spectrum["timecourse"]["x_name"] == "frequency index"
    np.testing.assert_array_equal(spectrum["timecourse"]["x"], np.arange(65))
    np.testing.assert_allclose(spectrum["timecourse"]["observed"], values)
