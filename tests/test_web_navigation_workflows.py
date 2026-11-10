"""Atomic navigation and bounded preview work on independent synthetic data."""

import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.analysis import ROI, broadcast_mask
from mrax.data_struct import Experiment
from mrax.gui.plotting import image_profile
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.roi_tools import roi_runs
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


def test_prefetch_does_not_change_the_workspace_revision_dirty_flag_or_current_plane(client):
    w = client.post("/api/workspaces/synthetic").json()
    record = client.app.state.records[w["id"]]
    workspace = record.workspace
    indices = dict(workspace.session.active_indices)
    before = workspace.session.frame_image().copy()
    dirty = workspace.dirty
    metadata, _ = decode(navigate(client, w, {"frame": 3, "z": 2}, commit=False))
    assert metadata["workspace"]["frame"] == 3
    assert metadata["workspace"]["z"] == 2
    assert record.workspace is workspace
    assert record.view_revision == w["view_revision"]
    assert record.workspace.dirty == dirty
    assert record.workspace.session.active_indices == indices
    np.testing.assert_array_equal(record.workspace.session.frame_image(), before)


def test_confirming_a_cached_plane_does_not_retransmit_image_data(client):
    w = client.post("/api/workspaces/synthetic").json()
    response = client.post(
        f"/api/workspaces/{w['id']}/navigate",
        json={
            "revision": w["revision"],
            "view_revision": w["view_revision"],
            "indices": {"frame": 1},
            "include_images": False,
        },
    )
    metadata, planes = decode(response)
    assert metadata["images_included"] is False
    assert planes == {}
    assert metadata["workspace"]["frame"] == 1


@pytest.mark.parametrize("indices", [{"frame": -1}, {"frame": 9999}, {"z": 999}, {"missing": 1}])
def test_invalid_navigation_cannot_change_the_active_cursor(client, indices):
    w = client.post("/api/workspaces/synthetic").json()
    assert navigate(client, w, indices).status_code in {400, 422}
    current = client.get(f"/api/workspaces/{w['id']}").json()
    assert current["view_revision"] == w["view_revision"]
    assert current["indices"] == w["indices"]


def test_stale_navigation_and_prefetch_are_rejected(client):
    w = client.post("/api/workspaces/synthetic").json()
    decode(navigate(client, w, {"frame": 1}))
    assert navigate(client, w, {"frame": 2}).status_code == 409
    assert navigate(client, w, {"frame": 2}, commit=False).status_code == 409


def test_binary_navigation_preserves_mixed_precision_and_odd_plane_padding(client):
    from mrax.gui.services.workspace import StudyWorkspace
    from mrax.gui.session import GuiSession
    from mrax.web.app import Record

    session = GuiSession()
    session.create_empty_study("Independent precision study", field_strength_mhz=100)
    first = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    first[1, 0, 0] = np.nan
    second = np.full((2, 3, 5), np.pi, dtype=np.float64)
    second[1, 0, :3] = [np.inf, -0.0, 1e100]
    dims = {"frame": 2, "x": 3, "y": 5}
    session.add_experiment("A", first, dims=dims)
    session.current_experiment["mr_image"] = first
    session.add_experiment("B", np.ones_like(first), dims=dims)
    session.current_experiment["mr_image"] = second
    session.select_experiment("A")
    workspace = StudyWorkspace("precision", session.study_name, session)
    client.app.state.records[workspace.key] = Record(workspace)
    w = client.get("/api/workspaces/precision").json()
    layer = {
        "id": "image",
        "experiment": "B",
        "kind": "image",
        "component": "auto",
        "colormap": "gray",
        "opacity": 0.5,
        "visible": True,
    }
    response = client.patch(
        "/api/workspaces/precision/view", json={"revision": 0, "scene": {"layers": [layer]}}
    )
    assert response.status_code == 200, response.text
    w = response.json()
    metadata, planes = decode(navigate(client, w, {"frame": 1}))
    assert metadata["workspace"]["shape"] == [3, 5]
    assert planes["image"].dtype == np.dtype("<f4")
    assert planes["layer:image"].dtype == np.dtype("<f8")
    np.testing.assert_array_equal(planes["image"], first[1])
    np.testing.assert_array_equal(planes["layer:image"], second[1])


def test_raw_kspace_navigation_handles_all_named_controls(client):
    w = client.post("/api/workspaces/synthetic?kind=multimodal").json()
    response = client.patch(
        f"/api/workspaces/{w['id']}/view",
        json={"revision": w["revision"], "experiment": "Complex coils", "view": "K-space"},
    )
    assert response.status_code == 200, response.text
    w = response.json()
    for control in w["dimension_controls"]:
        if control["size"] > 1:
            metadata, planes = decode(navigate(client, w, {control["key"]: 1}))
            w = metadata["workspace"]
            assert w["indices"][control["key"]] == 1
            assert list(planes["image"].shape) == w["shape"]


def test_comparison_navigation_and_prefetch_keep_independent_image_state(client, monkeypatch):
    w = client.post("/api/workspaces/synthetic").json()
    settings = {**w["scene"]["comparison"], "enabled": True}
    w = client.patch(
        f"/api/workspaces/{w['id']}/view",
        json={"revision": w["revision"], "scene": {"comparison": settings}},
    ).json()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Comparison navigation must reuse its catalog")

    monkeypatch.setattr("mrax.gui.services.scene.experiment_catalog", forbidden)
    monkeypatch.setattr("mrax.web.app.clone_workspace", forbidden)
    preview, _ = decode(navigate(client, w, {"frame": 2, "z": 1}, commit=False))
    assert preview["workspace"]["scene"]["comparison"]["frame"] == 2
    record = client.app.state.records[w["id"]]
    assert record.workspace.session.active_frame == w["frame"]
    assert (
        record.workspace.display_settings[w["experiment"]]["scene"]["comparison"]["frame"]
        == settings["frame"]
    )
    result, planes = decode(navigate(client, w, {"frame": 3, "z": 2}))
    assert result["workspace"]["frame"] == w["frame"]
    assert result["workspace"]["scene"]["comparison"]["frame"] == 3
    arrays, metadata = render_comparison(record.workspace, catalog=record.info_cache["catalog"])
    assert result["comparison"] == metadata
    for name, array in zip(("left", "right"), arrays):
        np.testing.assert_array_equal(planes[name], array)


@pytest.mark.parametrize(
    "dimension,plane", [("frame", ("x", "y")), ("frame", ("x", "z")), ("z", ("x", "y"))]
)
@pytest.mark.parametrize("mask_dims", [("x", "y", "z"), ("frame", "x", "y", "z"), ("y", "z"), None])
def test_curve_masks_are_selected_before_broadcast_without_changing_signal_values(
    monkeypatch, dimension, plane, mask_dims
):
    dims = {"frame": 3, "x": 4, "y": 5, "z": 2}
    experiment = Experiment(dims=dims)
    data = np.random.default_rng(5).normal(size=tuple(dims.values()))
    data[0, 1, 2, 1] = np.nan
    shape = tuple(dims[name] for name in (mask_dims or ("y", "z")))
    mask = np.random.default_rng(6).integers(0, 3, size=shape)
    roi = ROI("Independent mask", mask, dims=mask_dims)
    indices = {"frame": 1, "x": 2, "y": 3, "z": 1}
    selection = tuple(
        slice(None) if name == dimension or name in plane else indices[name] for name in dims
    )
    retained = [name for name in dims if name == dimension or name in plane]
    full_mask = broadcast_mask(mask, data.shape, mask_dims=mask_dims, target_dims=tuple(dims))
    expected = np.where(full_mask[selection], data[selection], np.nan)
    axes = tuple(index for index, name in enumerate(retained) if name != dimension)
    finite = np.isfinite(expected)
    count = finite.sum(axis=axes)
    expected = np.divide(
        np.where(finite, expected, 0).sum(axis=axes),
        count,
        out=np.full(count.shape, np.nan),
        where=count > 0,
    )
    shapes = []

    def limited(mask, target_shape, **kwargs):
        shapes.append(target_shape)
        assert len(target_shape) < len(dims)
        return broadcast_mask(mask, target_shape, **kwargs)

    monkeypatch.setattr("mrax.gui.plotting.broadcast_mask", limited)
    actual = image_profile(data, experiment, dimension, indices, roi=roi, plane_dims=plane)
    np.testing.assert_allclose(actual, np.atleast_1d(expected).reshape(-1), equal_nan=True)
    assert shapes


@pytest.mark.parametrize("shape", [(0, 0), (1, 1), (1, 7), (7, 1), (21, 17)])
def test_vectorized_roi_runs_preserve_labels_and_exact_voxel_positions(shape):
    mask = np.random.default_rng(8).integers(0, 4, size=shape, dtype=np.int64)
    result = roi_runs(mask)
    rebuilt = np.zeros(shape, dtype=np.int64)
    for y, x, length, label in result:
        rebuilt[y, x : x + length] = label
    np.testing.assert_array_equal(rebuilt, mask)
    assert result == sorted(result, key=lambda row: (row[0], row[1]))


@pytest.mark.parametrize("order", [("z", "x", "y"), ("y", "z", "x")])
def test_profile_mask_dimension_order_does_not_change_the_signal(order):
    dims = {"frame": 3, "x": 4, "y": 5, "z": 2}
    experiment = Experiment(dims=dims)
    data = np.random.default_rng(11).normal(size=tuple(dims.values()))
    mask = np.random.default_rng(12).integers(0, 3, size=(4, 5, 2))
    indices = {"frame": 1, "z": 1}
    expected = image_profile(
        data, experiment, "frame", indices, roi=ROI("A", mask, dims=("x", "y", "z"))
    )
    shuffled = mask.transpose(tuple(("x", "y", "z").index(name) for name in order))
    actual = image_profile(data, experiment, "frame", indices, roi=ROI("B", shuffled, dims=order))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
