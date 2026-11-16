"""Atomic navigation and bounded preview work on independent synthetic data."""

import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.data_struct import Quantity
from mrax.gui.services.analysis_workflows import run_operation
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
    response = client.post(
        f"/api/workspaces/{w['id']}/rois",
        json={
            "revision": w["revision"],
            "view_revision": w["view_revision"],
            "name": "Region",
            "bounds": [0, 0, 4, 4],
        },
    )
    assert response.status_code == 200, response.text
    w = response.json()
    assert w["timecourse"]["roi_name"] == "Region"
    np.testing.assert_allclose(w["timecourse"]["observed"], data[:, :4, :4, 1].mean(axis=(1, 2)))
    response = client.patch(
        f"/api/workspaces/{w['id']}/view",
        json={
            "revision": w["revision"],
            "plot_whole_image": True,
        },
    )
    assert response.status_code == 200, response.text
    whole = response.json()
    assert whole["selected_roi"] == "Region"
    assert whole["revision"] == w["revision"]
    assert whole["timecourse"]["roi_name"] is None
    assert whole["preview"] is None
    np.testing.assert_allclose(whole["timecourse"]["observed"], data[..., 1].mean(axis=(1, 2)))
    cached = record.info_cache["curve"]
    header, _ = decode(navigate(client, whole, {"frame": 3}))
    assert record.info_cache["curve"] is cached
    assert header["workspace"]["timecourse"]["roi_name"] is None
    response = client.patch(
        f"/api/workspaces/{w['id']}/view",
        json={
            "revision": w["revision"],
            "selected_roi": "Region",
        },
    )
    assert response.status_code == 200, response.text
    w = response.json()
    assert w["display"]["plot_whole_image"] is False
    assert w["timecourse"]["roi_name"] == "Region"
    np.testing.assert_allclose(w["timecourse"]["observed"], data[:, :4, :4, 1].mean(axis=(1, 2)))
    response = client.post(
        f"/api/workspaces/{w['id']}/actions",
        json={
            "revision": w["revision"],
            "view_revision": w["view_revision"],
            "action": "roi.remove",
            "options": {"name": "Region"},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["timecourse"]["roi_name"] is None
    np.testing.assert_array_equal(record.workspace.session.original_image_array, data)

    echoes = client.post("/api/workspaces/new").json()
    record = client.app.state.records[echoes["id"]]
    values = np.arange(8 * 6 * 5, dtype=np.float32).reshape(8, 6, 5)
    record.workspace.session.add_experiment(
        "Echo series", values, dims={"te": 8, "x": 6, "y": 5}, model_name="t2_se",
        observables={"TE": Quantity(np.arange(1, 9) * 10, "ms")},
    )
    run_operation(record.workspace, "process", {
        "method": "gaussian", "input": "image", "axes": ["te", "x", "y"],
        "parameters": {"sigma": 0.5},
    }, lambda *_: None)
    record.revision += 1
    record.info_cache.clear()
    w = client.get(f"/api/workspaces/{echoes['id']}").json()
    processed = record.workspace.session.image_array.copy()
    for indices in ({"te": 3, "frame": 0}, {"frame": 0, "te": 3}, {"te": 3}):
        header, planes = decode(navigate(client, w, indices, commit=False))
        assert header["workspace"]["frame"] == 3
        assert header["workspace"]["indices"]["te"] == 3
        np.testing.assert_array_equal(planes["image"], processed[3])
    header, _ = decode(navigate(client, w, {"te": 6, "frame": 0}))
    w = header["workspace"]
    assert w["indices"]["te"] == 6
    header, planes = decode(navigate(client, w, {"te": 2, "frame": 6}))
    assert header["workspace"]["frame"] == 2
    np.testing.assert_array_equal(planes["image"], processed[2])
    np.testing.assert_array_equal(record.workspace.session.image_array, processed)

    saturation = client.post("/api/workspaces/new").json()
    record = client.app.state.records[saturation["id"]]
    offsets = np.linspace(-3, 3, 7, dtype=np.float32)
    for repetitions, powers, unit, per_frame in ((1, 1, "ppm", False), (2, 2, "Hz", True)):
        dims = {"t": repetitions, "b1": powers, "offset": len(offsets), "x": 4, "y": 3}
        values = np.arange(np.prod(list(dims.values())), dtype=np.float32).reshape(tuple(dims.values()))
        frame_count = repetitions * powers * len(offsets)
        coordinates = np.broadcast_to(offsets, (repetitions, powers, len(offsets))).copy()
        if per_frame:
            coordinates += np.arange(repetitions)[:, None, None] * 0.1
            coordinates += np.arange(powers)[None, :, None] * 0.01
        record.workspace.session.add_experiment(
            f"Saturation sweep {repetitions}", values, dims=dims, exp_type="sat_trans",
            observables={
                "TR": Quantity(np.linspace(300, 360, frame_count), "ms"),
                "freq_offsets": Quantity(coordinates.reshape(-1) if per_frame else offsets, unit),
                "sat_b1": Quantity(1 if powers == 1 else np.arange(1, powers + 1), "uT"),
                "sat_time": Quantity(2, "s"),
            },
        )
        record.revision += 1
        record.info_cache.clear()
        response = client.patch(
            f"/api/workspaces/{saturation['id']}/view",
            json={"revision": record.revision, "plot_axis": "offset",
                  "indices": {"t": repetitions - 1, "b1": powers - 1, "offset": 4}},
        )
        assert response.status_code == 200, response.text
        curve = response.json()["timecourse"]
        assert curve["x_name"] == f"freq_offsets ({unit})"
        assert curve["x_unit"] == unit
        assert curve["dimension"] == "offset"
        np.testing.assert_allclose(curve["x"], coordinates[-1, -1])
        np.testing.assert_allclose(curve["observed"], values[-1, -1].mean(axis=(1, 2)))
        curve = client.patch(
            f"/api/workspaces/{saturation['id']}/view",
            json={"revision": record.revision, "plot_axis": "b1"},
        ).json()["timecourse"]
        assert curve["x_name"] == "sat_b1 (uT)" and curve["x_unit"] == "uT"
        np.testing.assert_array_equal(curve["x"], np.arange(1, powers + 1))

    timed = client.post("/api/workspaces/new").json()
    record = client.app.state.records[timed["id"]]
    values = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
    for index, (observables, expected) in enumerate(
        (
            ({"scan_time": Quantity(4500, "ms")}, np.arange(6) * 0.75),
            ({"t": Quantity(np.arange(6) * 250, "ms")}, np.arange(6) * 0.25),
        )
    ):
        record.workspace.session.add_experiment(
            f"Timed acquisition {index}",
            values,
            dims={"t": 6, "x": 4, "y": 3},
            exp_type="dce",
            observables=observables,
        )
        record.revision += 1
        record.info_cache.clear()
        response = client.get(f"/api/workspaces/{timed['id']}")
        assert response.status_code == 200, response.text
        curve = response.json()["timecourse"]
        assert curve["x_name"] == "Elapsed time" and curve["x_unit"] == "s"
        np.testing.assert_allclose(curve["x"], expected)
        np.testing.assert_allclose(curve["observed"], values.mean(axis=(1, 2)))

    spectrum = client.post("/api/workspaces/new").json()
    record = client.app.state.records[spectrum["id"]]
    frequency = np.linspace(-200, 200, 65)
    values = 2 + 10 * np.exp(-(((frequency - 35) / 25) ** 2))
    record.workspace.session.add_experiment(
        "Spectrum",
        np.stack([values, 2 * values])[..., None],
        dims={"frame": 2, "x": 65, "y": 1},
        exp_type="spectroscopy",
        non_imaging=True,
        data_role="spectrum",
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

    complex_values = (values - 5) + 1j * frequency / 50
    record.workspace.session.add_experiment(
        "Complex spectrum",
        complex_values[None, :],
        dims={"frame": 1, "x": 65},
        exp_type="spectroscopy",
        non_imaging=True,
        observables={"frequency": Quantity(frequency, "Hz")},
    )
    record.revision += 1
    record.info_cache.clear()
    source = np.asarray(record.workspace.session.original_image_array).copy()
    for component, expected in (
        ("real", complex_values.real),
        ("imag", complex_values.imag),
        ("phase", np.angle(complex_values)),
        ("magnitude", np.abs(complex_values)),
        ("auto", np.abs(complex_values)),
    ):
        response = client.patch(
            f"/api/workspaces/{spectrum['id']}/view",
            json={
                "revision": record.revision,
                "component": component,
            },
        )
        assert response.status_code == 200, response.text
        np.testing.assert_allclose(response.json()["timecourse"]["observed"], expected)
    np.testing.assert_array_equal(record.workspace.session.original_image_array, source)

    record.workspace.session.add_experiment(
        "Uncalibrated spectrum",
        np.stack([values, 2 * values]),
        dims={"repeat": 2, "frequency": 65},
        exp_type="spectroscopy",
        non_imaging=True,
    )
    record.revision += 1
    record.info_cache.clear()
    spectrum = client.get(f"/api/workspaces/{spectrum['id']}").json()
    assert spectrum["timecourse"]["dimension"] == "frequency"
    assert spectrum["timecourse"]["x_name"] == "frequency index"
    np.testing.assert_array_equal(spectrum["timecourse"]["x"], np.arange(65))
    np.testing.assert_allclose(spectrum["timecourse"]["observed"], values)
