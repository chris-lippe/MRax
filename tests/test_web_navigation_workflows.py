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

    for kind in ("t1_ir", "wasabi"):
        fitted = client.post("/api/workspaces/new").json()
        record = client.app.state.records[fitted["id"]]
        session = record.workspace.session
        if kind == "t1_ir":
            dims = {"slice_package": 2, "ti": 4, "fair": 2, "x": 2, "y": 3, "z": 1}
            coordinates = np.asarray([50, 200, 800, 2500], dtype=np.float32)
            signal = np.abs(1e6 - 2e6 * np.exp(-coordinates / 1400))
            values = np.broadcast_to(signal[None, :, None, None, None, None], tuple(dims.values())).copy()
            values *= np.asarray([1, 1.1])[:, None, None, None, None, None]
            observations = {
                "TR": Quantity(np.linspace(1000, 1100, 16), "ms"),
                "TI": Quantity(coordinates, "ms"), "fair_label": Quantity(np.asarray([0, 1]), "1"),
            }
            exp_type, axes = "fair_asl", ("ti", "fair", "slice_package")
        else:
            dims = {"t": 60, "b1": 1, "offset": 7, "x": 2, "y": 3, "z": 1}
            coordinates = np.asarray([0, -3, 3, -1, 1, -2, 2], dtype=np.float32)
            values = np.broadcast_to((1 - 0.7 * np.exp(-coordinates ** 2))[None, None, :, None, None, None], tuple(dims.values())).copy()
            values *= (1 + 0.05 * np.sin(np.arange(60)))[:, None, None, None, None, None]
            observations = {
                "TR": Quantity(np.linspace(1700, 1800, 7), "ms"),
                "freq_offsets": Quantity(coordinates, "ppm"), "sat_time": Quantity(5, "ms"),
                "sat_b1": Quantity(1, "uT"), "b0_map": Quantity(np.zeros((2, 3, 1)), "ppm"),
            }
            session.add_experiment("Reference", np.ones((2, 3, 1)), dims={"x": 2, "y": 3, "z": 1},
                                   exp_type="sat_trans", observables={"sat_b1": Quantity(0, "uT")})
            exp_type, axes = "sat_trans", ("offset", "b1", "t")
        session.add_experiment("Sweep", values, dims=dims, exp_type=exp_type, model_name=kind,
                               observables=observations)
        session.study.study_globals["field_strength_mhz"] = Quantity(400, "MHz")
        session.data["study_globals"]["field_strength_mhz"] = Quantity(400, "MHz")
        if kind == "wasabi":
            session.fit_settings.regularizer_configs = {"b1": [
                {"type": "smoothness", "weight": 0.01, "axes": [0]},
                {"type": "tv", "weight": 0.001, "axes": [1, 2]},
            ]}
            session.fit_settings.parameter_constant_dims = {"c": ("t",)}
            session.fit_settings.data_normalization = "full"
        run_operation(record.workspace, "preview", {
            "steps": 2, "use_roi": False, "excluded_frames": [11] if kind == "t1_ir" else [],
        }, lambda *_: None)
        record.revision += 1
        record.info_cache.clear()
        path = f"/api/workspaces/{fitted['id']}"
        for axis in axes:
            response = client.patch(path + "/view", json={"revision": record.revision, "plot_axis": axis})
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["preview"] is not None
            assert result["preview"]["x"] == result["timecourse"]["x"]
            assert result["preview"]["x_name"] == result["timecourse"]["x_name"]
            assert len(result["preview"]["predicted"]) == dims[axis]
        if kind == "t1_ir":
            result = client.patch(path + "/view", json={
                "revision": record.revision, "plot_axis": "ti", "indices": {"slice_package": 1, "fair": 1},
            }).json()
            np.testing.assert_allclose(result["preview"]["observed"], values[1, :, 1].mean(axis=(1, 2, 3)), rtol=1e-6)
            predicted = record.workspace.preview_results_by_experiment["Sweep"]["predicted"]
            np.testing.assert_allclose(result["preview"]["predicted"], np.asarray(predicted).reshape(2, 4, 2)[1, :, 1])
            assert result["preview"]["included"] == [True, False, True, True]
        else:
            result = client.patch(path + "/view", json={
                "revision": record.revision, "plot_axis": "offset", "indices": {"t": 59},
            }).json()
            np.testing.assert_allclose(result["preview"]["observed"], values[59, 0].mean(axis=(1, 2, 3)), rtol=1e-6)
            run_operation(record.workspace, "full_fit", {"settings": {"num_steps": 2}}, lambda *_: None)
            fitted_session = record.workspace.session
            assert fitted_session.map_array("b1").shape == (60, 2, 3, 1)
            np.testing.assert_allclose(fitted_session.map_array("c"), np.broadcast_to(fitted_session.map_array("c")[:1], (60, 2, 3, 1)))
            record.info_cache.clear()
            result = client.get(path).json()
        saved = client.post(path + "/save")
        restored = client.post("/api/workspaces/upload", content=saved.content)
        assert restored.status_code == 200, restored.text
        assert restored.json()["preview"] == result["preview"]
        assert restored.json()["fit_settings"]["data_normalization"] == session.fit_settings.data_normalization
        np.testing.assert_array_equal(record.workspace.session.original_image_array, values)
        for closed in (restored.json(), result):
            assert client.request("DELETE", f"/api/workspaces/{closed['id']}", json={
                "revision": closed["revision"], "discard": True,
            }).status_code == 204

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
        ("phase", np.angle(complex_values.astype(source.dtype, copy=False))),
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
