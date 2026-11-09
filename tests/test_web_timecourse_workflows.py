"""Analyze workflows use independent arrays and shared native-grid masks."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.analysis import ROI
from mrax.data_struct import Quantity
from mrax.gui.session import GuiSession
from mrax.web.app import create_app


@pytest.fixture
def analysis(tmp_path):
    app = create_app(runtime_dir=tmp_path)
    with TestClient(app, base_url="http://localhost") as client:
        client.headers["x-mrax-token"] = client.get("/api/session").json()["token"]
        created = client.post("/api/workspaces/synthetic").json()
        record = app.state.records[created["id"]]
        values = np.arange(120, dtype=float).reshape(2, 3, 4, 5) + 1
        data = values + 1j * (values % 7)
        data[1, 2, 0, 0] = np.nan
        session = GuiSession()
        session.create_empty_study("Independent analysis")
        session.add_experiment(
            "Series", data, dims={"t": 2, "te": 3, "x": 4, "y": 5},
            model_name="t2_se", exp_type="t2_se",
            observables={"TE": Quantity([5, 15, 30], "ms"), "time": Quantity([0, 2], "s")},
            extra_metadata={"image_unit": "1"},
        )
        for name, rows in [("Left", slice(0, 2)), ("Right", slice(2, 4))]:
            mask = np.zeros((4, 5), dtype=np.uint16)
            mask[rows] = 17
            session.add_roi(ROI(name, mask, dims=("x", "y")))
        record.workspace.session = session
        record.workspace.label = "Independent analysis"
        w = client.get(f"/api/workspaces/{created['id']}").json()
        yield client, record, w, data


def calculate(client, w, **options):
    return client.post(
        f"/api/workspaces/{w['id']}/analysis/timecourse",
        json={
            "revision": w["revision"], "view_revision": w["view_revision"],
            "input": "image", "component": "real", "statistic": "mean",
            "axis": "te", "roi_names": ["Left", "Right"], **options,
        },
    )


@pytest.mark.parametrize("statistic", ["mean", "median", "std", "variance", "sum"])
def test_timecourse_reducers_keep_units_counts_and_current_acquisition(analysis, statistic):
    client, record, w, data = analysis
    changed = client.patch(
        f"/api/workspaces/{w['id']}/view", json={"revision": w["revision"], "indices": {"t": 1}}
    )
    assert changed.status_code == 200, changed.text
    w = changed.json()
    before = client.get(f"/api/workspaces/{w['id']}").json()
    response = calculate(client, w, statistic=statistic)
    assert response.status_code == 200, response.text
    result = response.json()
    np.testing.assert_array_equal(result["x"], [5, 15, 30])
    assert result["x_name"] == "TE (ms)" and result["unit"] == "1"
    function = getattr(np, "nanvar" if statistic == "variance" else "nan" + statistic)
    for row in result["series"]:
        roi = record.workspace.session.roi(row["roi"])
        source = data.real[1][:, roi.mask != 0]
        np.testing.assert_allclose(row["values"], function(source, axis=1))
        np.testing.assert_array_equal(row["counts"], np.isfinite(source).sum(axis=1))
    assert client.get(f"/api/workspaces/{w['id']}").json() == before
    np.testing.assert_array_equal(record.workspace.session.image_array, data)


def test_time_axis_and_fit_exclusions_remain_explicit(analysis):
    client, record, w, _ = analysis
    record.workspace.session.frame_mask.exclude(3)
    response = calculate(client, w, axis="t", respect_exclusions=True)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["x_name"] == "Elapsed time (s)" and result["x"] == [0, 2]
    assert result["included"] == [True, False]
    assert result["series"][0]["values"][1] is None
    assert result["series"][0]["counts"][1] == 0
    original = calculate(client, w, axis="t").json()
    assert original["series"][0]["values"][1] is not None


@pytest.mark.parametrize("component", ["auto", "magnitude", "real", "imag", "phase"])
def test_components_are_reduced_before_statistics(analysis, component):
    client, _, w, data = analysis
    result = calculate(client, w, component=component, roi_names=["Left"]).json()
    array = {"auto": np.abs, "magnitude": np.abs, "real": np.real, "imag": np.imag, "phase": np.angle}[component](data)
    np.testing.assert_allclose(result["series"][0]["values"], np.mean(array[0, :, :2, :], axis=(1, 2)))
    assert result["unit"] == ("rad" if component == "phase" else "1")
    if component == "phase":
        variance = calculate(client, w, component=component, statistic="variance").json()
        assert variance["unit"] == "(rad)^2"


def test_missing_samples_are_not_fabricated_as_zero(analysis):
    client, record, w, data = analysis
    data[:, 1] = np.nan
    record.workspace.session.replace_image_data(data)
    result = calculate(client, w, statistic="sum").json()
    assert all(row["values"][1] is None and row["counts"][1] == 0 for row in result["series"])


def test_original_and_view_inputs_do_not_mutate_processed_data(analysis):
    client, record, w, data = analysis
    record.workspace.session.replace_image_data(data * 2)
    image = calculate(client, w).json()
    original = calculate(client, w, input="original").json()
    viewed = calculate(client, w, input="view").json()
    np.testing.assert_allclose(image["series"][0]["values"], np.asarray(original["series"][0]["values"]) * 2)
    assert original["series"] == viewed["series"]
    np.testing.assert_array_equal(record.workspace.session.image_array, data * 2)


def test_view_mask_applies_only_to_the_explicit_view_input(analysis):
    client, record, w, _ = analysis
    state = record.workspace.session.visual_state
    state.show_original = False
    state.mask_enabled = True
    state.mask_roi = "Left"
    result = calculate(client, w, input="view", roi_names=["Right"]).json()
    assert result["series"][0]["values"] == [None, None, None]
    assert result["series"][0]["counts"] == [0, 0, 0]
    original = calculate(client, w, input="image", roi_names=["Right"]).json()
    assert all(value is not None for value in original["series"][0]["values"])


@pytest.mark.parametrize("component", ["magnitude", "phase"])
def test_reconstruction_components_are_not_projected_twice(analysis, component):
    from mrax.reco.components import RAW_RECONSTRUCTION_COMPLEX_KEY, RAW_RECONSTRUCTION_KEY

    client, record, _, data = analysis
    values = data.reshape(6, 4, 5)
    session = GuiSession()
    session.create_empty_study("Independent reconstruction")
    session.add_experiment(
        "Reconstruction", values, dims={"frame": 6, "x": 4, "y": 5}, model_name="identity",
        processed={RAW_RECONSTRUCTION_KEY: np.abs(values), RAW_RECONSTRUCTION_COMPLEX_KEY: values},
        extra_metadata={"image_unit": "1"},
    )
    mask = np.zeros((4, 5), dtype=np.uint16)
    mask[:2] = 17
    session.add_roi(ROI("Left", mask, dims=("x", "y")))
    session.active_reconstruction_component = "phase"
    session.active_reconstruction_unwrap_phase = False
    record.workspace.session = session
    w = client.get(f"/api/workspaces/{record.workspace.key}").json()
    response = calculate(client, w, input="reconstruction", component=component,
                         axis="frame", roi_names=["Left"])
    assert response.status_code == 200, response.text
    result = response.json()
    expected = np.abs(values) if component == "magnitude" else np.angle(values)
    np.testing.assert_allclose(result["series"][0]["values"], np.nanmean(expected[:, :2], axis=(1, 2)), atol=1e-6)
    assert result["unit"] == ("rad" if component == "phase" else "1")
    assert session.active_reconstruction_component == "phase"


def test_single_frame_axis_uses_echo_coordinates(analysis):
    client, _, _, _ = analysis
    w = client.post("/api/workspaces/synthetic").json()
    response = client.post(
        f"/api/workspaces/{w['id']}/actions",
        json={"revision": w["revision"], "view_revision": w["view_revision"],
              "action": "roi.create", "options": {"name": "ROI", "bounds": [4, 4, 6, 6]}},
    )
    assert response.status_code == 200, response.text
    result = calculate(client, response.json(), axis="frame", roi_names=["ROI"]).json()
    assert result["x_name"] == "TE (ms)"
    assert result["x"] == [10, 20, 40, 60, 80, 120, 160, 200]


@pytest.mark.parametrize("options", [
    {"axis": "x"}, {"axis": "missing"}, {"roi_names": []},
    {"roi_names": ["missing"]}, {"roi_names": ["Left", "Left"]},
    {"input": "map:t2"}, {"component": "invalid"}, {"statistic": "invalid"},
])
def test_invalid_analysis_requests_are_rejected(analysis, options):
    client, _, w, _ = analysis
    assert calculate(client, w, **options).status_code == 422


def test_stale_and_disabled_roi_requests_are_rejected(analysis):
    client, _, w, _ = analysis
    assert calculate(client, w, revision=w["revision"] + 1).status_code == 409
    assert calculate(client, w, view_revision=w["view_revision"] + 1).status_code == 409
    changed = client.post(
        f"/api/workspaces/{w['id']}/actions",
        json={"revision": w["revision"], "action": "roi.disable_all"},
    )
    assert changed.status_code == 200, changed.text
    assert calculate(client, changed.json()).status_code == 422
