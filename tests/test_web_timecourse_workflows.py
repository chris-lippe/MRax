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
            "Series",
            data,
            dims={"t": 2, "te": 3, "x": 4, "y": 5},
            model_name="t2_se",
            exp_type="t2_se",
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
            "revision": w["revision"],
            "view_revision": w["view_revision"],
            "input": "image",
            "component": "real",
            "statistic": "mean",
            "axis": "te",
            "roi_names": ["Left", "Right"],
            **options,
        },
    )


def test_timecourse_spread_keeps_units_counts_and_current_acquisition(analysis):
    client, record, w, data = analysis
    changed = client.patch(
        f"/api/workspaces/{w['id']}/view", json={"revision": w["revision"], "indices": {"t": 1}}
    )
    assert changed.status_code == 200, changed.text
    w = changed.json()
    before = client.get(f"/api/workspaces/{w['id']}").json()
    response = calculate(client, w, statistic="std")
    assert response.status_code == 200, response.text
    result = response.json()
    np.testing.assert_array_equal(result["x"], [5, 15, 30])
    assert result["x_name"] == "TE (ms)" and result["unit"] == "1"
    function = np.nanstd
    for row in result["series"]:
        roi = record.workspace.session.roi(row["roi"])
        source = data.real[1][:, roi.mask != 0]
        np.testing.assert_allclose(row["values"], function(source, axis=1))
        np.testing.assert_array_equal(row["counts"], np.isfinite(source).sum(axis=1))
    assert client.get(f"/api/workspaces/{w['id']}").json() == before
    np.testing.assert_array_equal(record.workspace.session.image_array, data)
