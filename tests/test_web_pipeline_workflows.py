"""Independent synthetic data for non-destructive experiment pipelines."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.ndimage import gaussian_filter

from mrax.gui.services.pipeline import (
    change_pipeline,
    pipeline_info,
)
from mrax.gui.services.processing import run_processing
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession
from mrax.web.app import create_app


def workspace():
    data = np.random.default_rng(9).normal(10, 1, (4, 9, 11)).astype(np.float32)
    session = GuiSession()
    session.create_empty_study("Independent synthetic pipeline")
    session.add_experiment("Experiment", data, dims={"frame": 4, "x": 9, "y": 11})
    return StudyWorkspace(key="synthetic", label=session.study_name, session=session), data


def process(workspace, method, **options):
    return run_processing(
        workspace, {"method": method, "input": "image", **options}, lambda *_: None
    )


def change(workspace, **options):
    return change_pipeline(workspace, options, lambda *_: None)


def test_editing_replays_downstream_steps_from_protected_source_without_new_experiments():
    w, source = workspace()
    process(w, "gaussian", axes=["x", "y"], parameters={"sigma": 1})
    first = pipeline_info(w)["steps"][0]["id"]
    process(w, "baseline", axis="frame", parameters={"baseline_frames": 1, "mode": "subtract"})
    expected = gaussian_filter(source, sigma=(0, 1, 1), mode="mirror")
    np.testing.assert_allclose(w.session.image_array, expected - expected[:1], atol=1e-5)
    process(w, "gaussian", axes=["x", "y"], parameters={"sigma": 2})
    expected = gaussian_filter(source, sigma=(0, 2, 2), mode="mirror")
    np.testing.assert_allclose(w.session.image_array, expected - expected[:1], atol=1e-5)
    assert w.session.experiment_names == ["Experiment"]
    assert [s["method"] for s in pipeline_info(w)["steps"]] == ["gaussian", "baseline"]
    assert pipeline_info(w)["steps"][0]["id"] == first
    np.testing.assert_array_equal(w.session.original_image_array, source)
    change(w, action="remove", step_id=first)
    np.testing.assert_allclose(w.session.image_array, source - source[:1], atol=1e-5)
    change(w, action="enabled", enabled=False)
    np.testing.assert_array_equal(w.session.image_array, source)
    change(w, action="enabled", enabled=True)
    np.testing.assert_allclose(w.session.image_array, source - source[:1], atol=1e-5)
    change(w, action="clear")
    np.testing.assert_array_equal(w.session.image_array, source)
    assert pipeline_info(w)["steps"] == []


def test_pipeline_failures_and_cancellation_cannot_partially_replace_working_state():
    w, source = workspace()
    process(w, "gaussian", axes=["x", "y"], parameters={"sigma": 1})
    before = np.asarray(w.session.image_array).copy()
    state = pipeline_info(w)
    with pytest.raises(ValueError):
        process(w, "baseline", axis="missing")
    np.testing.assert_array_equal(w.session.image_array, before)
    assert pipeline_info(w) == state

    def cancel(*_args):
        raise RuntimeError("Synthetic cancellation")

    with pytest.raises(RuntimeError, match="Synthetic cancellation"):
        run_processing(w, {"method": "median", "axes": ["x", "y"]}, cancel)
    np.testing.assert_array_equal(w.session.image_array, before)
    np.testing.assert_array_equal(w.session.original_image_array, source)
    assert pipeline_info(w) == state


def test_fits_use_the_current_pipeline_and_keep_maps_until_rerun(tmp_path):
    import time

    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as client:
        token = client.get("/api/session").json()["token"]
        headers = {"x-mrax-token": token}
        current = client.post("/api/workspaces/synthetic", headers=headers).json()
        path = "/api/workspaces/" + current["id"]

        def submit(operation, **options):
            nonlocal current
            response = client.post(
                path + "/tasks",
                headers=headers,
                json={
                    "revision": current["revision"],
                    "operation": operation,
                    "options": options,
                },
            )
            assert response.status_code == 200, response.text
            job = response.json()
            for _ in range(1000):
                state = client.get("/api/jobs/" + job["id"]).json()
                if state["status"] in {"completed", "failed", "cancelled"}:
                    assert state["status"] == "completed", state["detail"]
                    current = client.get(path).json()
                    return
                time.sleep(0.01)
            raise AssertionError("Synthetic pipeline job did not finish")

        submit("process", method="gaussian", axes=["x", "y"], parameters={"sigma": 1})
        submit("full_fit", use_roi=False, settings={"num_steps": 600})
        assert current["pipeline"]["fits_stale"] is False
        record = client.app.state.records[current["id"]]
        fields = {
            k: np.asarray(v.values if hasattr(v, "values") else v).copy()
            for k, v in record.workspace.session.fitted_maps.items()
        }
        first_signature = current["fit_info"]["pipeline_signature"]
        submit("process", method="median", axes=["x", "y"], parameters={"size": 3})
        assert len(current["experiments"]) == 1 and len(current["pipeline"]["steps"]) == 1
        assert current["pipeline"]["fits_stale"]
        for name, values in fields.items():
            value = record.workspace.session.fitted_maps[name]
            np.testing.assert_array_equal(
                np.asarray(value.values if hasattr(value, "values") else value), values
            )
        submit("full_fit", use_roi=False, settings={"num_steps": 600})
        assert not current["pipeline"]["fits_stale"]
        assert current["fit_info"]["pipeline_signature"] != first_signature
        assert current["fit_info"]["pipeline_signature"] == current["pipeline"]["signature"]
