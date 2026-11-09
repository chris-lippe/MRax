"""Independent synthetic data for non-destructive experiment pipelines."""

import copy

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.ndimage import gaussian_filter

from mrax.analysis import ROI
from mrax.gui.services.analysis_workflows import run_operation, update_metadata
from mrax.gui.services.export import _export_object_records_for_session
from mrax.gui.services.pipeline import (
    change_pipeline,
    display_session,
    pipeline_info,
    project_roi,
    snapshot,
)
from mrax.gui.services.processing import run_processing
from mrax.gui.services.roi_tools import roi_command
from mrax.gui.services.timecourse import analyze_timecourse
from mrax.gui.services.view_data import curve_for_workspace, render_image
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import load_workspace_bundle, save_workspace_bundle
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


def test_external_reference_and_processing_masks_are_frozen_for_reproducible_replay(tmp_path):
    w, source = workspace()
    w.session.add_experiment(
        "Reference", np.full((1, 9, 11), 2), dims={"frame": 1, "x": 9, "y": 11}
    )
    w.session.select_experiment("Experiment")
    mask = np.zeros((9, 11), dtype=np.int16)
    mask[2:7, 3:8] = 17
    w.session.add_roi(ROI("Tissue", mask, dims=("x", "y")))
    process(w, "normalize", reference_experiment="Reference", roi_names=["Tissue"])
    np.testing.assert_allclose(w.session.image_array[:, 3, 4], source[:, 3, 4] / 2)
    w.session.select_experiment("Reference")
    w.session.replace_image_data(np.full((1, 9, 11), 20))
    w.session.select_experiment("Experiment")
    w.session.remove_shared_roi("Tissue")
    step = pipeline_info(w)["steps"][0]["id"]
    change(w, action="toggle", step_id=step, enabled=False)
    change(w, action="toggle", step_id=step, enabled=True)
    np.testing.assert_allclose(w.session.image_array[:, 3, 4], source[:, 3, 4] / 2)
    assert np.isnan(w.session.image_array[:, 0, 0]).all()
    path = save_workspace_bundle(w, tmp_path / "pipeline.mrax")
    restored = load_workspace_bundle(path, tmp_path / "open", key="restored")
    assert pipeline_info(restored) == pipeline_info(w)
    change(restored, action="clear")
    np.testing.assert_array_equal(restored.session.image_array, source)
    with pytest.raises(ValueError):
        restored.pipelines["Experiment"]["source"]["data"][0, 0, 0] = 0


def test_registration_is_replaced_in_place_and_projects_labels_without_modifying_native_masks():
    w, source = workspace()
    mask = np.zeros((9, 11), dtype=np.int16)
    mask[3:6, 4:7] = 17
    w.session.add_roi(ROI("Tissue", mask, dims=("x", "y")))
    options = {
        "manual": True,
        "manual_settings": {"shift_x": 1},
        "reference_experiment": "Experiment",
        "moving_experiment": "Experiment",
    }
    run_operation(w, "register", options, lambda *_: None)
    first = pipeline_info(w)["steps"][0]["id"]
    assert w.session.experiment_names == ["Experiment"]
    projected = project_roi(w, w.session.rois[0])
    assert set(np.unique(projected.mask)) == {0, 17}
    assert not np.array_equal(projected.mask, mask)
    np.testing.assert_array_equal(w.session.rois[0].mask, mask)
    options["manual_settings"]["shift_x"] = 2
    run_operation(w, "register", options, lambda *_: None)
    assert len(pipeline_info(w)["steps"]) == 1
    assert pipeline_info(w)["steps"][0]["id"] == first
    change(w, action="remove", step_id=first)
    np.testing.assert_array_equal(w.session.image_array, source)
    np.testing.assert_array_equal(project_roi(w, w.session.rois[0]).mask, mask)


def test_image_processing_keeps_existing_fit_fields_until_a_new_fit_replaces_them():
    w, _source = workspace()
    fields = {"test_map": np.arange(99).reshape(9, 11)}
    w.session.replace_fitted_maps(fields)
    w.fit_results_by_experiment["Experiment"] = {
        "fields": fields,
        "pipeline_signature": "original",
        "model": "identity",
    }
    before = copy.deepcopy(fields)
    process(w, "gaussian", axes=["x", "y"])
    assert pipeline_info(w)["fits_stale"]
    np.testing.assert_array_equal(w.session.fitted_maps["test_map"], before["test_map"])
    np.testing.assert_array_equal(
        w.fit_results_by_experiment["Experiment"]["fields"]["test_map"], before["test_map"]
    )
    change(w, action="clear")
    assert not pipeline_info(w)["fits_stale"]
    np.testing.assert_array_equal(w.session.fitted_maps["test_map"], before["test_map"])


def test_retained_maps_keep_their_own_grid_and_roi_edits_are_stored_in_source_coordinates():
    w, source = workspace()
    mask = np.zeros((9, 11), dtype=np.int16)
    mask[2:6, 3:7] = 17
    w.session.add_roi(ROI("Tissue", mask, dims=("x", "y")))
    w.session.replace_fitted_maps({"map": np.arange(99, dtype=np.float32).reshape(9, 11)})
    state = snapshot(w.session)
    w.fit_results_by_experiment["Experiment"] = {
        "fields": w.session.fitted_maps,
        "pipeline_signature": "original",
        "pipeline_grid": {k: state[k] for k in ("dims", "geometry", "unit")},
        "pipeline_steps": [],
    }
    w.session.current_experiment["fit_grid"] = w.fit_results_by_experiment["Experiment"][
        "pipeline_grid"
    ]
    w.session.add_experiment("Reference", np.zeros((1, 7, 8)), dims={"frame": 1, "x": 7, "y": 8})
    w.session.select_experiment("Experiment")
    run_operation(
        w,
        "register",
        {
            "reference_experiment": "Reference",
            "moving_experiment": "Experiment",
            "manual": True,
            "manual_settings": {"shift_x": 1},
        },
        lambda *_: None,
    )
    assert w.session.spatial_shape == (7, 8)
    assert render_image(w)[0].shape == (7, 8)
    projected_mask = np.asarray(project_roi(w, w.session.roi("Tissue")).mask) != 0
    expected_signal = np.asarray(w.session.image_array)[:, projected_mask].mean(axis=1)
    np.testing.assert_allclose(curve_for_workspace(w)["observed"], expected_signal, rtol=1e-6)
    timecourse = analyze_timecourse(
        w,
        {
            "roi_names": ["Tissue"],
            "component": "auto",
            "input": "image",
            "axis": "frame",
            "statistic": "mean",
            "respect_exclusions": True,
        },
    )
    np.testing.assert_allclose(timecourse["series"][0]["values"], expected_signal, rtol=1e-6)
    roi_command(w, "create", {"name": "Drawn", "tool": "rect", "bounds": [2, 2, 2, 2], "label": 19})
    assert w.session.roi("Drawn").mask.shape == (9, 11)
    assert project_roi(w, w.session.roi("Drawn")).mask.shape == (7, 8)
    np.testing.assert_array_equal(w.session.roi("Tissue").mask, mask)
    w.view_label = "Map"
    assert render_image(w)[0].shape == (9, 11)
    assert display_session(w).spatial_shape == (9, 11)
    assert pipeline_info(w)["fits_stale"]
    records = {
        r["kind"]: r
        for r in _export_object_records_for_session(w.session)
        if r.get("experiment") == "Experiment"
    }
    assert records["image"]["dims"] == {"frame": 4, "x": 7, "y": 8}
    assert records["original_image"]["dims"] == {"frame": 4, "x": 9, "y": 11}
    assert records["map"]["dims"] == {"frame": 4, "x": 9, "y": 11}
    change(w, action="clear")
    np.testing.assert_array_equal(w.session.image_array, source)
    assert w.session.roi("Drawn").mask.shape == (9, 11)


def test_metadata_edits_and_experiment_selection_survive_pipeline_replay_and_reload(tmp_path):
    w, source = workspace()
    process(w, "gaussian", axes=["x", "y"])
    update_metadata(
        w,
        {
            "observables": {"time": {"values": [0, 2, 4, 6], "unit": "s"}},
            "metadata": {"note": "Synthetic edit"},
        },
    )
    process(w, "median", axes=["x", "y"], parameters={"size": 3})
    assert list(w.session.current_experiment["observables"]["time"].values) == [0, 2, 4, 6]
    assert w.session.current_experiment["extra_metadata"]["note"] == "Synthetic edit"
    w.session.add_experiment("Other", source * 3, dims={"frame": 4, "x": 9, "y": 11})
    process(w, "gaussian", axes=["x", "y"])
    assert set(w.pipelines) == {"Experiment", "Other"}
    w.session.select_experiment("Experiment")
    change(w, action="enabled", enabled=False)
    target = tmp_path / "synthetic-pipelines.mrax"
    save_workspace_bundle(w, target)
    restored = load_workspace_bundle(target, tmp_path / "restored", key="restored")
    assert restored.session.selected_experiment == "Experiment"
    assert pipeline_info(restored)["enabled"] is False
    np.testing.assert_array_equal(restored.session.image_array, source)
    change(restored, action="enabled", enabled=True)
    assert list(restored.session.current_experiment["observables"]["time"].values) == [0, 2, 4, 6]
    restored.session.select_experiment("Other")
    np.testing.assert_allclose(
        restored.session.image_array,
        gaussian_filter(source * 3, sigma=(0, 1, 1), mode="mirror"),
        atol=1e-5,
    )


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


def test_reconstruction_becomes_the_first_pipeline_step_and_preserves_original_vendor_data(
    tmp_path, monkeypatch
):
    import jax.numpy as jnp

    from mrax.data_struct import AcquisitionData
    from mrax.reco import fftnc

    image = np.arange(20, dtype=np.float32).reshape(1, 4, 5) / 20 + 1
    acquisition = AcquisitionData(
        np.asarray(fftnc(jnp.asarray(image[:, None]))), (4, 5), ("phase", "read")
    )
    session = GuiSession()
    session.create_empty_study("Synthetic raw pipeline")
    session.add_experiment(
        "Raw", image * 2, dims={"frame": 1, "x": 4, "y": 5}, kspace=acquisition.samples
    )
    session.current_experiment["acquisition"] = acquisition
    session.study.datasets[0].acquisition = acquisition
    w = StudyWorkspace(key="raw", label=session.study_name, session=session)
    process(w, "gaussian", axes=["x", "y"], parameters={"sigma": 0.5})
    run_operation(w, "reconstruct", {"view": "mrax_reconstruction"}, lambda *_: None)
    assert [s["kind"] for s in pipeline_info(w)["steps"]] == ["reconstruct", "process"]
    assert w.session.has_reconstruction
    expected = gaussian_filter(image, sigma=(0, 0.5, 0.5), mode="mirror")
    np.testing.assert_allclose(w.session.image_array, expected, atol=1e-5)
    change(w, action="enabled", enabled=False)
    np.testing.assert_array_equal(w.session.image_array, image * 2)
    change(w, action="enabled", enabled=True)
    reconstruction, denoising = pipeline_info(w)["steps"]
    mask = np.full((4, 5), 9, dtype=np.uint16)
    w.session.add_roi(ROI("Tissue", mask, dims=("x", "y")))
    fitted = np.full((4, 5), 73.0)
    w.session.replace_fitted_maps({"retained": fitted})
    state = snapshot(w.session)
    grid = {key: state[key] for key in ("dims", "geometry", "unit")}
    w.fit_results_by_experiment["Raw"] = {
        "fields": w.session.fitted_maps,
        "pipeline_signature": pipeline_info(w)["signature"],
        "pipeline_grid": grid,
        "pipeline_steps": copy.deepcopy(pipeline_info(w)["steps"]),
    }
    w.session.current_experiment["fit_grid"] = grid

    def unavailable_raw(*_args, **_kwargs):
        raise AssertionError("A cached reconstruction must not read raw data or reconstruct again")

    monkeypatch.setattr(GuiSession, "raw_data_loaded", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(GuiSession, "load_raw_data_for_experiment", unavailable_raw)
    monkeypatch.setattr(GuiSession, "reconstruct_raw_data_for_experiment", unavailable_raw)
    change(w, action="toggle", step_id=reconstruction["id"], enabled=False)
    np.testing.assert_allclose(w.session.image_array, expected * 2, atol=1e-5)
    assert pipeline_info(w)["fits_stale"]
    change(w, action="toggle", step_id=reconstruction["id"], enabled=True)
    np.testing.assert_allclose(w.session.image_array, expected, atol=1e-5)
    assert not pipeline_info(w)["fits_stale"]
    assert [step["id"] for step in pipeline_info(w)["steps"]] == [
        reconstruction["id"],
        denoising["id"],
    ]
    np.testing.assert_array_equal(w.session.fitted_maps["retained"], fitted)
    np.testing.assert_array_equal(w.session.rois[0].mask, mask)
    np.testing.assert_array_equal(w.session.original_image_array, image * 2)
    change(w, action="remove", step_id=reconstruction["id"])
    np.testing.assert_allclose(w.session.image_array, expected * 2, atol=1e-5)
    run_operation(w, "reconstruct", {"view": "mrax_reconstruction"}, lambda *_: None)
    assert pipeline_info(w)["steps"][1]["id"] == denoising["id"]
    np.testing.assert_allclose(w.session.image_array, expected, atol=1e-5)
    bundle = save_workspace_bundle(w, tmp_path / "source-choice.mrax")
    restored = load_workspace_bundle(bundle, tmp_path / "restored", key="restored")
    assert pipeline_info(restored)["steps"][0]["kind"] == "reconstruct"
    assert pipeline_info(restored)["steps"][0]["enabled"]
    np.testing.assert_allclose(restored.session.image_array, expected, atol=1e-5)
    np.testing.assert_array_equal(restored.session.fitted_maps["retained"], fitted)
    np.testing.assert_array_equal(restored.session.rois[0].mask, mask)
    change(restored, action="clear")
    np.testing.assert_array_equal(restored.session.image_array, image * 2)
    np.testing.assert_array_equal(restored.session.fitted_maps["retained"], fitted)
