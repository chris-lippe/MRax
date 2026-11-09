"""Toolbox processing, shared masks, quantitative outputs and persistent workflows."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.analysis import ROI
from mrax.data_struct import Quantity
from mrax.data_struct.geometry import SpatialGeometry
from mrax.gui.services.analysis_workflows import run_alignment
from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
from mrax.gui.services.processing import run_processing
from mrax.gui.services.roi_tools import roi_command
from mrax.gui.services.scene import experiment_catalog, scene_info
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def create(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def latest(client, w):
    return client.get(f"/api/workspaces/{w['id']}").json()


def action(client, w, operation, **options):
    response = client.post(
        f"/api/workspaces/{w['id']}/actions",
        json={
            "revision": w["revision"],
            "view_revision": w["view_revision"],
            "action": operation,
            "options": options,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def view(client, w, **options):
    response = client.patch(
        f"/api/workspaces/{w['id']}/view", json={"revision": w["revision"], **options}
    )
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, w, job):
    deadline = time.monotonic() + 80
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] not in {"queued", "running", "cancelling"}:
            assert state["status"] == "completed", state["detail"]
            return latest(client, w), state
        time.sleep(0.05)
    pytest.fail("Synthetic processing exceeded its bounded timeout")


def job(client, w, operation, **options):
    response = client.post(
        f"/api/workspaces/{w['id']}/tasks",
        json={"revision": w["revision"], "operation": operation, "options": options},
    )
    assert response.status_code == 200, response.text
    return finish(client, w, response.json())


def series(values, *, complex_data=False, dims=None):
    session = synthetic_t2_session()
    dims = dims or {"frame": len(values), "x": 6, "y": 7}
    data = np.broadcast_to(
        np.asarray(values, dtype=np.float32)[:, None, None], tuple(dims.values())
    ).copy()
    if complex_data:
        data = data * (1 + 2j)
    session.add_experiment("Series", data, dims=dims, model_name="identity")
    return StudyWorkspace("test", "Synthetic processing", session)


def run(w, method, **options):
    return run_processing(w, {"method": method, "input": "image", **options}, lambda *args: None)


def test_denoise_fit_statistics_and_save_form_one_workflow(client):
    w = create(client)
    workspace = client.app.state.records[w["id"]].workspace
    truth = np.asarray(workspace.session.image_array).copy()
    source = workspace.session.build_experiment()
    geometry = workspace.session.build_dataset().spatial_geometry
    noisy = truth + np.random.default_rng(12).normal(0, 20000, truth.shape)
    workspace.session.add_experiment(
        "Noisy",
        noisy,
        dims=source.dims,
        observables=source.observables,
        model_name=workspace.session.current_model_name,
        extra_metadata={"spatial_geometry": geometry.to_metadata()},
    )
    w = latest(client, w)
    w = action(client, w, "roi.create", name="Tissue", tool="rect", bounds=[7, 5, 12, 10], label=7)
    w = action(client, w, "frames.exclude", indices=[7])
    w = view(client, w, scene={"hidden_rois": ["Tissue"], "fit_rois": ["Tissue"]})
    w, result = job(
        client,
        w,
        "process",
        method="gaussian",
        input="image",
        axes=["x", "y"],
        parameters={"sigma": 1.0},
        name="Denoised",
    )
    assert result["summary"]["experiment"] == "Noisy" and w["experiment"] == "Noisy"
    assert w["experiments"] == ["T2 phantom", "Noisy"]
    workspace = client.app.state.records[w["id"]].workspace
    denoised = np.asarray(workspace.session.image_array)
    assert np.mean((denoised - truth) ** 2) < 0.3 * np.mean((noisy - truth) ** 2)
    np.testing.assert_array_equal(workspace.session.original_image_array, noisy.astype(np.float32))
    assert w["rois"][0]["enabled"] and not w["rois"][0]["visible"]
    assert w["rois"][0]["labels"] == [7] and w["frame_exclusions"] == [7]
    assert w["scene"]["fit_rois"] == ["Tissue"]
    assert w["scene"]["layers"] == []
    w, _ = job(
        client, w, "full_fit", roi_names=["Tissue"], use_roi=True, settings={"num_steps": 600}
    )
    statistics = client.get(f"/api/workspaces/{w['id']}/roi-statistics", params={"map_name": "t2"})
    assert statistics.status_code == 200, statistics.text
    rows = {row["stat"]: row for row in statistics.json()}
    assert rows["median"]["value"] == pytest.approx(80, abs=2)
    assert rows["count"]["value"] == 120 and rows["mean"]["unit"] == "ms"
    w, _ = job(
        client,
        w,
        "process",
        method="baseline",
        input="image",
        axis="frame",
        parameters={"baseline_frames": 2, "mode": "divide"},
        name="Normalized",
    )
    assert w["model"] == "identity" and w["unit"] == "1"
    catalog = client.get(f"/api/workspaces/{w['id']}/processing").json()
    assert [row["method"] for row in catalog["history"]] == ["gaussian", "baseline"]
    bundle = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=bundle.content)
    assert restored.status_code == 200, restored.text
    r = restored.json()
    assert (
        client.get(f"/api/workspaces/{r['id']}/processing").json()["history"] == catalog["history"]
    )
    assert r["scene"]["fit_rois"] == ["Tissue"] and not r["rois"][0]["visible"]
    r = view(client, r, experiment="Noisy", view="Map", map_name="t2")
    assert r["fit_info"]["roi_names"] == ["Tissue"]


@pytest.mark.parametrize(
    "method,parameters",
    [
        ("gaussian", {"sigma": 1}),
        ("median", {"size": 3}),
        ("uniform", {"size": 3}),
        ("regularized", {"num_steps": 2}),
    ],
)
def test_spatial_filters_preserve_complex_values_and_acquisition_axis_order(method, parameters):
    session = synthetic_t2_session()
    x, frame, y = np.indices((6, 3, 7))
    data = (np.sin(x) + np.cos(y) + frame * 10) * (1 + 2j)
    session.add_experiment(
        "Permuted", data, dims={"x": 6, "frame": 3, "y": 7}, model_name="identity"
    )
    w = StudyWorkspace("test", "Synthetic processing", session)
    run(w, method, axes=["x", "y"], parameters=parameters)
    processed = np.asarray(w.session.image_array)
    assert processed.shape == data.shape and list(w.session.experiment_dims) == ["x", "frame", "y"]
    assert np.iscomplexobj(processed) and np.any(processed.imag)
    if method != "regularized":
        np.testing.assert_allclose(processed.imag, 2 * processed.real, atol=2e-5)
    np.testing.assert_allclose(
        np.diff(processed, axis=1), 10 + 20j, atol=0.002 if method == "regularized" else 2e-5
    )


@pytest.mark.parametrize("mode,expected", [("subtract", [0, 0, 10]), ("divide", [1, 1, 2])])
def test_baseline_correction_reuses_existing_modes(mode, expected):
    w = series([10, 10, 20])
    run(w, "baseline", axis="frame", parameters={"baseline_frames": 2, "mode": mode})
    np.testing.assert_allclose(np.asarray(w.session.image_array)[:, 0, 0], expected)


def test_temporal_smoothing_and_spike_correction_leave_other_voxels_unchanged():
    w = series([0, 0, 0, 0, 0])
    data = np.zeros((5, 6, 7), dtype=np.float32)
    data[2, 2, 3] = 10
    w.session.replace_image_data(data)
    w.session.current_experiment["original_mr_image"] = data.copy()
    run(w, "temporal_gaussian", axis="frame", parameters={"sigma": 1})
    result = np.asarray(w.session.image_array)
    assert 0 < result[2, 2, 3] < 10 and result[1, 2, 3] > 0
    assert not np.any(result[:, 0, 0])
    w = series([10, 10, 100, 10, 10])
    run(w, "despike", axis="frame")
    np.testing.assert_array_equal(w.session.image_array, np.full((5, 6, 7), 10))


def test_reference_and_dilution_correction_preserve_input_and_normalized_units():
    w = series([2, 4, 6])
    run(w, "normalize", reference_experiment="Series", reference_frame=0)
    np.testing.assert_allclose(np.asarray(w.session.image_array)[:, 0, 0], [1, 2, 3])
    assert w.session.current_experiment["extra_metadata"]["image_unit"] == "1"
    w = series([10, 20, 30])
    w.session.add_experiment(
        "Dilution",
        np.broadcast_to(np.array([1, 2, 3])[:, None, None], (3, 6, 7)),
        dims={"frame": 3, "x": 6, "y": 7},
        model_name="identity",
    )
    w.session.select_experiment("Series")
    run(
        w,
        "dilution",
        reference_experiment="Dilution",
        parameters={"baseline_frames": 1, "smooth_sigma": 0},
    )
    np.testing.assert_allclose(w.session.image_array, 10)


def test_normalizing_inside_roi_does_not_mix_signal_and_ratio_units():
    w = series([2, 4, 6])
    mask = np.zeros((6, 7), dtype=np.uint16)
    mask[1:3, 2:4] = 9
    w.session.add_roi(ROI("Mask", mask, dims=("x", "y")))
    run(w, "normalize", reference_experiment="Series", reference_frame=0, roi_names=["Mask"])
    values = np.asarray(w.session.image_array)
    np.testing.assert_allclose(values[:, 1, 2], [1, 2, 3])
    assert np.isnan(values[:, 0, 0]).all()
    np.testing.assert_array_equal(w.pipelines["Series"]["steps"][0]["mask"]["mask"], mask != 0)
    assert w.session.reconstruction_array is None
    np.testing.assert_array_equal(w.session.original_image_array[:, 0, 0], [2, 4, 6])


def test_reference_processing_rejects_different_units_and_physical_grids():
    w = series([2, 4, 6])
    w.session.current_experiment.setdefault("extra_metadata", {})["image_unit"] = "ms"
    w.session.build_dataset().spatial_geometry = SpatialGeometry(
        affine=np.eye(4), spatial_shape=(6, 7), status="known"
    )
    shifted = np.eye(4)
    shifted[0, 3] = 3
    w.session.add_experiment(
        "Reference",
        np.ones((3, 6, 7)),
        dims={"frame": 3, "x": 6, "y": 7},
        model_name="identity",
        extra_metadata={
            "image_unit": "s",
            "spatial_geometry": SpatialGeometry(
                affine=shifted, spatial_shape=(6, 7), status="known"
            ).to_metadata(),
        },
    )
    w.session.select_experiment("Series")
    with pytest.raises(ValueError, match="physical geometry"):
        run(w, "normalize", reference_experiment="Reference")
    reference = w.session.study.entries_by_name()["Reference"][2]
    reference.spatial_geometry = w.session.build_dataset().spatial_geometry
    with pytest.raises(ValueError, match="matching units"):
        run(w, "normalize", reference_experiment="Reference")
    assert w.session.experiment_names[-1] == "Reference"


def test_dilution_reference_requires_matching_acquisition_coordinates():
    w = series([10, 20, 30])
    w.session.build_experiment().observables["time"] = Quantity([0, 1, 2], "s")
    w.session.add_experiment(
        "Dilution",
        np.ones((3, 6, 7)),
        dims={"frame": 3, "x": 6, "y": 7},
        model_name="identity",
        observables={"time": Quantity([0, 2, 4], "s")},
    )
    w.session.select_experiment("Series")
    with pytest.raises(ValueError, match="acquisition coordinates"):
        run(w, "dilution", reference_experiment="Dilution")


def test_dti_metrics_are_available_to_view_statistics_and_bundle_round_trip(client):
    w = create(client)
    workspace = client.app.state.records[w["id"]].workspace
    shape = workspace.session.spatial_shape
    fields = {
        name: Quantity(np.full(shape, value, dtype=np.float32), "um^2/ms")
        for name, value in {"dxx": 3, "dyy": 2, "dzz": 1, "dxy": 0, "dxz": 0, "dyz": 0}.items()
    }
    workspace.session.replace_fitted_maps(fields)
    w = latest(client, w)
    w = action(client, w, "roi.create", name="Tensor mask", tool="rect", bounds=[2, 2, 4, 4])
    w, result = job(client, w, "process", method="dti", roi_names=["Tensor mask"])
    assert len(result["summary"]["maps"]) == 6 and w["view"] == "Map" and w["unit"] == "1"
    data = client.app.state.records[w["id"]].workspace.session.fitted_maps
    assert np.asarray(data["DTI ADC"])[2, 2, 0] == pytest.approx(2)
    assert np.asarray(data["DTI FA"])[2, 2, 0] == pytest.approx(np.sqrt(3 / 14))
    assert np.isnan(np.asarray(data["DTI FA"])[0, 0, 0])
    rows = client.get(f"/api/workspaces/{w['id']}/roi-statistics?map_name=DTI%20ADC").json()
    assert next(row for row in rows if row["stat"] == "mean")["unit"] == "um^2/ms"
    bundle = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=bundle.content)
    assert restored.status_code == 200, restored.text
    assert restored.json()["active_map"] == "DTI FA"


def test_pipeline_registration_projects_integer_masks_without_applying_transform_twice():
    w = series([1, 2, 3])
    roi_command(w, "create", {"name": "Label", "tool": "rect", "bounds": [3, 2, 2, 2], "label": 17})
    source = np.asarray(w.session.image_array).copy()
    from mrax.gui.services.analysis_workflows import run_operation
    from mrax.gui.services.pipeline import project_roi

    run_operation(
        w,
        "register",
        {
            "manual": True,
            "manual_settings": {"shift_x": 1},
            "reference_frame": 0,
            "moving_frame": 0,
        },
        lambda *args: None,
    )
    assert w.session.visual_state.transform_key is None
    roi = project_roi(w, w.session.rois[0])
    assert set(np.unique(roi.mask)) == {0, 17}
    assert np.count_nonzero(roi.mask) == 4
    np.testing.assert_array_equal(w.session.original_image_array, source)
    np.testing.assert_array_equal(np.asarray(w.session.image_array)[:, :, :-1], source[:, :, 1:])


@pytest.mark.parametrize(
    "options",
    [
        {"method": "gaussian", "parameters": {"sigma": -1}},
        {"method": "gaussian", "parameters": {"unknown": 1}},
        {"method": "median", "axes": ["frame", "missing"]},
        {"method": "baseline", "axis": "x"},
        {"method": "normalize", "reference_experiment": "Missing"},
        {"method": "gaussian", "roi_names": []},
    ],
)
def test_invalid_processing_does_not_change_inputs_or_create_results(options):
    w = series([1, 2, 3])
    source = np.asarray(w.session.image_array).copy()
    with pytest.raises(ValueError):
        run_processing(w, {"input": "image", **options}, lambda *args: None)
    assert w.session.selected_experiment == "Series" and len(w.session.experiment_names) == 2
    np.testing.assert_array_equal(w.session.image_array, source)


def test_processing_completion_preserves_browsing_and_keeps_the_new_scene(client):
    w = create(client)
    w = action(client, w, "roi.create", name="Mask", tool="rect", bounds=[2, 2, 4, 4])
    w = view(client, w, scene={"hidden_rois": ["Mask"], "fit_rois": ["Mask"]})
    response = client.post(
        f"/api/workspaces/{w['id']}/tasks",
        json={
            "revision": w["revision"],
            "operation": "process",
            "options": {"method": "gaussian", "axes": ["x", "y"], "name": "Filtered"},
        },
    )
    assert response.status_code == 200
    w = view(client, w, frame=2, scene={"roi_opacity": 0.6})
    w, _ = finish(client, w, response.json())
    assert w["experiment"] == "T2 phantom" and w["frame"] == 2 and w["scene"]["roi_opacity"] == 0.6
    assert len(w["experiments"]) == 1 and len(w["pipeline"]["steps"]) == 1
    assert w["scene"]["hidden_rois"] == ["Mask"] and w["scene"]["fit_rois"] == ["Mask"]
    assert w["scene"]["layers"] == []


def test_processing_mask_snapshot_survives_later_roi_edits_and_save(tmp_path):
    w = series([2, 4, 6])
    mask = np.zeros((6, 7), dtype=np.uint16)
    mask[1:3, 2:4] = 9
    w.session.add_roi(ROI("Mask", mask, dims=("x", "y")))
    run(w, "normalize", reference_experiment="Series", roi_names=["Mask"])
    name = w.session.selected_experiment
    roi_command(w, "remove", {"name": "Mask"})
    target = tmp_path / "processing.mrax"
    save_workspace_bundle(w, target)
    restored = load_workspace_bundle(target, tmp_path / "restored", key="restored")
    assert restored.session.selected_experiment == name
    np.testing.assert_array_equal(restored.pipelines[name]["steps"][0]["mask"]["mask"], mask != 0)
    assert restored.session.current_experiment["extra_metadata"]["image_unit"] == "1"


def test_registered_overlay_remains_available_after_processing_its_reference():
    w = series([1, 2, 3])
    run_alignment(
        w,
        {"manual": True, "manual_settings": {}, "reference_frame": 0, "moving_frame": 0},
        lambda *args: None,
    )
    run(w, "gaussian", axes=["x", "y"])
    scene = scene_info(w, experiment_catalog(w))
    layer = next(layer for layer in scene["layers"] if layer["kind"] == "registration")
    assert layer["visible"] and layer["unavailable"] is None
