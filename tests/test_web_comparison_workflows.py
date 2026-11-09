"""Read-only comparisons of independently generated images and pipeline states."""

import copy
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mrax.data_struct import SpatialGeometry
from mrax.gui.services.analysis_workflows import run_operation
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.pipeline import change_pipeline, snapshot
from mrax.gui.services.processing import run_processing
from mrax.gui.services.scene import scene_settings, update_scene
from mrax.gui.services.synthetic import synthetic_multimodal_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import load_workspace_bundle, save_workspace_bundle
from mrax.gui.session import GuiSession
from tests import test_web_analysis_workflows as workflow

browser = workflow.browser


def comparison(client, workspace):
    response = client.get(f"/api/workspaces/{workspace['id']}/comparison")
    assert response.status_code == 200, response.text
    assert int(response.headers["x-view-revision"]) == workspace["view_revision"]
    metadata = json.loads(response.headers["x-comparison"])
    data = np.frombuffer(response.content, dtype="<f8")
    count = np.prod(metadata["planes"][0]["shape"])
    return [
        data[:count].reshape(metadata["planes"][0]["shape"]),
        data[count:].reshape(metadata["planes"][1]["shape"]),
    ], metadata


def synthetic():
    data = np.random.default_rng(19).normal(10, 2, (3, 7, 9)).astype(np.float32)
    session = GuiSession()
    session.create_empty_study("Independent comparison")
    session.add_experiment("Experiment", data, dims={"frame": 3, "x": 7, "y": 9})
    return StudyWorkspace(key="comparison", label=session.study_name, session=session), data


def test_original_and_cached_processed_comparison_does_not_toggle_the_pipeline(browser):
    client, _ = browser
    w = workflow.new(client)
    source = workflow.pixels(client, w).copy()
    w, _ = workflow.job(
        client, w, "process", method="gaussian", axes=["x", "y"], parameters={"sigma": 1.2}
    )
    processed = workflow.pixels(client, w).copy()
    config = {**w["scene"]["comparison"], "enabled": True, "position": 0.37}
    revision = w["revision"]
    w = workflow.view(client, w, scene={"comparison": config})
    before = copy.deepcopy(w["pipeline"])
    arrays, metadata = comparison(client, w)
    np.testing.assert_array_equal(arrays[0], source)
    np.testing.assert_array_equal(arrays[1], processed)
    assert metadata["geometry_compatible"]
    assert w["pipeline"] == before and w["revision"] == revision
    w, _ = workflow.job(client, w, "pipeline", action="enabled", enabled=False)
    arrays, _ = comparison(client, w)
    np.testing.assert_array_equal(arrays[0], source)
    np.testing.assert_array_equal(arrays[1], processed)
    np.testing.assert_array_equal(workflow.pixels(client, w), source)
    assert not w["pipeline"]["enabled"]
    assert len(w["experiments"]) == 1


def test_arbitrary_sources_frames_and_overlays_survive_save_and_reopen(browser):
    client, _ = browser
    w = workflow.new(client)
    original = workflow.pixels(client, w)
    w = workflow.view(client, w, frame=2)
    other_frame = workflow.pixels(client, w)
    config = {
        **w["scene"]["comparison"],
        "enabled": True,
        "layout": "side-by-side",
        "position": 0.68,
        "linked_window": False,
    }
    config["left"].update(frame=0, colormap="gray", window={"center": 600000, "width": 100000})
    config["right"].update(frame=2, colormap="hot")
    layer = {
        "id": "echo",
        "experiment": w["experiment"],
        "kind": "image",
        "frame": 1,
        "opacity": 0.25,
    }
    w = workflow.view(client, w, scene={"comparison": config, "layers": [layer]})
    arrays, metadata = comparison(client, w)
    np.testing.assert_array_equal(arrays[0], original)
    np.testing.assert_array_equal(arrays[1], other_frame)
    assert [plane["frame"] for plane in metadata["planes"]] == [0, 2]
    saved = client.post(f"/api/workspaces/{w['id']}/save")
    response = client.post("/api/workspaces/upload", content=saved.content)
    assert response.status_code == 200, response.text
    restored = response.json()
    assert restored["scene"] == w["scene"]
    restored_arrays, _ = comparison(client, restored)
    for actual, expected in zip(restored_arrays, arrays):
        np.testing.assert_array_equal(actual, expected)
    restored = workflow.view(
        client,
        restored,
        scene={"comparison": {**restored["scene"]["comparison"], "enabled": False}},
    )
    assert restored["scene"]["layers"][0]["opacity"] == 0.25


@pytest.mark.parametrize("mode", ["split", "overlay"])
def test_native_grid_mismatch_uses_side_by_side_without_resampling(mode):
    w, source = synthetic()
    other = np.arange(4 * 5, dtype=np.float32).reshape(1, 4, 5)
    w.session.add_experiment("Other grid", other, dims={"frame": 1, "x": 4, "y": 5})
    w.session.select_experiment("Experiment")
    w.session.set_active_frame(2)
    config = scene_settings(w)["comparison"]
    config.update(enabled=True, mode=mode, right={**config["right"], "experiment": "Other grid"})
    update_scene(w, {"comparison": config})
    arrays, info = render_comparison(w)
    np.testing.assert_array_equal(arrays[0], source[2])
    np.testing.assert_array_equal(arrays[1], other[0])
    assert not info["geometry_compatible"]
    assert "register" in info["geometry_message"].lower()
    assert info["planes"][1]["frame"] == 0
    assert w.session.selected_experiment == "Experiment" and w.session.active_frame == 2


@pytest.mark.parametrize("mode", ["split", "overlay"])
def test_same_shape_with_different_physical_geometry_cannot_use_swipe_or_overlay(mode):
    w, source = synthetic()
    first = SpatialGeometry(voxel_sizes=(1.0, 1.0), affine=np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 4
    second = SpatialGeometry(voxel_sizes=(1.0, 1.0), affine=shifted)
    w.session.add_experiment(
        "Reference",
        source,
        dims={"frame": 3, "x": 7, "y": 9},
        extra_metadata={"spatial_geometry": first.to_metadata()},
    )
    w.session.add_experiment(
        "Displaced",
        source,
        dims={"frame": 3, "x": 7, "y": 9},
        extra_metadata={"spatial_geometry": second.to_metadata()},
    )
    w.session.select_experiment("Reference")
    config = scene_settings(w)["comparison"]
    config["mode"] = mode
    config["right"]["experiment"] = "Displaced"
    update_scene(w, {"comparison": config})
    arrays, info = render_comparison(w)
    assert arrays[0].shape == arrays[1].shape
    assert not info["geometry_compatible"]


def test_overlay_pairs_two_other_experiments_and_preserves_active_data_and_view(tmp_path):
    workspace, active = synthetic()
    first = active + 5
    second = active * 2
    for name, data in [("Image A", first), ("Image B", second)]:
        workspace.session.add_experiment(name, data, dims={"frame": 3, "x": 7, "y": 9})
    workspace.session.select_experiment("Experiment")
    workspace.session.set_active_frame(2)
    config = scene_settings(workspace)["comparison"]
    config.update(enabled=True, mode="overlay", opacity=0.35, layout="side-by-side")
    config["left"].update(experiment="Image A", frame=0)
    config["right"].update(experiment="Image B", frame=1)
    update_scene(workspace, {"comparison": config})
    arrays, metadata = render_comparison(workspace)
    assert metadata["geometry_compatible"] and metadata["geometry_message"] is None
    np.testing.assert_array_equal(arrays[0], first[0])
    np.testing.assert_array_equal(arrays[1], second[1])
    np.testing.assert_array_equal(workspace.session.image_array, active)
    assert workspace.session.selected_experiment == "Experiment"
    assert workspace.session.active_frame == 2
    assert not workspace.pipelines and not workspace.fit_results_by_experiment
    pair = copy.deepcopy(config)
    update_scene(workspace, {"comparison": {**config, "mode": "split"}})
    split = scene_settings(workspace)["comparison"]
    assert split["left"] == pair["left"] and split["right"] == pair["right"]
    assert split["opacity"] == 0.35 and split["layout"] == "side-by-side"
    update_scene(workspace, {"comparison": {**split, "mode": "overlay"}})
    bundle = save_workspace_bundle(workspace, tmp_path / "overlay.mrax")
    restored = load_workspace_bundle(bundle, tmp_path / "restored", key="restored")
    assert scene_settings(restored)["comparison"] == pair
    restored_arrays, _ = render_comparison(restored)
    for expected, actual in zip(arrays, restored_arrays):
        np.testing.assert_array_equal(actual, expected)


def test_pair_cursor_can_navigate_other_volumes_without_changing_active_experiment():
    workspace, active = synthetic()
    volume = np.arange(5 * 4 * 6 * 8, dtype=np.float32).reshape(5, 4, 6, 8)
    for name, values in [("Volume A", volume), ("Volume B", volume + 100)]:
        workspace.session.add_experiment(name, values, dims={"frame": 5, "x": 4, "y": 6, "z": 8})
    workspace.session.select_experiment("Experiment")
    indices = dict(workspace.session.active_indices)
    config = scene_settings(workspace)["comparison"]
    config.update(
        enabled=True, mode="overlay", frame=4, plane="axial", indices={"x": 0, "y": 0, "z": 7}
    )
    config["left"].update(experiment="Volume A")
    config["right"].update(experiment="Volume B")
    update_scene(workspace, {"comparison": config})
    arrays, metadata = render_comparison(workspace)
    assert metadata["geometry_compatible"]
    np.testing.assert_array_equal(arrays[0], volume[4, :, :, 7])
    np.testing.assert_array_equal(arrays[1], volume[4, :, :, 7] + 100)
    assert workspace.session.selected_experiment == "Experiment"
    assert workspace.session.active_frame == 0
    assert workspace.session.active_indices == indices
    np.testing.assert_array_equal(workspace.session.image_array, active)


@pytest.mark.parametrize(
    "patch", [{"mode": "unknown"}, {"opacity": -0.1}, {"opacity": 1.1}, {"opacity": float("nan")}]
)
def test_pair_display_mode_and_opacity_are_validated(patch):
    workspace, _ = synthetic()
    with pytest.raises(ValueError):
        update_scene(
            workspace, {"comparison": {**scene_settings(workspace)["comparison"], **patch}}
        )


def test_original_and_processed_remain_independent_after_processing_and_bundle_reload(tmp_path):
    w, source = synthetic()
    run_processing(
        w, {"method": "gaussian", "axes": ["x", "y"], "parameters": {"sigma": 1}}, lambda *_: None
    )
    expected = gaussian_filter(source, sigma=(0, 1, 1), mode="mirror")
    change_pipeline(w, {"action": "enabled", "enabled": False}, lambda *_: None)
    config = scene_settings(w)["comparison"]
    config.update(enabled=True, position=0.71)
    config["left"]["frame"] = 1
    config["right"]["frame"] = 1
    update_scene(w, {"comparison": config})
    saved = save_workspace_bundle(w, tmp_path / "comparison.mrax")
    restored = load_workspace_bundle(saved, tmp_path / "reopen", key="restored")
    arrays, _ = render_comparison(restored)
    np.testing.assert_array_equal(arrays[0], source[1])
    np.testing.assert_allclose(arrays[1], expected[1], atol=3e-6)
    np.testing.assert_array_equal(restored.session.image_array, source)
    assert not restored.pipelines["Experiment"]["enabled"]


def test_complex_components_reconstruction_and_anatomical_planes_are_comparable():
    session = synthetic_multimodal_session()
    session.select_experiment("Complex coils")
    session.set_image_plane("coronal")
    session.set_active_index(session.image_slice_dim, 5)
    session.set_active_frame(1)
    expected_real = np.real(session.frame_image())
    expected_phase = session.reconstruction_image(
        component="phase", combined=True, unwrap_phase=False
    )
    w = StudyWorkspace(key="complex", label=session.study_name, session=session)
    config = scene_settings(w)["comparison"]
    config["left"]["component"] = "real"
    config["right"].update(kind="reconstruction", component="phase")
    update_scene(w, {"comparison": config})
    arrays, info = render_comparison(w)
    np.testing.assert_array_equal(arrays[0], expected_real)
    np.testing.assert_array_equal(arrays[1], expected_phase)
    assert info["planes"][0]["plane_dims"] == list(session.image_plane_dims)
    assert info["planes"][1]["unit"] == "rad" and not info["same_units"]
    assert info["geometry_compatible"]
    assert session.active_frame == 1 and session.active_image_plane == "coronal"


def test_retained_fit_maps_and_original_keep_native_grids_after_registration():
    w, source = synthetic()
    fitted = np.arange(63, dtype=np.float32).reshape(7, 9)
    w.session.replace_fitted_maps({"t2": fitted})
    w.session.current_experiment["fitted_map_units"] = {"t2": "ms"}
    state = snapshot(w.session)
    grid = {key: state[key] for key in ("dims", "geometry", "unit")}
    w.fit_results_by_experiment["Experiment"] = {
        "fields": w.session.fitted_maps,
        "pipeline_grid": grid,
        "pipeline_steps": [],
    }
    w.session.current_experiment["fit_grid"] = grid
    w.session.add_experiment("Reference", np.ones((1, 4, 5)), dims={"frame": 1, "x": 4, "y": 5})
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
    arrays, info = render_comparison(w)
    np.testing.assert_array_equal(arrays[0], source[0])
    assert arrays[0].shape == (7, 9) and arrays[1].shape == (4, 5)
    assert not info["geometry_compatible"]
    config = scene_settings(w)["comparison"]
    config["left"].update(kind="map", map_name="t2")
    update_scene(w, {"comparison": config})
    arrays, info = render_comparison(w)
    np.testing.assert_array_equal(arrays[0], fitted)
    assert arrays[1].shape == (4, 5)
    assert info["planes"][0]["unit"] == "ms" and not info["same_units"]
    assert w.session.image_array.shape == (3, 4, 5)
    np.testing.assert_array_equal(w.session.fitted_maps["t2"], fitted)


def test_comparison_sources_follow_experiment_renaming(browser):
    client, _ = browser
    w = workflow.new(client)
    w = workflow.view(
        client, w, scene={"comparison": {**w["scene"]["comparison"], "enabled": True}}
    )
    expected, _ = comparison(client, w)
    w = workflow.action(client, w, "experiment.rename", name="Renamed experiment")
    assert w["scene"]["comparison"]["left"]["experiment"] == "Renamed experiment"
    assert w["scene"]["comparison"]["right"]["experiment"] == "Renamed experiment"
    actual, _ = comparison(client, w)
    for values, original in zip(actual, expected):
        np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize(
    "patch",
    [
        {"position": 1.2},
        {"layout": "unknown"},
        {"frame": 100},
        {"indices": {"z": 100}},
        {"indices": {"unknown": 0}},
        {"left": {"experiment": "Missing", "kind": "original"}},
        {"right": {"experiment": "Synthetic T2", "kind": "processed", "frame": -1}},
    ],
)
def test_invalid_comparison_settings_are_transactional(browser, patch):
    client, _ = browser
    w = workflow.new(client)
    response = client.patch(
        f"/api/workspaces/{w['id']}/view",
        json={
            "revision": w["revision"],
            "scene": {"comparison": {**w["scene"]["comparison"], **patch}},
        },
    )
    assert response.status_code in {400, 422}
    current = client.get(f"/api/workspaces/{w['id']}").json()
    assert current["scene"] == w["scene"] and current["view_revision"] == w["view_revision"]
