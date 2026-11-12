"""Read-only comparisons of independently generated images and pipeline states."""

import json

import numpy as np
import pytest

from mrax.data_struct import SpatialGeometry
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.scene import scene_settings, update_scene
from mrax.gui.services.workspace import StudyWorkspace
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


def test_arbitrary_sources_frames_and_overlays_survive_save_and_reopen(browser):
    client, _ = browser
    w = workflow.new(client)
    original = workflow.pixels(client, w)
    w = workflow.view(client, w, frame=2)
    w, _ = workflow.job(
        client, w, "process", method="gaussian", input="image", parameters={"sigma": 1.0}
    )
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
    session = client.app.state.records[w["id"]].workspace.session
    working = session.current_experiment["mr_image"]
    dataset = session.study.datasets[0].data
    arrays, metadata = comparison(client, w)
    assert session.current_experiment["mr_image"] is working
    assert session.study.datasets[0].data is dataset
    np.testing.assert_array_equal(workflow.pixels(client, w), other_frame)
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
