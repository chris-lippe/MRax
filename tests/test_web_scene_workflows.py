"""Shared display/analysis workflows, using independent synthetic studies only."""

import numpy as np
import pytest

from tests import test_web_analysis_workflows as workflow

browser = workflow.browser
action = workflow.action
job = workflow.job
new = workflow.new
pixels = workflow.pixels
view = workflow.view


def overlay_pixels(client, workspace, layer):
    response = client.get(f"/api/workspaces/{workspace['id']}/slice", params={"layer": layer})
    assert response.status_code == 200, response.text
    assert int(response.headers["x-view-revision"]) == workspace["view_revision"]
    return np.frombuffer(response.content, dtype="<f8").reshape(workspace["shape"])


def test_layers_and_fit_mask_union_survive_edits_and_bundle_round_trip(browser):
    client, _ = browser
    w = new(client)
    original = pixels(client, w).copy()
    w = action(client, w, "roi.create", name="Anterior", tool="rect", bounds=[2, 2, 4, 4])
    w = action(client, w, "roi.create", name="Posterior", tool="rect", bounds=[12, 8, 5, 3])
    revision = w["revision"]
    w = view(
        client,
        w,
        selected_roi="Posterior",
        scene={
            "hidden_rois": ["Anterior"],
            "fit_rois": ["Anterior", "Posterior"],
            "layers": [
                {
                    "id": "echo",
                    "experiment": w["experiment"],
                    "kind": "image",
                    "frame": 1,
                    "opacity": 0.3,
                }
            ],
        },
    )
    assert w["revision"] == revision
    assert not next(r for r in w["rois"] if r["name"] == "Anterior")["visible"]
    np.testing.assert_array_equal(pixels(client, w), original)
    echo = overlay_pixels(client, w, "echo")
    other_frame = view(client, w, frame=1)
    np.testing.assert_array_equal(echo, pixels(client, other_frame))
    w = view(client, other_frame, frame=0)
    w, _ = job(
        client,
        w,
        "full_fit",
        roi_names=w["scene"]["fit_rois"],
        use_roi=True,
        settings={"num_steps": 600},
    )
    w = view(client, w, view="Map", map_name="t2")
    fitted = pixels(client, w)
    assert np.median(fitted[2:6, 2:6]) == pytest.approx(80, abs=0.1)
    assert np.median(fitted[8:11, 12:17]) == pytest.approx(80, abs=0.1)
    assert fitted[0, 0] == 0
    assert w["fit_info"]["roi_names"] == ["Anterior", "Posterior"]
    w = view(
        client,
        w,
        view="Image",
        scene={
            "layers": [
                {
                    "id": "t2",
                    "experiment": w["experiment"],
                    "kind": "map",
                    "map_name": "t2",
                    "colormap": "hot",
                    "window": {"center": 80, "width": 40},
                },
            ]
        },
    )
    np.testing.assert_array_equal(overlay_pixels(client, w, "t2"), fitted)
    w = action(client, w, "roi.rename", name="Anterior", new_name="Cortex")
    assert w["scene"]["hidden_rois"] == ["Cortex"]
    assert w["scene"]["fit_rois"] == ["Cortex", "Posterior"]
    w = action(client, w, "roi.undo")
    assert w["scene"]["hidden_rois"] == ["Anterior"]
    assert w["scene"]["fit_rois"] == ["Anterior", "Posterior"]
    bundle = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=bundle.content)
    assert restored.status_code == 200, restored.text
    r = restored.json()
    assert r["scene"] == w["scene"] and r["selected_roi"] == "Posterior"
    np.testing.assert_array_equal(overlay_pixels(client, r, "t2"), fitted)


def test_registration_joins_reference_scene_and_can_follow_sequence_frames(browser):
    client, _ = browser
    w = view(client, new(client, "multimodal"), experiment="Motion shifted", frame=3)
    w, result = job(
        client,
        w,
        "register",
        reference_experiment="Motion reference",
        moving_experiment="Motion shifted",
        reference_frame=0,
        moving_frame=3,
        manual=True,
        manual_settings={"shift_x": 3},
    )
    assert w["experiment"] == "Motion shifted" and w["frame"] == 3
    assert not w["scene"]["layers"]
    w = view(client, w, experiment="Motion reference", frame=0)
    layer = w["scene"]["layers"][0]
    assert layer["registration_key"] == result["summary"]["key"] and layer["unavailable"] is None
    aligned = overlay_pixels(client, w, layer["id"])
    np.testing.assert_allclose(aligned[:, :-3], pixels(client, w)[:, :-3], atol=1e-6)
    assert not np.any(aligned[:, -3:])  # No acquired support outside the transformed field of view.
    layer = {k: v for k, v in layer.items() if k not in {"label", "unavailable"}}
    layer["frame"] = None
    w = view(client, w, frame=1, scene={"layers": [layer]})
    followed = overlay_pixels(client, w, layer["id"])
    # The frame-3 transform shifts frame 1 left by two columns on the reference grid.
    reference = pixels(client, w)
    np.testing.assert_allclose(followed[:, :-3], reference[:, 2:-1], atol=1e-6)
    w = view(
        client, w, experiment="Complex coils", view="MRax reconstruction", plane="coronal", frame=1
    )
    w = view(client, w, experiment="Motion reference")
    assert w["view"] == "Image" and w["frame"] == 1
    assert w["scene"]["layers"][0]["frame"] is None
    w = view(client, w, experiment="Complex coils")
    assert w["view"] == "MRax reconstruction" and w["plane"] == "coronal" and w["frame"] == 1


@pytest.mark.parametrize(
    "scene",
    [
        {"layers": None},
        {"hidden_rois": ["Missing"]},
        {"fit_rois": ["Missing"]},
        {"roi_opacity": -0.1},
        {"layers": [{"id": "missing", "experiment": "Missing", "kind": "image"}]},
        {
            "layers": [
                {
                    "id": "badwindow",
                    "experiment": "T2 phantom",
                    "kind": "image",
                    "window": {"center": 0, "width": 0},
                }
            ]
        },
    ],
)
def test_invalid_scene_updates_leave_workspace_unchanged(browser, scene):
    client, _ = browser
    w = new(client)
    response = client.patch(
        f"/api/workspaces/{w['id']}/view", json={"revision": w["revision"], "scene": scene}
    )
    assert response.status_code == 422, response.text
    assert client.get(f"/api/workspaces/{w['id']}").json() == w


def test_incompatible_layers_stay_explicit_and_do_not_break_base_navigation(browser):
    client, _ = browser
    w = new(client, "multimodal")
    original = pixels(client, w).copy()
    w = view(
        client,
        w,
        scene={"layers": [{"id": "other", "experiment": "Motion shifted", "kind": "image"}]},
    )
    assert "grids differ" in w["scene"]["layers"][0]["unavailable"]
    assert client.get(f"/api/workspaces/{w['id']}/slice?layer=other").status_code == 422
    np.testing.assert_array_equal(pixels(client, w), original)
    w = view(client, w, frame=2)
    assert w["frame"] == 2 and w["scene"]["layers"][0]["unavailable"]


def test_roi_removal_and_undo_preserve_independent_selections(browser):
    client, _ = browser
    w = action(client, new(client), "roi.create", name="Mask", tool="rect", bounds=[2, 2, 4, 4])
    w = view(client, w, scene={"hidden_rois": ["Mask"], "fit_rois": ["Mask"]})
    w = action(client, w, "roi.remove", name="Mask")
    assert w["scene"]["hidden_rois"] == [] and w["scene"]["fit_rois"] == []
    w = action(client, w, "roi.undo")
    assert w["scene"]["hidden_rois"] == ["Mask"] and w["scene"]["fit_rois"] == ["Mask"]
    assert w["selected_roi"] == "Mask" and not w["rois"][0]["visible"]
