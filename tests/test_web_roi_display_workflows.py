"""ROI appearance and native export previews on independent synthetic arrays."""

from types import SimpleNamespace

import numpy as np
import pytest

from mrax.analysis import ROI
from mrax.gui.masks import mask_slice_for_display
from mrax.gui.services.export import _export_object_records_for_session, _write_single_export_object
from mrax.gui.services.mask_preview import native_mask_preview
from mrax.gui.services.persistence import load_workspace_bundle
from mrax.io.nifti import load_nifti
from tests import test_web_analysis_workflows as workflow
from tests.test_web_workflows import wait_job

browser = workflow.browser
action, new, view, pixels = workflow.action, workflow.new, workflow.view, workflow.pixels


def decode_preview(payload):
    result = np.zeros(payload["plane_shape"], dtype=np.uint16)
    for row, column, length, value in payload["runs"]:
        result[row, column : column + length] = value
    return result


@pytest.mark.parametrize("value", [-1, 1.25, np.nan, 70000])
def test_native_preview_rejects_values_that_cannot_be_shown_as_label_masks(value):
    mask = np.array([[0, value], [0, 0]])
    original = mask.copy()
    session = SimpleNamespace(shared_rois=[ROI("Invalid labels", mask, dims=("x", "y"))])
    with pytest.raises(ValueError, match="integer label"):
        native_mask_preview(session, "Invalid labels")
    np.testing.assert_array_equal(mask, original)


def test_roi_appearance_and_conversion_keep_voxels_images_and_saved_fit(browser):
    client, _ = browser
    w = action(
        client, new(client), "roi.create", name="Ellipse", tool="ellipse", bounds=[2, 3, 8, 7]
    )
    original = pixels(client, w).copy()
    before_mask = w["rois"][0]["mask"]
    response = client.post(
        f"/api/workspaces/{w['id']}/preview-fit",
        json={"revision": w["revision"], "roi_name": "Ellipse", "steps": 200},
    )
    assert response.status_code == 200
    assert wait_job(client, response.json()["id"])["status"] == "completed"
    w = client.get(f"/api/workspaces/{w['id']}").json()
    preview = w["preview"]
    revision = w["revision"]
    styles = {"Ellipse": {"color": "#c462ee", "opacity": 0.7, "label_colors": {"1": "#ee4488"}}}
    w = view(client, w, scene={"roi_styles": styles})
    assert w["revision"] == revision and w["preview"] == preview
    assert w["rois"][0]["mask"] == before_mask
    w = action(client, w, "roi.consolidate", name="Ellipse")
    assert w["rois"][0]["mask"] == before_mask
    assert w["rois"][0]["metadata"]["consolidated"]
    assert "bounds" not in w["rois"][0]["metadata"]
    assert w["preview"] == preview
    np.testing.assert_array_equal(pixels(client, w), original)
    w = view(client, w, scene={"roi_styles": {"Ellipse": {"color": "#c462ee", "opacity": 0.4}}})
    w = action(client, w, "roi.undo")
    assert w["rois"][0]["metadata"]["kind"] == "ellipse"
    assert w["rois"][0]["mask"] == before_mask
    assert w["scene"]["roi_styles"]["Ellipse"]["opacity"] == 0.4
    w = action(client, w, "roi.rename", name="Ellipse", new_name="Renamed")
    assert w["scene"]["roi_styles"]["Renamed"]["color"] == "#c462ee"
    w = action(client, w, "roi.undo")
    assert w["scene"]["roi_styles"]["Ellipse"]["opacity"] == 0.4
    assert "Renamed" not in w["scene"]["roi_styles"]
    w = action(client, w, "roi.rename", name="Ellipse", new_name="Renamed again")
    w = view(client, w, scene={"roi_styles": {}})
    w = action(client, w, "roi.undo")
    assert not w["scene"]["roi_styles"]


@pytest.mark.parametrize(
    "styles",
    [
        {"ROI": {"color": "red"}},
        {"ROI": {"opacity": 1.01}},
        {"ROI": {"label_colors": {"0": "#aa55ee"}}},
        {"ROI": {"label_colors": {"01": "#aa55ee"}}},
        {"Missing": {"color": "#aa55ee"}},
    ],
)
def test_invalid_appearance_does_not_change_workspace(browser, styles):
    client, _ = browser
    w = action(client, new(client), "roi.create", name="ROI", bounds=[2, 3, 5, 6])
    response = client.patch(
        f"/api/workspaces/{w['id']}/view",
        json={"revision": w["revision"], "scene": {"roi_styles": styles}},
    )
    assert response.status_code in {400, 422}
    unchanged = client.get(f"/api/workspaces/{w['id']}").json()
    assert unchanged["view_revision"] == w["view_revision"]
    assert unchanged["scene"]["roi_styles"] == {}


def test_native_preview_matches_nifti_labels_and_survives_bundle_reload(browser, tmp_path):
    client, _ = browser
    w = action(client, new(client), "roi.create", name="Label seven", bounds=[2, 3, 5, 6], label=7)
    w = action(client, w, "roi.copy_slice", name="Label seven", axis="z", source=0)
    w = view(client, w, scene={"roi_styles": {"Label seven": {"color": "#3196dd", "opacity": 0}}})
    revision = w["revision"]
    saved = client.post(f"/api/workspaces/{w['id']}/save")
    path = tmp_path / "synthetic-roi.mrax"
    path.write_bytes(saved.content)
    restored = load_workspace_bundle(path, tmp_path / "restored", key="restored")
    assert (
        restored.display_settings[w["experiment"]]["scene"]["roi_styles"]["Label seven"]["opacity"]
        == 0
    )
    record = next(
        r for r in _export_object_records_for_session(restored.session) if r["kind"] == "roi"
    )
    export = _write_single_export_object(record, tmp_path / "synthetic-mask.nii", "nifti")[0]
    exported, _ = load_nifti(export)
    roi = restored.session.shared_rois[0]
    np.testing.assert_array_equal(exported, roi.mask)
    for axes in [roi.dims[:2], (roi.dims[0], roi.dims[2]), roi.dims[1:]]:
        slice_dim = next(dim for dim in roi.dims if dim not in axes)
        response = client.get(
            f"/api/workspaces/{w['id']}/roi-mask",
            params={
                "name": "Label seven",
                "axes": ",".join(axes),
                "slice_index": 0,
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        expected = mask_slice_for_display(
            exported, dims=roi.dims, plane_dims=axes, indices={slice_dim: 0}
        )
        np.testing.assert_array_equal(decode_preview(data), expected)
        assert data["shape"] == list(roi.mask.shape)
    assert client.get(f"/api/workspaces/{w['id']}").json()["revision"] == revision
    response = client.get(
        f"/api/workspaces/{w['id']}/roi-mask", params={"name": "Label seven", "slice_index": 999}
    )
    assert response.status_code == 422


def test_native_preview_does_not_apply_display_registration(browser):
    client, _ = browser
    w = view(client, new(client, "multimodal"), experiment="Motion shifted", frame=3)
    w = action(client, w, "roi.create", name="Registered ROI", bounds=[6, 6, 5, 5])
    path = f"/api/workspaces/{w['id']}/roi-mask"
    original = client.get(path, params={"name": "Registered ROI"}).json()
    w, _ = workflow.job(
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
    preview = client.get(path, params={"name": "Registered ROI"}).json()
    assert preview == original
    native_mask = decode_preview(preview)
    w = action(client, w, "roi.consolidate", name="Registered ROI")
    np.testing.assert_array_equal(
        decode_preview(client.get(path, params={"name": "Registered ROI"}).json()), native_mask
    )
