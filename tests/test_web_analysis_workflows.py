"""Scientific and file contracts using independent synthetic arrays."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.web.app import create_app


@pytest.fixture
def browser(tmp_path):
    source = tmp_path / "connected"
    source.mkdir()
    with TestClient(
        create_app(
            read_roots=[source],
            write_roots=[source],
            runtime_dir=tmp_path,
            filesystem_scope="shared",
        ),
        base_url="http://localhost",
    ) as client:
        client.headers["x-mrax-token"] = client.get("/api/session").json()["token"]
        yield client, source


def new(client, kind="t2"):
    response = client.post("/api/workspaces/synthetic", params={"kind": kind})
    assert response.status_code == 200, response.text
    return response.json()


def view(client, workspace, **values):
    response = client.patch(
        f"/api/workspaces/{workspace['id']}/view",
        json={"revision": workspace["revision"], **values},
    )
    assert response.status_code == 200, response.text
    return response.json()


def action(client, workspace, operation, **options):
    response = client.post(
        f"/api/workspaces/{workspace['id']}/actions",
        json={
            "revision": workspace["revision"],
            "view_revision": workspace["view_revision"],
            "action": operation,
            "options": options,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def job(client, workspace, operation, **options):
    response = client.post(
        f"/api/workspaces/{workspace['id']}/tasks",
        json={"revision": workspace["revision"], "operation": operation, "options": options},
    )
    assert response.status_code == 200, response.text
    key = response.json()["id"]
    deadline = time.monotonic() + 80
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{key}").json()
        if result["status"] not in {"queued", "running", "cancelling"}:
            assert result["status"] == "completed", result["detail"]
            return client.get(f"/api/workspaces/{workspace['id']}").json(), result
        time.sleep(0.05)
    pytest.fail("Synthetic operation exceeded its bounded timeout")


def pixels(client, workspace):
    response = client.get(f"/api/workspaces/{workspace['id']}/slice")
    assert response.status_code == 200
    dtype = "<f8" if response.headers["x-dtype"] == "float64-le" else "<f4"
    return np.frombuffer(response.content, dtype=dtype).reshape(
        json.loads(response.headers["x-shape"])
    )


def roi_array(client, workspace, name):
    session = client.app.state.records[workspace["id"]].workspace.session
    return next(np.asarray(r.mask).copy() for r in session.shared_rois if r.name == name)


def test_auto_subtract_covers_shapes_rounds_and_brush_in_one_undo(browser):
    client, _ = browser
    w = new(client)
    w = action(client, w, "roi.create", name="A", tool="rect", bounds=[2, 2, 4, 4], label=7)
    w = action(client, w, "roi.create", name="Inactive", tool="rect", bounds=[2, 2, 4, 4], label=5)
    w = action(client, w, "roi.disable", name="Inactive")
    original = {r["name"]: roi_array(client, w, r["name"]) for r in w["rois"]}
    w = action(
        client,
        w,
        "roi.create",
        name="B",
        tool="rect",
        bounds=[4, 4, 4, 4],
        label=11,
        auto_subtract=True,
    )
    assert w["rois"][0]["pixels"] == 12 and w["rois"][2]["pixels"] == 16
    np.testing.assert_array_equal(roi_array(client, w, "Inactive"), original["Inactive"])
    w = action(client, w, "roi.undo")
    assert [r["name"] for r in w["rois"]] == ["A", "Inactive"]
    for name, mask in original.items():
        np.testing.assert_array_equal(roi_array(client, w, name), mask)
    w = action(client, w, "roi.create", name="B", tool="rect", bounds=[10, 10, 2, 2], label=11)
    before = {r["name"]: roi_array(client, w, r["name"]) for r in w["rois"]}
    for operation, options in [
        ("roi.draw", {"tool": "rect", "bounds": [2, 2, 4, 4], "mode": "add"}),
        ("roi.brush", {"points": [[3, 3], [4, 4]], "size": 2}),
    ]:
        w = action(client, w, operation, name="B", label=11, auto_subtract=True, **options)
        assert np.count_nonzero(roi_array(client, w, "A")) < 16
        assert not np.any((roi_array(client, w, "A") != 0) & (roi_array(client, w, "B") != 0))
        np.testing.assert_array_equal(roi_array(client, w, "Inactive"), before["Inactive"])
        w = action(client, w, "roi.undo")
        for name, mask in before.items():
            np.testing.assert_array_equal(roi_array(client, w, name), mask)


@pytest.mark.parametrize(
    "output,overlap,counts",
    [
        ("binary", "first", {1: 28}),
        ("indexed", "first", {1: 16, 2: 12}),
        ("indexed", "last", {1: 12, 2: 16}),
    ],
)
def test_binary_and_indexed_composition_preserve_sources_and_label_identity(
    browser, output, overlap, counts
):
    client, _ = browser
    w = new(client)
    for name, bounds in [("A", [2, 2, 4, 4]), ("B", [4, 4, 4, 4])]:
        w = action(client, w, "roi.create", name=name, tool="rect", bounds=bounds, label=1)
    originals = {name: roi_array(client, w, name) for name in ["A", "B"]}
    w = action(
        client, w, "roi.compose", name="Joint", sources=["A", "B"], output=output, overlap=overlap
    )
    assert w["selected_roi"] == "Joint" and len(w["rois"]) == 3
    joint = roi_array(client, w, "Joint")
    assert {int(v): int(np.sum(joint == v)) for v in np.unique(joint) if v} == counts
    for name, mask in originals.items():
        np.testing.assert_array_equal(roi_array(client, w, name), mask)
    if output == "indexed":
        assert w["rois"][-1]["metadata"]["label_names"] == {"1": "A", "2": "B"}
        w = action(client, w, "roi.relabel", name="Joint", old=1, label=9)
        assert w["rois"][-1]["metadata"]["label_names"] == {"9": "A", "2": "B"}
        w = action(client, w, "roi.delete_label", name="Joint", label=9)
        assert w["rois"][-1]["labels"] == [2]
        assert w["rois"][-1]["metadata"]["label_names"] == {"2": "B"}


def test_growth_and_selective_label_edits_keep_labels_and_image_values(browser):
    client, _ = browser
    w = view(client, new(client), z=1)
    original = pixels(client, w).copy()
    w = action(
        client, w, "roi.create", name="Painted", tool="brush",
        points=[[3, 3], [6, 4]], size=3, label=7,
    )
    assert w["rois"][0]["metadata"]["kind"] == "mask"
    assert w["rois"][0]["pixels"] > 0 and w["rois"][0]["labels"] == [7]
    w = action(client, w, "roi.undo")
    assert w["rois"] == []
    w = action(client, w, "roi.create", name="Labels", tool="rect", bounds=[8, 8, 1, 1], label=7)
    w = action(
        client, w, "roi.draw", name="Labels", tool="rect", bounds=[12, 8, 1, 1], label=9, mode="add"
    )
    before = roi_array(client, w, "Labels")
    w = action(client, w, "roi.grow", name="Labels", radius=1)
    expanded = roi_array(client, w, "Labels")
    assert np.sum(expanded == 7) == 7 and np.sum(expanded == 9) == 7
    np.testing.assert_array_equal(expanded[before != 0], before[before != 0])
    w = action(
        client, w, "roi.brush", name="Labels", points=[[8, 8], [12, 8]],
        size=8, label=7, label_only=True,
    )
    painted = roi_array(client, w, "Labels")
    np.testing.assert_array_equal(painted[expanded == 9], expanded[expanded == 9])
    assert np.sum(painted == 7) > np.sum(expanded == 7)
    w = action(client, w, "roi.undo")
    w = action(
        client,
        w,
        "roi.brush",
        name="Labels",
        points=[[8, 8], [12, 8]],
        size=8,
        label=7,
        erase=True,
        label_only=True,
    )
    erased = roi_array(client, w, "Labels")
    np.testing.assert_array_equal(erased[expanded == 9], expanded[expanded == 9])
    assert np.sum(erased == 7) < 7
    w = action(client, w, "roi.undo")
    np.testing.assert_array_equal(roi_array(client, w, "Labels"), expanded)
    w = action(client, w, "roi.delete_label", name="Labels", label=9)
    assert w["rois"][0]["labels"] == [7]
    w = action(client, w, "roi.undo")
    np.testing.assert_array_equal(roi_array(client, w, "Labels"), expanded)
    np.testing.assert_array_equal(pixels(client, w), original)


def test_label_appearance_preserves_masks_and_survives_workspace_save(browser):
    client, _ = browser
    w = new(client)
    w = action(client, w, "roi.create", name="Labels", tool="rect", bounds=[2, 2, 4, 4], label=7)
    w = action(
        client, w, "roi.draw", name="Labels", tool="rect", bounds=[8, 2, 4, 4], label=9, mode="add"
    )
    mask = roi_array(client, w, "Labels")
    image = pixels(client, w)
    style = {
        "color": "#fbbf24", "opacity": 0.4,
        "label_colors": {"7": "#cc3355"}, "label_opacities": {"7": 0.75, "9": 0.0},
        "hidden_labels": [9],
    }
    w = view(client, w, scene={"roi_styles": {"Labels": style}})
    assert w["scene"]["roi_styles"]["Labels"] == style
    assert w["rois"][0]["labels"] == [7, 9]
    np.testing.assert_array_equal(roi_array(client, w, "Labels"), mask)
    np.testing.assert_array_equal(pixels(client, w), image)
    preview = client.get(f"/api/workspaces/{w['id']}/roi-mask", params={"name": "Labels"})
    assert preview.status_code == 200 and preview.json()["labels"] == [7, 9]
    saved = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=saved.content)
    assert restored.status_code == 200, restored.text
    loaded = restored.json()
    assert loaded["scene"]["roi_styles"]["Labels"] == style
    np.testing.assert_array_equal(roi_array(client, loaded, "Labels"), mask)
    for invalid in [
        {"label_opacities": {"01": 0.5}}, {"label_opacities": {"7": -0.1}},
        {"label_opacities": {"0": 0.5}}, {"hidden_labels": [9, 9]},
        {"hidden_labels": [0]}, {"hidden_labels": [True]},
    ]:
        response = client.patch(
            f"/api/workspaces/{w['id']}/view",
            json={"revision": w["revision"], "scene": {"roi_styles": {"Labels": invalid}}},
        )
        assert response.status_code in {400, 422}
    np.testing.assert_array_equal(roi_array(client, w, "Labels"), mask)


def test_otsu_settings_control_foreground_regions_holes_and_validation(browser):
    from mrax.gui.services.geometry import _auto_threshold_mask

    image = np.zeros((24, 32))
    image[3:10, 3:10] = 10
    image[5:7, 5:7] = 0
    image[14:17, 20:23] = 10
    image[6, 6] = np.nan
    all_regions, metadata = _auto_threshold_mask(image, largest_component=False, fill_holes=False)
    np.testing.assert_array_equal(all_regions, image == 10)
    filled, _ = _auto_threshold_mask(image)
    assert filled.sum() == 48 and not filled[6, 6] and not filled[15, 21]
    dark, _ = _auto_threshold_mask(
        image, foreground="dark", largest_component=False, fill_holes=False
    )
    np.testing.assert_array_equal(dark, image == 0)
    assert 0 < metadata["threshold"] < 10

    client, _ = browser
    w = new(client)
    original = pixels(client, w).copy()
    settings = dict(foreground="dark", largest_component=False, fill_holes=False, bins=64)
    w = action(client, w, "roi.create", name="Threshold ROI", tool="otsu", **settings)
    assert all(w["rois"][0]["metadata"][key] == value for key, value in settings.items())
    for invalid in ({"bins": 1}, {"bins": 16.5}, {"foreground": "unknown"}, {"fill_holes": "false"}):
        response = client.post(
            f"/api/workspaces/{w['id']}/actions",
            json={
                "revision": w["revision"], "view_revision": w["view_revision"],
                "action": "roi.create", "options": {"name": "Invalid", "tool": "otsu", **invalid},
            },
        )
        assert response.status_code == 422, response.text
        assert len(client.get(f"/api/workspaces/{w['id']}").json()["rois"]) == 1
    np.testing.assert_array_equal(pixels(client, w), original)


def test_roi_export_names_preserve_dots_and_control_multi_file_outputs(browser, tmp_path):
    from mrax.gui.services.export_workflow import export_workspace

    client, _ = browser
    w = new(client)
    w = action(client, w, "roi.create", name="Tissue.v1", tool="rect", bounds=[2, 2, 4, 4])
    workspace = client.app.state.records[w["id"]].workspace
    destination = tmp_path / "exports"
    destination.mkdir()
    sentinel = destination / "existing.txt"
    sentinel.write_text("Keep this file")
    options = {
        "kind": "object", "object": "roi/Tissue.v1", "format": "roi_set",
        "destination": str(destination),
    }
    export_workspace(workspace, options, tmp_path / "default", lambda *args: None)
    assert {path.name for path in destination.iterdir()} == {
        "Tissue.v1.npz", "Tissue.v1.json", "existing.txt",
    }
    export_workspace(
        workspace, {**options, "filename": "label-map.npz"},
        tmp_path / "named", lambda *args: None,
    )
    assert (destination / "label-map.npz").is_file()
    assert json.loads((destination / "label-map.json").read_text())["npz"] == "label-map.npz"
    export_workspace(
        workspace, {**options, "format": "nifti", "filename": "label-map.nii"},
        tmp_path / "nifti", lambda *args: None,
    )
    assert (destination / "label-map.nii").is_file()
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    for index, filename in enumerate(("../outside.npz", "C:\\outside.npz", "CON.npz")):
        with pytest.raises(ValueError, match="file name"):
            export_workspace(
                workspace, {**options, "filename": filename},
                tmp_path / f"invalid-{index}", lambda *args: None,
            )
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
