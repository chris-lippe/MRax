"""Scientific and file contracts using independent synthetic arrays."""

from __future__ import annotations

import json

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


def test_indexed_composition_preserves_sources_and_label_identity(browser):
    client, _ = browser
    w = new(client)
    for name, bounds in [("A", [2, 2, 4, 4]), ("B", [4, 4, 4, 4])]:
        w = action(client, w, "roi.create", name=name, tool="rect", bounds=bounds, label=1)
    originals = {name: roi_array(client, w, name) for name in ["A", "B"]}
    w = action(
        client,
        w,
        "roi.compose",
        name="Joint",
        sources=["A", "B"],
        output="indexed",
        overlap="first",
    )
    assert w["selected_roi"] == "Joint" and len(w["rois"]) == 3
    joint = roi_array(client, w, "Joint")
    assert {int(v): int(np.sum(joint == v)) for v in np.unique(joint) if v} == {1: 16, 2: 12}
    for name, mask in originals.items():
        np.testing.assert_array_equal(roi_array(client, w, name), mask)
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
        client,
        w,
        "roi.create",
        name="Painted",
        tool="brush",
        points=[[3, 3], [6, 4]],
        size=3,
        label=7,
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
        client,
        w,
        "roi.brush",
        name="Labels",
        points=[[8, 8], [12, 8]],
        size=8,
        label=7,
        label_only=True,
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
