"""Scientific and file contracts using independent synthetic arrays."""

from __future__ import annotations

import io
import json
import time
import zipfile

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


@pytest.mark.parametrize(
    "plane,shape", [("axial", [24, 32]), ("coronal", [24, 3]), ("sagittal", [32, 3])]
)
def test_native_planes_label_values_and_undo(browser, plane, shape):
    client, _ = browser
    w = view(client, new(client), plane=plane)
    assert w["shape"] == shape
    w = action(
        client, w, "roi.create", name="Label seven", tool="ellipse", bounds=[0, 2, 2, 8], label=7
    )
    assert w["rois"][0]["labels"] == [7]
    count = w["rois"][0]["pixels"]
    assert count > 0
    w = action(client, w, "roi.brush", name="Label seven", points=[[0, 3], [1, 5]], size=1, label=9)
    assert 9 in w["rois"][0]["labels"]
    w = action(client, w, "roi.undo")
    assert w["rois"][0]["labels"] == [7] and w["rois"][0]["pixels"] == count
    w = action(client, w, "roi.copy_slice", name="Label seven")
    assert w["rois"][0]["pixels"] == count * w["slice_count"]


@pytest.mark.parametrize(
    "tool,options",
    [
        ("rect", {}),
        ("ellipse", {}),
        ("regular_polygon", {"sides": 5}),
        ("polygon", {"points": [[4, 4], [14, 4], [10, 15]]}),
        ("lasso", {"points": [[4, 4], [14, 4], [14, 14], [4, 14]]}),
        ("wand", {"seed": [12, 10], "tolerance": 0.1}),
        ("otsu", {}),
    ],
)
def test_native_roi_algorithms_are_available(browser, tool, options):
    client, _ = browser
    w = new(client)
    w = action(
        client,
        w,
        "roi.create",
        name="Selection",
        tool=tool,
        bounds=[4, 4, 10, 10],
        label=23,
        **options,
    )
    assert w["rois"][0]["pixels"] > 0 and w["rois"][0]["labels"] == [23]
    assert all(run[3] == 23 for run in w["rois"][0]["runs"])


def test_roi_edit_enablement_combination_and_slice_tools(browser):
    client, _ = browser
    w = new(client)
    w = action(client, w, "roi.create", name="A", tool="rect", bounds=[2, 2, 4, 4], label=7)
    w = action(client, w, "roi.create", name="B", tool="rect", bounds=[4, 4, 4, 4], label=11)
    w = action(client, w, "roi.combine", name="A", others=["B"])
    assert w["rois"][0]["pixels"] == 28 and w["rois"][0]["labels"] == [7, 11]
    w = action(client, w, "roi.subtract", name="A", others=["B"])
    assert w["rois"][0]["pixels"] == 12
    w = action(client, w, "roi.shift", name="A", shifts={"x": 1})
    w = action(client, w, "roi.copy_slice", name="A", targets=[2])
    w = action(client, w, "roi.interpolate", name="A", axis="z", method="linear")
    assert w["rois"][0]["pixels"] == 36 and w["rois"][0]["labels"] == [7]
    w = action(client, w, "roi.disable", name="B")
    assert not w["rois"][1]["enabled"]
    w = action(client, w, "roi.enable", name="B")
    assert w["rois"][1]["enabled"]


def roi_array(client, workspace, name):
    session = client.app.state.records[workspace["id"]].workspace.session
    return next(np.asarray(r.mask).copy() for r in session.shared_rois if r.name == name)


def test_drawing_rounds_preserve_other_slices_and_support_subtraction_and_undo(browser):
    client, _ = browser
    w = view(client, new(client), z=0)
    original = pixels(client, w).copy()
    w = action(client, w, "roi.create", name="Tissue", tool="rect", bounds=[2, 2, 4, 4], label=7)
    w = action(client, w, "roi.draw", name="Tissue", tool="rect",
               bounds=[10, 2, 4, 4], label=7, mode="add")
    assert len(w["rois"]) == 1 and w["rois"][0]["pixels"] == 32
    w = view(client, w, z=2)
    w = action(client, w, "roi.draw", name="Tissue", tool="ellipse",
               bounds=[2, 2, 4, 4], label=9, mode="add")
    before = roi_array(client, w, "Tissue")
    assert w["rois"][0]["labels"] == [7, 9]
    w = view(client, w, z=0)
    w = action(client, w, "roi.draw", name="Tissue", tool="rect",
               bounds=[2, 2, 2, 4], label=7, mode="subtract")
    assert np.count_nonzero(roi_array(client, w, "Tissue")) == np.count_nonzero(before) - 8
    assert np.count_nonzero(roi_array(client, w, "Tissue") == 9) == np.count_nonzero(before == 9)
    w = action(client, w, "roi.undo")
    np.testing.assert_array_equal(roi_array(client, w, "Tissue"), before)
    np.testing.assert_array_equal(pixels(client, w), original)


def test_auto_subtract_covers_shapes_rounds_and_brush_in_one_undo(browser):
    client, _ = browser
    w = new(client)
    w = action(client, w, "roi.create", name="A", tool="rect", bounds=[2, 2, 4, 4], label=7)
    w = action(client, w, "roi.create", name="Inactive", tool="rect",
               bounds=[2, 2, 4, 4], label=5)
    w = action(client, w, "roi.disable", name="Inactive")
    original = {r["name"]: roi_array(client, w, r["name"]) for r in w["rois"]}
    w = action(client, w, "roi.create", name="B", tool="rect",
               bounds=[4, 4, 4, 4], label=11, auto_subtract=True)
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


@pytest.mark.parametrize("output,overlap,counts", [
    ("binary", "first", {1: 28}),
    ("indexed", "first", {1: 16, 2: 12}),
    ("indexed", "last", {1: 12, 2: 16}),
])
def test_binary_and_indexed_composition_preserve_sources_and_label_identity(browser, output, overlap, counts):
    client, _ = browser
    w = new(client)
    for name, bounds in [("A", [2, 2, 4, 4]), ("B", [4, 4, 4, 4])]:
        w = action(client, w, "roi.create", name=name, tool="rect", bounds=bounds, label=1)
    originals = {name: roi_array(client, w, name) for name in ["A", "B"]}
    w = action(client, w, "roi.compose", name="Joint", sources=["A", "B"],
               output=output, overlap=overlap)
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


def test_composition_rejects_overlap_or_missing_sources_without_modifying_masks(browser):
    client, _ = browser
    w = new(client)
    for name in ["A", "B"]:
        w = action(client, w, "roi.create", name=name, tool="rect", bounds=[2, 2, 4, 4])
    for sources in [["A", "B"], ["A", "missing"], ["A", "A"]]:
        result = client.post(
            f"/api/workspaces/{w['id']}/actions",
            json={"revision": w["revision"], "action": "roi.compose",
                  "options": {"name": "Joint", "sources": sources, "output": "indexed", "overlap": "error"}},
        )
        assert result.status_code == 422
        assert client.get(f"/api/workspaces/{w['id']}").json() == w


def test_growth_and_selective_label_edits_keep_labels_and_image_values(browser):
    client, _ = browser
    w = view(client, new(client), z=1)
    original = pixels(client, w).copy()
    w = action(client, w, "roi.create", name="Labels", tool="rect", bounds=[8, 8, 1, 1], label=7)
    w = action(client, w, "roi.draw", name="Labels", tool="rect",
               bounds=[12, 8, 1, 1], label=9, mode="add")
    before = roi_array(client, w, "Labels")
    w = action(client, w, "roi.grow", name="Labels", radius=1)
    expanded = roi_array(client, w, "Labels")
    assert np.sum(expanded == 7) == 7 and np.sum(expanded == 9) == 7
    np.testing.assert_array_equal(expanded[before != 0], before[before != 0])
    w = action(client, w, "roi.brush", name="Labels", points=[[8, 8], [12, 8]],
               size=8, label=7, erase=True, label_only=True)
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


@pytest.mark.parametrize("method", ["linear", "nearest", "spline"])
def test_interpolation_retains_other_labels_on_drawn_slices(browser, method):
    client, _ = browser
    w = view(client, new(client), z=0)
    w = action(client, w, "roi.create", name="Labels", tool="rect",
               bounds=[2, 2, 4, 4], label=2)
    w = view(client, w, z=2)
    w = action(client, w, "roi.draw", name="Labels", tool="rect",
               bounds=[2, 2, 4, 4], label=2, mode="add")
    w = view(client, w, z=1)
    w = action(client, w, "roi.draw", name="Labels", tool="rect",
               bounds=[3, 3, 2, 2], label=1, mode="add")
    original = roi_array(client, w, "Labels")
    w = action(client, w, "roi.interpolate", name="Labels", axis="z", method=method)
    actual = roi_array(client, w, "Labels")
    np.testing.assert_array_equal(actual[original != 0], original[original != 0])
    assert w["rois"][0]["pixels"] == 48
    assert w["rois"][0]["labels"] == [1, 2]


def test_signed_complex_nonfinite_and_coil_views_preserve_values(browser):
    client, _ = browser
    w = new(client, "multimodal")
    w = view(client, w, experiment="Complex coils", component="real", frame=1)
    image = pixels(client, w)
    assert image.dtype == np.dtype("float32") and np.min(image) < 0 < np.max(image)
    w = view(client, w, component="phase")
    assert np.max(np.abs(pixels(client, w))) <= np.pi
    w = view(
        client,
        w,
        view="MRax reconstruction",
        reconstruction_component="phase",
        coil=0,
        unwrap_phase=False,
    )
    expected = pixels(client, w)
    assert w["unit"] == "rad"
    assert np.any(expected != 0)
    w = view(client, w, coil=1)
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (pixels(client, w) - expected))), 0.4, atol=1e-6
    )
    w = view(client, w, view="K-space", kspace_mode="magnitude")
    assert w["unit"] == ""
    assert np.min(pixels(client, w)) >= 0
    w = view(client, w, experiment="Signed with missing samples")
    data = pixels(client, w)
    assert np.isnan(data[0, 0]) and np.isinf(data[0, 1])
    w = action(client, w, "roi.create", name="Missing", tool="rect", bounds=[0, 0, 2, 1])
    assert w["timecourse"] is not None
    assert w["timecourse"]["observed"][0] is None


def test_invalid_edits_and_views_are_transactional(browser):
    client, _ = browser
    w = new(client)
    for command in (
        {"action": "fit.settings", "options": {"num_steps": -1}},
        {
            "action": "metadata.update",
            "options": {"observables": {"TE": {"values": [1, 2], "unit": "ms"}}},
        },
        {
            "action": "roi.create",
            "options": {"name": "bad", "tool": "regular_polygon", "sides": 1000},
        },
    ):
        response = client.post(
            f"/api/workspaces/{w['id']}/actions", json={"revision": w["revision"], **command}
        )
        assert response.status_code == 422
        assert client.get(f"/api/workspaces/{w['id']}").json()["revision"] == w["revision"]
    assert (
        client.patch(
            f"/api/workspaces/{w['id']}/view", json={"revision": 0, "view": "Registration overlay"}
        ).status_code
        == 422
    )
    assert client.get(f"/api/workspaces/{w['id']}").json()["view"] == "Image"


def test_frame_exclusions_and_full_map_fit_round_trip(browser):
    client, _ = browser
    w = new(client)
    w = action(client, w, "roi.create", name="Fit", tool="rect", bounds=[4, 3, 6, 5])
    w = action(client, w, "roi.select", name="Fit")
    w = action(client, w, "frames.exclude", indices=[7])
    w = action(client, w, "fit.settings", num_steps=600)
    w, result = job(client, w, "full_fit", roi_name="Fit", use_roi=True, device="cpu")
    assert result["summary"]["maps"] and w["fit_info"]["loss_history"]
    w, _ = job(client, w, "preview", roi_name="Fit", steps=300)
    assert not list(client.app.state.root.rglob("input.mrax"))
    assert not list(client.app.state.root.rglob("output.mrax"))
    w = view(client, w, view="Map", map_name="t2")
    values = pixels(client, w)
    assert np.median(values[3:8, 4:10]) == pytest.approx(80, abs=0.1)
    assert w["unit"] == "ms" and w["frame_exclusions"] == [7]
    w = action(client, w, "experiment.rename", name="Renamed T2")
    assert w["fit_info"]["name"] == "Renamed T2"
    saved = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=saved.content)
    assert restored.status_code == 200, restored.text
    r = restored.json()
    assert r["fit_info"]["loss_history"] == w["fit_info"]["loss_history"]
    np.testing.assert_array_equal(pixels(client, r), values)


def test_manual_registration_metrics_labels_and_transform_persistence(browser):
    client, _ = browser
    w = view(client, new(client, "multimodal"), experiment="Motion shifted", frame=3)
    w = action(
        client, w, "roi.create", name="Motion label", tool="rect", bounds=[13, 10, 5, 6], label=17
    )
    w, result = job(
        client,
        w,
        "register",
        reference_experiment="Motion reference",
        moving_experiment="Motion shifted",
        reference_frame=0,
        moving_frame=3,
        manual=True,
        manual_settings={"shift_x": 3, "shift_y": 0},
    )
    key = result["summary"]["key"]
    assert result["summary"]["diagnostics"]["post"]["mse"] < 1e-10
    w = action(client, w, "experiment.rename", name="Renamed motion")
    assert w["registrations"][0]["metadata"]["moving_experiment"] == "Renamed motion"
    w = action(client, w, "roi.transform", name="Motion label", transform=key)
    transferred = next(r for r in w["rois"] if r["name"] == "Motion label transformed")
    assert transferred["labels"] == [17] and transferred["pixels"] == 30
    w = view(client, w, view="Registration overlay")
    assert pixels(client, w).shape == (32, 40, 3)
    saved = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=saved.content)
    assert restored.status_code == 200, restored.text
    assert restored.json()["registrations"][0]["key"] == key
    np.testing.assert_array_equal(pixels(client, restored.json()), pixels(client, w))
    coils = view(
        client,
        restored.json(),
        experiment="Complex coils",
        view="MRax reconstruction",
        reconstruction_component="phase",
        coil=0,
        unwrap_phase=False,
    )
    first = pixels(client, coils)
    coils = view(client, coils, coil=1)
    assert coils["coil_count"] == 2
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (pixels(client, coils) - first))), 0.4, atol=1e-6
    )


def test_numpy_raw_and_blank_import_use_explicit_file_roots(browser):
    client, folder = browser
    data = np.arange(24, dtype=np.int16).reshape(4, 6) - 12
    np.save(folder / "signed.npy", data)
    data.astype(">i2").tofile(folder / "signed.raw")
    root = client.get("/api/files/roots").json()[0]["id"]
    assert client.get("/api/files/list", params={"root": root, "path": "../"}).status_code == 422
    for source, options in [
        ("signed.npy", {"kind": "numpy", "dim_order": ["x", "y"]}),
        (
            "signed.raw",
            {
                "kind": "raw",
                "dtype": "int16",
                "endian": "big",
                "shape": [4, 6],
                "dim_order": ["x", "y"],
            },
        ),
    ]:
        response = client.post(
            "/api/import", json={"root": root, "path": source, "options": options}
        )
        assert response.status_code == 200, response.text
        info = response.json()
        placeholder = client.get(f"/api/workspaces/{info['workspace_id']}").json()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and client.get(f"/api/jobs/{info['id']}").json()[
            "status"
        ] in {"running", "queued"}:
            time.sleep(0.05)
        outcome = client.get(f"/api/jobs/{info['id']}").json()
        assert outcome["status"] == "completed", outcome["detail"]
        w = client.get(f"/api/workspaces/{placeholder['id']}").json()
        np.testing.assert_array_equal(pixels(client, w), data)


def test_connected_folder_rejects_symlinks_outside_its_root(browser):
    client, folder = browser
    outside = folder.parent / "outside.npy"
    np.save(outside, np.ones((2, 3)))
    try:
        (folder / "escape.npy").symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account cannot create symbolic links")
        raise
    root = client.get("/api/files/roots").json()[0]["id"]
    entries = client.get("/api/files/list", params={"root": root}).json()["entries"]
    assert not any(entry["name"] == "escape.npy" for entry in entries)
    assert client.post("/api/import", json={"root": root, "path": "escape.npy"}).status_code == 422


@pytest.mark.parametrize("model,observable", [("t1_ir", "TI"), ("t1_vtr", "TR")])
def test_t1_import_roi_fit_and_persistence_use_the_shared_models(browser, model, observable):
    client, folder = browser
    times = np.asarray([60.0, 160.0, 350.0, 700.0, 1200.0, 2000.0, 3500.0])
    signal = (
        np.abs(120 * (1 - 2 * np.exp(-times / 900))) + 1e-6
        if model == "t1_ir"
        else 120 * (1 - np.exp(-times / 900)) + 5
    )
    data = np.broadcast_to(signal[:, None, None], (len(times), 3, 4)).copy()
    np.save(folder / "independent.npy", data)
    root = client.get("/api/files/roots").json()[0]["id"]
    response = client.post(
        "/api/import",
        json={
            "root": root,
            "path": "independent.npy",
            "options": {
                "kind": "numpy",
                "dim_order": ["frame", "x", "y"],
                "model_name": model,
                "exp_type": model,
                "observables": {observable: {"values": times.tolist(), "unit": "ms"}},
            },
        },
    )
    assert response.status_code == 200, response.text
    pending = response.json()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        imported = client.get(f"/api/jobs/{pending['id']}").json()
        if imported["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert imported["status"] == "completed", imported["detail"]
    w = client.get(f"/api/workspaces/{pending['workspace_id']}").json()
    w = action(client, w, "roi.create", name="T1 ROI", tool="rect", bounds=[0, 0, 4, 3])
    overrides = {"t1": {"init": 1100.0}, "amplitude": {"init": 100.0}}
    if model == "t1_vtr":
        overrides["bias"] = {"value": 5.0, "trainable": False, "transform": "identity"}
    w = action(client, w, "fit.settings", step_size=0.02, param_overrides=overrides)
    w, result = job(client, w, "preview", roi_name="T1 ROI", steps=1000)
    fitted = result["result"]
    assert fitted["parameters"]["t1"] == pytest.approx(900, rel=0.01)
    assert fitted["relative_rmse"] < 1e-3
    np.testing.assert_allclose(fitted["observed"], signal, rtol=1e-6)
    bundle = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=bundle.content).json()
    assert restored["model"] == model
    assert restored["preview"]["parameters"] == fitted["parameters"]


def test_export_files_preserve_existing_folder_contents(browser):
    client, folder = browser
    w = new(client)
    w = action(client, w, "roi.create", name="Exported", tool="rect", bounds=[4, 3, 6, 5], label=9)
    inspected = client.get(f"/api/workspaces/{w['id']}/inspect").json()
    roi = next(row for row in inspected["objects"] if row["kind"] == "roi")
    w, result = job(client, w, "export", kind="object", object=roi["id"], format="npy")
    response = client.get(result["summary"]["download"])
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        mask = np.load(io.BytesIO(archive.read("mrax-export.npy")), allow_pickle=False)
    assert sorted(np.unique(mask).tolist()) == [0, 9]
    root = client.get("/api/files/roots").json()[0]["id"]
    w, _ = job(client, w, "save", kind="package", root=root, path="saved")
    assert (folder / "saved" / "mrax_package.json").is_file()
    assert not w["dirty"]
    (folder / "unrelated").mkdir()
    (folder / "unrelated" / "keep.txt").write_text("keep")
    response = client.post(
        f"/api/workspaces/{w['id']}/tasks",
        json={
            "revision": w["revision"],
            "operation": "save",
            "options": {"kind": "package", "root": root, "path": "unrelated", "overwrite": True},
        },
    )
    key = response.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and client.get(f"/api/jobs/{key}").json()["status"] in {
        "queued",
        "running",
    }:
        time.sleep(0.05)
    exported = client.get(f"/api/jobs/{key}").json()
    assert exported["status"] == "completed"
    assert exported["summary"]["saved"]
    assert (folder / "unrelated" / "keep.txt").read_text() == "keep"
    assert (folder / "unrelated" / "mrax_package.json").is_file()


def test_fitting_preserves_navigation_to_another_experiment_and_view(browser):
    client, _ = browser
    w = action(
        client,
        new(client, "multimodal"),
        "roi.create",
        name="Fit",
        tool="rect",
        bounds=[4, 3, 6, 5],
    )
    w = action(client, w, "roi.create", name="Inspect", tool="rect", bounds=[12, 9, 4, 4])
    w = view(client, w, selected_roi="Fit")
    response = client.post(
        f"/api/workspaces/{w['id']}/tasks",
        json={
            "revision": w["revision"],
            "operation": "preview",
            "options": {"roi_name": "Fit", "steps": 600},
        },
    )
    assert response.status_code == 200, response.text
    w = view(client, w, selected_roi="Inspect", frame=2, colormap="hot")
    w = view(
        client,
        w,
        experiment="Complex coils",
        view="MRax reconstruction",
        plane="coronal",
        frame=1,
        component="real",
        colormap="viridis",
        reconstruction_component="phase",
        coil=1,
        unwrap_phase=False,
        scene={"hidden_rois": ["Fit"], "fit_rois": [], "roi_opacity": 0.6},
    )
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{response.json()['id']}").json()
        if result["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert result["status"] == "completed", result["detail"]
    current = client.get(f"/api/workspaces/{w['id']}").json()
    assert current["experiment"] == "Complex coils" and current["view"] == "MRax reconstruction"
    assert (
        current["frame"] == 1
        and current["plane"] == "coronal"
        and current["reconstruction_coil"] == 1
    )
    assert not current["unwrap_phase"] and current["reconstruction_component"] == "phase"
    assert current["display"]["component"] == "real" and current["display"]["colormap"] == "viridis"
    assert current["scene"]["hidden_rois"] == ["Fit"]
    assert current["scene"]["fit_rois"] == [] and current["scene"]["roi_opacity"] == 0.6
    previous = view(client, current, experiment="T2 phantom")
    assert previous["selected_roi"] == "Inspect" and previous["frame"] == 2
    assert previous["display"]["colormap"] == "hot"


def test_new_study_constants_survive_the_first_background_import(browser):
    client, _ = browser
    w = client.post("/api/workspaces/new").json()
    w = action(client, w, "study.globals", field_strength_mhz={"values": 123.4, "unit": "MHz"})
    response = client.post(
        "/api/import",
        json={
            "workspace_id": w["id"],
            "revision": w["revision"],
            "options": {"kind": "blank", "shape": [2, 3], "dim_order": ["x", "y"]},
        },
    )
    assert response.status_code == 200, response.text
    key = response.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{key}").json()
        if result["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert result["status"] == "completed", result["detail"]
    values = client.get(f"/api/workspaces/{w['id']}/inspect").json()["study_globals"][
        "field_strength_mhz"
    ]
    assert values["values"] == pytest.approx(123.4) and values["unit"] == "MHz"


def test_recovery_restores_only_explicit_checkpoints(tmp_path):
    state = tmp_path / "recovery"
    with TestClient(create_app(state_dir=state), base_url="http://localhost") as client:
        client.headers["x-mrax-token"] = client.get("/api/session").json()["token"]
        w = action(
            client,
            new(client),
            "roi.create",
            name="Recovered",
            tool="rect",
            bounds=[2, 2, 4, 4],
            label=31,
        )
    assert len(list(state.glob("*.mrax"))) == 1
    with TestClient(create_app(state_dir=state), base_url="http://localhost") as client:
        client.headers["x-mrax-token"] = client.get("/api/session").json()["token"]
        rows = client.get("/api/workspaces").json()
        assert rows[0]["rois"][0]["labels"] == [31]
        assert (
            client.request(
                "DELETE", f"/api/workspaces/{w['id']}", json={"revision": 0, "discard": True}
            ).status_code
            == 204
        )
    assert not list(state.glob("*.mrax"))


@pytest.mark.parametrize("kind", ["bruker", "pulseq", "dicom", "nifti", "blank"])
def test_vendor_discovery_and_imports_use_existing_scientific_readers(browser, kind):
    import nibabel as nib
    from _synthetic import _write_classic_dicom_slice, _write_minimal_bruker_scan

    client, folder = browser
    root = client.get("/api/files/roots").json()[0]["id"]
    source = folder / kind
    source.mkdir()
    options = {"kind": kind, "field_strength_mhz": 100}
    relative = kind
    if kind == "bruker":
        _write_minimal_bruker_scan(source)
    elif kind == "pulseq":
        (source / "relaxation.seq").write_text(
            "[DEFINITIONS]\nName inversion_recovery\nTI 0.03 0.1 0.4\nFREQ 120\n"
        )
        curve = np.abs(1000 * (1 - 2 * np.exp(-np.array([30.0, 100.0, 400.0]) / 180.0)))
        nib.save(
            nib.Nifti1Image(
                np.broadcast_to(curve, (4, 3, 2, 3)).copy(), np.diag([2.0, 3.0, 4.0, 1.0])
            ),
            source / "images.nii.gz",
        )
    elif kind == "dicom":
        for echo in range(2):
            for index in range(2):
                _write_classic_dicom_slice(
                    source / f"image_{echo}_{index}.dcm",
                    series_uid="1.2.826.0.1.3680043.10.1000.99",
                    echo_index=echo,
                    slice_index=index,
                )
    elif kind == "nifti":
        nib.save(
            nib.Nifti1Image(
                np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.diag([2.0, 3.0, 4.0, 1.0])
            ),
            source / "image.nii",
        )
        relative += "/image.nii"
        options["dim_order"] = ["x", "y", "z"]
    else:
        options.update(shape=[4, 6], dtype="float32", dim_order=["x", "y"])
    preview = client.post(
        "/api/files/preview", json={"root": root, "path": relative, "options": options}
    )
    assert preview.status_code == 200, preview.text
    response = client.post("/api/import", json={"root": root, "path": relative, "options": options})
    assert response.status_code == 200, response.text
    info = response.json()
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        outcome = client.get(f"/api/jobs/{info['id']}").json()
        if outcome["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert outcome["status"] == "completed", outcome["detail"]
    w = client.get(f"/api/workspaces/{info['workspace_id']}").json()
    if kind == "bruker":
        assert w["shape"] == [2, 2]
    elif kind == "pulseq":
        # The initial display follows the importer's native anatomical plane.
        assert w["frame_count"] == 3 and w["spacing"] == [3, 4] and w["shape"] == [3, 2]
    elif kind == "dicom":
        assert w["frame_count"] == 2 and w["spacing"] == [3, 2] and w["plane_dims"] == ["y", "x"]
    elif kind == "nifti":
        w = view(client, w, plane="axial")
        assert w["spacing"] == [2, 3]
        np.testing.assert_array_equal(
            pixels(client, w), np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, 0]
        )
    else:
        np.testing.assert_array_equal(pixels(client, w), np.zeros((4, 6)))


def test_automatic_registration_and_linear_drift_improve_synthetic_motion(browser):
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
        settings={
            "transform": "translation",
            "metric": "mean_squares",
            "optimizer": "regular_step_gradient_descent",
            "initializer": "identity",
            "sampling_strategy": "none",
            "iterations": 80,
            "learning_rate": 0.5,
        },
    )
    diagnostics = result["summary"]["diagnostics"]
    assert diagnostics["post"]["mse"] < diagnostics["pre"]["mse"] * 0.1
    w, result = job(
        client,
        w,
        "drift",
        settings={
            "transform": "translation",
            "metric": "negative_mse",
            "reference": "first",
            "num_steps": 80,
            "step_size": 0.1,
        },
    )
    summary = result["summary"]["summary"]
    assert summary["groups"] and np.isfinite(summary["groups"][0]["final_objective"])
    assert (
        summary["groups"][0]["similarity_after_mean"]
        > summary["groups"][0]["similarity_before_mean"]
    )
    corrected = pixels(client, w)
    w, _ = job(client, w, "pipeline", action="enabled", enabled=False)
    source = pixels(client, w)
    assert not np.array_equal(source, corrected)
    w, _ = job(client, w, "pipeline", action="enabled", enabled=True)
    saved = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=saved.content)
    assert restored.status_code == 200, restored.text
    np.testing.assert_array_equal(pixels(client, restored.json()), corrected)
