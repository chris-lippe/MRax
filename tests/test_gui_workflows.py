"""Synthetic gui contracts and numerical references."""

import numpy as np
import pytest

from mrax.analysis import ROI
from mrax.data_struct import Quantity
from mrax.gui import GuiSession
from mrax.gui.masks import import_mask, mask_slice_for_display
from mrax.gui.services.analysis_workflows import _registration_source
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.export import (
    _export_object_records_for_session,
    _export_object_records_for_workspace,
    _write_single_export_object,
)
from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
from mrax.gui.services.scene import render_overlay
from mrax.gui.services.view_data import display_options
from mrax.gui.services.workspace import StudyWorkspace
from mrax.io.image_export import image_frames
from mrax.io.nifti import load_nifti, save_nifti


def test_gui_session_preserves_native_planes_for_images_and_maps(tmp_path):
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape((2, 3, 4, 5))
    session = GuiSession()
    session.create_empty_study("volume_views")
    session.add_experiment(
        "volume",
        data,
        dims={"frame": 2, "x": 3, "y": 4, "z": 5},
    )
    session.set_active_index("frame", 1)
    session.set_active_index("z", 3)

    assert session.supports_image_planes()
    assert session.image_plane_dims == ("x", "y")
    assert [control["key"] for control in session.dimension_controls()] == ["frame", "z"]
    np.testing.assert_array_equal(session.frame_image(), data[1, :, :, 3])

    session.set_image_plane("coronal")
    session.set_active_index("y", 2)
    assert session.image_plane_dims == ("x", "z")
    assert [control["key"] for control in session.dimension_controls()] == ["frame", "y"]
    np.testing.assert_array_equal(session.frame_image(), data[1, :, 2, :])

    session.set_image_plane("sagittal")
    session.set_active_index("x", 1)
    assert session.image_plane_dims == ("y", "z")
    assert [control["key"] for control in session.dimension_controls()] == ["frame", "x"]
    np.testing.assert_array_equal(session.frame_image(), data[1, 1, :, :])

    session.add_experiment(
        "second_volume",
        np.zeros((3, 4, 5), dtype=np.float32),
        dims={"x": 3, "y": 4, "z": 5},
    )
    session.select_experiment("volume")
    assert session.active_image_plane == "sagittal"
    assert session.active_indices["x"] == 1

    session.add_experiment(
        "rotated_volume",
        data,
        dims={"frame": 2, "x": 3, "y": 4, "z": 5},
        extra_metadata={
            "spatial_geometry": {
                "affine": [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
                "spatial_shape": [3, 4, 5],
                "axis_anatomical_labels": ["x", "y", "z"],
                "status": "known",
            }
        },
    )
    session.set_image_plane("axial")
    session.set_active_index("z", 2)
    session.replace_fitted_maps({"Sample map": Quantity(data[1], "s")})
    assert session.image_plane_dims == ("y", "x")
    np.testing.assert_array_equal(session.map_image(), data[1, :, :, 2].T)

    times = np.asarray([10, 25, 50, 100], dtype=np.float32)
    truth = 30 + np.arange(7)[:, None] * 4 + np.arange(11)[None, :]
    planar = (1e4 * np.exp(-times[:, None, None] / truth)).astype(np.float32)
    affine = np.asarray([[0, -1.5, 0, 12], [2, 0, 0, -8], [0, 0, 3, 4], [0, 0, 0, 1]], dtype=float)
    session.add_experiment(
        "planar_fit",
        planar,
        dims={"te": 4, "x": 7, "y": 11},
        model_name="t2_se",
        observables={"TE": Quantity(times, "ms")},
        extra_metadata={
            "spatial_geometry": {
                "affine": affine.tolist(),
                "spatial_shape": [7, 11],
                "axis_anatomical_labels": ["x", "y"],
                "status": "known",
            }
        },
    )
    session.set_image_plane("axial")
    assert not session.rois
    session.fit_settings.param_overrides = {"bias": {"trainable": False, "value": 0}}
    session.fit_settings.regularizer_configs = {"t2": [{"type": "tv", "weight": 1e-6, "axes": [0, 1]}]}
    preview = session.run_preview_fit(num_steps=2)
    assert preview["roi_name"] is None
    np.testing.assert_allclose(preview["observed"], planar.mean(axis=(1, 2)), rtol=1e-6)
    roi = ROI("Preview region", np.broadcast_to(np.arange(7)[:, None] < 3, (7, 11)).copy(), dims=("x", "y"))
    session.rois = [roi]
    preview = session.run_preview_fit(roi_name=roi.name, use_roi=True, num_steps=2)
    assert preview["roi_name"] == roi.name
    np.testing.assert_allclose(preview["observed"], planar[:, :3].mean(axis=(1, 2)), rtol=1e-6)
    preview = session.run_preview_fit(use_roi=False, num_steps=2)
    assert preview["roi_name"] is None
    np.testing.assert_allclose(preview["observed"], planar.mean(axis=(1, 2)), rtol=1e-6)
    result = session.run_current_fit(use_roi=False, num_steps=600)
    session.replace_fitted_maps(result["fields"])
    assert session.image_plane_dims == ("y", "x")
    assert session.map_image("t2").shape == session.frame_image().shape == (11, 7)
    np.testing.assert_allclose(session.map_image("t2"), truth.T, rtol=0.02)
    np.testing.assert_array_equal(session.map_array("t2"), session.map_image("t2").T)
    np.testing.assert_array_equal(session.frame_image(), planar[0].T)
    np.testing.assert_array_equal(session.original_image_array, planar)
    session.rois = []
    session.add_experiment(
        "planar_image", planar[0], dims={"x": 7, "y": 11},
        extra_metadata={"spatial_geometry": session.image_geometry().to_metadata()},
    )
    np.testing.assert_array_equal(session.frame_image(), planar[0].T)

    grid = {"t": 2, "te": 4, "x": 7, "y": 11, "z": 3}
    field_dims = {"t": 2, "x": 7, "y": 11, "z": 3}
    field = np.arange(2 * 7 * 11 * 3, dtype=np.float32).reshape(tuple(field_dims.values()))
    signal = np.repeat(field[:, None], 4, axis=1)
    session.add_experiment(
        "dynamic_volume", signal, dims=grid,
        extra_metadata={"spatial_geometry": {"affine": affine.tolist(), "spatial_shape": [7, 11, 3], "axis_anatomical_labels": ["x", "y", "z"], "status": "known"}},
    )
    session.replace_fitted_maps({"Dynamic map": field}, dimensions={"Dynamic map": field_dims})
    session.set_active_index("t", 1)
    session.set_active_index("te", 3)
    mask = np.zeros((7, 11, 3), dtype=bool)
    mask[2:5, 3:8, 1] = True
    workspace = StudyWorkspace("views", "Synthetic views", session)
    for plane, cursor, expected in [
        ("axial", {"z": 1}, field[1, :, :, 1].T),
        ("coronal", {"y": 4}, field[1, :, 4, :].T),
        ("sagittal", {"x": 3}, field[1, 3, :, :].T),
    ]:
        session.set_image_plane(plane)
        for dimension, index in cursor.items():
            session.set_active_index(dimension, index)
        np.testing.assert_array_equal(session.map_image(), expected)
        np.testing.assert_array_equal(session.frame_image(), expected)
        projected = mask_slice_for_display(mask, dims=("x", "y", "z"), indices=session.active_indices, plane_dims=session.image_plane_dims)
        assert projected.shape == expected.shape
        overlay = render_overlay(workspace, {"experiment": "dynamic_volume", "kind": "map", "map_name": "Dynamic map", "component": "auto"})
        np.testing.assert_array_equal(overlay, expected)
        registered = _registration_source(session, "dynamic_volume", "map", 1, "Dynamic map", workspace=workspace)
        np.testing.assert_array_equal(registered, expected)
        display_options(workspace)["scene"] = {"comparison": {
            "enabled": True,
            "left": {"experiment": "dynamic_volume", "kind": "original"},
            "right": {"experiment": "dynamic_volume", "kind": "map", "map_name": "Dynamic map"},
        }}
        arrays, comparison = render_comparison(workspace)
        assert comparison["geometry_compatible"]
        np.testing.assert_array_equal(arrays[0], expected)
        np.testing.assert_array_equal(arrays[1], expected)
    assert session.map_frame_count() == 2
    session.set_map_frame(0)
    assert session.active_indices["te"] == 3
    assert session.map_frame_index() == 0
    record = next(row for row in _export_object_records_for_session(session) if row["kind"] == "map" and row["experiment"] == "dynamic_volume")
    assert record["dims"] == field_dims
    frames = list(image_frames(record["array"], record["dims"]))
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[1][2], field[1])
    bundle = save_workspace_bundle(workspace, tmp_path / "views.mrax")
    restored = load_workspace_bundle(bundle, tmp_path / "restored", key="restored").session
    assert restored.map_dimensions() == field_dims
    np.testing.assert_array_equal(restored.map_image(), session.map_image())
    shifted = affine.copy()
    shifted[0, 3] += 4
    session.replace_fitted_maps(
        {"Earlier map": field, "Later map": field},
        dimensions={name: field_dims for name in ["Earlier map", "Later map"]},
    )
    workspace.pipelines["dynamic_volume"] = {"fits": {
        name: {"maps": [name], "result": {"pipeline_grid": {"geometry": {
            "affine": geometry.tolist(), "spatial_shape": [7, 11, 3], "status": "known",
        }}}} for name, geometry in [("Earlier map", affine), ("Later map", shifted)]
    }}
    for record in _export_object_records_for_workspace(workspace):
        if record["experiment"] != "dynamic_volume" or record["kind"] != "map":
            continue
        files = _write_single_export_object(record, tmp_path / (record["name"] + ".nii"), "nifti")
        assert len(files) == 2
        for index, file in enumerate(files):
            nib = pytest.importorskip("nibabel")
            image = nib.load(file)
            np.testing.assert_array_equal(image.get_fdata(), field[index].transpose(2, 1, 0))
            np.testing.assert_allclose(image.affine, affine if record["name"] == "Earlier map" else shifted)


def test_gui_roi_nifti_export_uses_reference_experiment_affine(tmp_path):
    affine = np.asarray(
        [
            [0.0, -1.5, 0.0, 12.0],
            [2.0, 0.0, 0.0, -8.0],
            [0.0, 0.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    session = GuiSession()
    session.create_empty_study("affine_export")
    session.add_experiment(
        "reference_scan",
        np.zeros((2, 3), dtype=np.float32),
        dims={"x": 2, "y": 3},
        extra_metadata={"affine": affine.tolist()},
    )
    mask = np.asarray([[True, False, False], [False, True, False]])
    session.add_roi(
        ROI(
            "mask",
            mask,
            dims=("x", "y"),
            metadata={"source_experiment": "reference_scan", "slice_indices": {"frame": 0}},
        )
    )

    roi_record = next(
        record for record in _export_object_records_for_session(session) if record["kind"] == "roi"
    )
    output = _write_single_export_object(roi_record, tmp_path / "mask.nii", "nifti")[0]
    loaded_mask, loaded_affine = load_nifti(output)

    np.testing.assert_array_equal(loaded_mask, mask.astype(np.float32))
    np.testing.assert_allclose(loaded_affine, affine, atol=1e-6)
    assert roi_record["nifti_reference"]["identity_fallback"] is False
    imported = session.imported_roi(import_mask(output, name="restored", dims=("x", "y")))
    np.testing.assert_array_equal(imported.mask, mask)
    assert imported.dims == ("x", "y")
    shifted = affine.copy()
    shifted[0, 3] += 5
    path = tmp_path / "shifted.nii.gz"
    save_nifti(path, mask.astype(np.uint16), affine=shifted)
    with pytest.raises(ValueError, match="Mask geometry differs"):
        session.imported_roi(import_mask(path, dims=("x", "y")))
