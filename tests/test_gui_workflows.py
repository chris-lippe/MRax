"""Synthetic gui contracts and numerical references."""

import numpy as np
import pytest

from mrax.analysis import ROI
from mrax.data_struct import Quantity
from mrax.gui import GuiSession
from mrax.gui.masks import import_mask
from mrax.gui.services.export import _export_object_records_for_session, _write_single_export_object
from mrax.io.nifti import load_nifti, save_nifti


def test_gui_session_switches_3d_image_planes_and_spatial_slider():
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
