"""Synthetic experiments with independently oriented slice packages."""

import numpy as np
import pytest
from test_bruker_geometry_workflows import LPS_TO_RAS, _scan

from mrax.gui.services.analysis_workflows import run_alignment
from mrax.gui.services.roi_tools import roi_command
from mrax.gui.services.scene import fitting_roi, render_overlay
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession
from mrax.io.image_export import export_image_files


def workspace(path, counts=(3, 2, 1)):
    _scan(path, counts=counts)
    session = GuiSession()
    session.import_bruker_study(path, field_strength_mhz=100)
    return StudyWorkspace("packages", "Packages", session)


def test_package_selection_sets_orientation_and_hides_unmeasured_slices(tmp_path):
    work = workspace(tmp_path)
    session = work.session
    assert session.experiment_names == ["1"]
    for package, plane in enumerate(("axial", "coronal", "sagittal")):
        session.set_active_index("slice_package", package)
        assert session.active_image_plane == session.native_image_plane == plane
        assert session.frame_image().shape == (5, 7)
        assert session.view_dimensions["z"] == (3, 2, 1)[package]
        assert session.image_plane_axis_directions == ((-1, -1), (1, -1), (1, -1))[package]
        assert (
            session.image_plane_axis_labels
            == {"axial": ("y", "x"), "coronal": ("z", "x"), "sagittal": ("z", "y")}[plane]
        )
        session.set_active_index("z", 20)
        assert session.active_z == (2, 1, 0)[package]
        assert np.isfinite(session.frame_image()).all()
    assert not session.supports_image_planes()
    session.set_active_index("slice_package", 1)
    session.set_image_plane("axial")
    assert session.frame_image().shape == (2, 7)
    session.set_active_index("slice_package", 2)
    assert session.active_image_plane == "sagittal"
    session.replace_fitted_maps(
        {"Package map": session.image_array}, dimensions={"Package map": session.experiment_dims},
    )
    work.view_label = "Map"
    for package in range(3):
        session.set_active_index("slice_package", package)
        session.set_active_index("frame", 1)
        session.set_active_index("z", 0)
        result = run_alignment(work, {
            "reference_experiment": "1", "moving_experiment": "1",
            "reference_source": "map", "moving_source": "map",
            "reference_map": "Package map", "moving_map": "Package map",
            "reference_frame": session.map_frame_index(), "moving_frame": session.map_frame_index(),
            "manual": True,
        }, lambda *args: None)
        overlay = render_overlay(work, {"kind": "registration", "registration_key": result["key"], "component": "auto"})
        np.testing.assert_array_equal(overlay, session.map_image())


def test_package_rois_and_spatial_edits_do_not_cross_packages(tmp_path):
    work = workspace(tmp_path)
    session = work.session
    roi_command(work, "create", {"name": "Region", "tool": "rect", "bounds": [1, 1, 3, 2]})
    before = session.rois[0].mask.copy()
    session.set_active_index("slice_package", 1)
    roi_command(work, "draw", {"name": "Region", "tool": "rect", "bounds": [1, 1, 3, 2]})
    roi_command(work, "grow", {"name": "Region", "radius": 1})
    mask = session.rois[0].mask
    assert mask.shape == (3, 7, 5, 3)
    np.testing.assert_array_equal(mask[0], before[0])
    assert np.count_nonzero(mask[1]) > np.count_nonzero(mask[0])
    assert not mask[1, ..., 2:].any() and not mask[2].any()
    selection = fitting_roi(work, ["Region"])
    assert selection.dims == ("slice_package", "x", "y", "z")
    np.testing.assert_array_equal(selection.mask, mask != 0)


def test_exports_crop_each_package_and_use_its_own_geometry(tmp_path):
    nib = pytest.importorskip("nibabel")
    work = workspace(tmp_path / "input")
    session = work.session
    experiment, dataset = session.build_experiment(), session.build_dataset()
    files = export_image_files(
        tmp_path / "images.nii.gz",
        dataset.data,
        dims=experiment.dims,
        geometry=dataset.spatial_geometry,
    )
    assert len(files) == 6
    for index, file in enumerate(files):
        image = nib.load(file)
        count = (3, 2, 1)[index // 2]
        assert image.shape == (count, 5, 7)
        assert np.isfinite(image.get_fdata()).all()
        np.testing.assert_allclose(
            image.affine, dataset.spatial_geometry.affine_for_frame(index), atol=1e-6
        )
    roi_command(work, "create", {"name": "Region", "tool": "rect", "bounds": [1, 1, 3, 2]})
    roi = session.rois[0]
    masks = export_image_files(
        tmp_path / "mask.nii.gz",
        roi.mask,
        dims=dict(zip(roi.dims, roi.mask.shape)),
        geometry=dataset.spatial_geometry,
    )
    assert len(masks) == 3
    for package, file in enumerate(masks):
        image = nib.load(file)
        assert image.shape == ((3, 2, 1)[package], 5, 7)
        np.testing.assert_allclose(
            image.affine, dataset.spatial_geometry.affine_for_frame(package * 2), atol=1e-6
        )
        np.testing.assert_allclose(image.affine[:3, 3], LPS_TO_RAS @ [11, -7, 19], atol=1e-6)
