"""Synthetic gui contracts and numerical references."""

import jax.numpy as jnp
import numpy as np
import pytest

from mrax.analysis import ROI
from mrax.data_struct import Quantity
from mrax.gui import GuiSession
from mrax.gui.masks import embed_mask_slice, import_mask, mask_slice_for_display, replace_mask_slice
from mrax.gui.services.export import _export_object_records_for_session, _write_single_export_object
from mrax.gui.services.importing import (
    _apply_observable_dim_compatibility,
)
from mrax.io.nifti import load_nifti, save_nifti


@pytest.mark.parametrize("native_shape", [(4, 7), (6, 6)])
def test_two_dimensional_masks_follow_declared_display_axes(native_shape):
    displayed = np.zeros(native_shape[::-1], dtype=np.uint16)
    displayed[1:3, 1:3] = 7
    native = embed_mask_slice(
        displayed, spatial_shape=native_shape, spatial_dims=("x", "y"), plane_dims=("y", "x")
    )
    np.testing.assert_array_equal(native, displayed.T)
    np.testing.assert_array_equal(
        mask_slice_for_display(native, dims=("x", "y"), plane_dims=("y", "x")), displayed
    )
    changed = displayed.copy()
    changed[0, 1] = 3
    for mode in ("replace", "add", "erase"):
        actual = replace_mask_slice(
            native, changed, dims=("x", "y"), plane_dims=("y", "x"), mode=mode
        )
        expected = changed.T if mode == "replace" else native.copy()
        if mode == "add":
            expected[changed.T != 0] = changed.T[changed.T != 0]
        elif mode == "erase":
            expected[changed.T != 0] = 0
        np.testing.assert_array_equal(actual, expected)


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
        "rotated_volume", data, dims={"frame": 2, "x": 3, "y": 4, "z": 5},
        extra_metadata={"spatial_geometry": {
            "affine": [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            "spatial_shape": [3, 4, 5], "axis_anatomical_labels": ["x", "y", "z"],
            "status": "known",
        }},
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



def test_dti_roi_fit_keeps_native_maps_and_limits_solver_grid(monkeypatch):
    import mrax.gui.session as session_module

    rng = np.random.default_rng(91)
    directions = rng.normal(size=(18, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.vstack([np.zeros(3), directions])
    bvals = np.r_[0.0, np.full(18, 800.0)]
    tensor = np.asarray([[0.0014, 0.0001, 0.00002],
                         [0.0001, 0.0008, -0.00003],
                         [0.00002, -0.00003, 0.0005]])
    attenuation = np.einsum("ni,ij,nj->n", directions, tensor, directions)
    shape = (11, 13, 5)
    amplitude = 1000 + np.indices(shape).sum(axis=0)
    data = np.exp(-bvals[:, None, None, None] * attenuation[:, None, None, None]) * amplitude
    affine = np.diag([0.6, 0.8, 1.5, 1.0])
    session = GuiSession()
    session.create_empty_study("Tensor")
    session.add_experiment(
        "Diffusion", data, dims={"frame": 19, "x": 11, "y": 13, "z": 5},
        exp_type="dti", model_name="dti",
        observables={"bvals": Quantity(bvals, "s/mm^2"), "bvecs": Quantity(directions, "1")},
        extra_metadata={"affine": affine.tolist()},
    )
    original_image = session.original_image_array.copy()
    mask = np.zeros(shape, bool)
    mask[2:5, 4:7, 2:3] = True
    session.add_roi(ROI("Region", mask, dims=("x", "y", "z")))
    calls = []
    original = session_module.run_fit

    def observe(experiment, *args, **kwargs):
        calls.append(experiment.spatial_shape)
        return original(experiment, *args, **kwargs)

    monkeypatch.setattr(session_module, "run_fit", observe)
    result = session.run_current_fit(roi_name="Region", num_steps=0)
    assert calls == [(3, 3, 1)]
    np.testing.assert_allclose(result["spatial_geometry"]["affine"], affine)
    for name, expected in (("dxx", tensor[0, 0]), ("dyy", tensor[1, 1]),
                           ("dzz", tensor[2, 2]), ("dxy", tensor[0, 1])):
        values = np.asarray(result["fields"][name])
        assert values.shape == shape
        np.testing.assert_allclose(values[mask], expected, atol=2e-7)
        np.testing.assert_array_equal(values[~mask], 0)
    np.testing.assert_allclose(np.asarray(result["fields"]["s0"])[mask], amplitude[mask], rtol=1e-5)
    session.fit_settings.mask_fill_value = None
    unmasked = session.run_current_fit(roi_name="Region", num_steps=0)
    assert calls[-1] == shape
    np.testing.assert_allclose(np.asarray(unmasked["fields"]["s0"]), amplitude, rtol=1e-5)
    np.testing.assert_array_equal(session.original_image_array, original_image)


def test_observable_compatibility_fails_on_mismatched_leading_axis():
    observables = {"TE": Quantity(jnp.asarray([10.0, 20.0, 30.0, 40.0]), "ms")}
    dims = {"frame": 3, "x": 2, "y": 2}

    with pytest.raises(ValueError, match="TE"):
        _apply_observable_dim_compatibility(
            observables,
            dims,
            exp_type="t2_se",
            model_name="identity",
            policy="fail",
        )
