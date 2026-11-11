"""One experiment with independently oriented packages; all inputs are synthetic."""

import hashlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_bruker_geometry_workflows import LPS_TO_RAS, _scan

from mrax.analysis import ROI
from mrax.core.loss_builder import build_loss, make_model_callable
from mrax.data_struct import Model
from mrax.gui.services.export import _export_object_records_for_session, _write_single_export_object
from mrax.gui.services.mask_preview import native_mask_preview
from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
from mrax.gui.services.processing import apply_processing
from mrax.gui.services.roi_tools import roi_command
from mrax.gui.services.scene import fitting_roi
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession
from mrax.io.bruker_import import import_bruker_experiment, import_bruker_study
from mrax.io.image_export import export_image_files
from mrax.library import make_t2_se_model
from mrax.reco.acquisition import format_coil_images, reconstruct_acquisition
from mrax.reco.layout import encoding_frame_map, image_to_encoding


def workspace(path, counts=(3, 2, 1)):
    _scan(path, counts=counts)
    session = GuiSession()
    session.import_bruker_study(path, field_strength_mhz=100)
    return StudyWorkspace("packages", "Packages", session)


@pytest.mark.parametrize("counts", [(3, 2, 1), (2, 2, 2), (1, 1, 1)])
@pytest.mark.parametrize("echo_first", [True, False])
def test_packages_are_one_experiment_with_an_explicit_axis(tmp_path, counts, echo_first):
    scan, frames, coordinates, positions, matrices, _ = _scan(
        tmp_path, counts=counts, echo_first=echo_first, reverse=True
    )
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in scan.rglob("*") if p.is_file()}
    study, metadata = import_bruker_study(tmp_path, field_strength_mhz=100)
    assert study.names == ["1"]
    experiment, dataset = study.experiments[0], study.datasets[0]
    assert experiment.dims["slice_package"] == 3
    assert dataset.shape == (3, 2, 7, 5, max(counts))
    for package, count in enumerate(counts):
        first = sum(counts[:package])
        for echo in range(2):
            indices = [
                i
                for i, (e, z) in enumerate(coordinates)
                if e == echo and first <= z < first + count
            ]
            indices.sort(key=lambda i: positions[coordinates[i][1]] @ matrices[first][2])
            for z, stored in enumerate(indices):
                np.testing.assert_array_equal(dataset.data[package, echo, :, :, z], frames[stored])
        assert np.isnan(dataset.data[package, ..., count:]).all()
    assert len(metadata["experiments"]["1"]["slice_packages"]) == 3
    result = reconstruct_acquisition(dataset.acquisition)
    np.testing.assert_allclose(
        result.image, np.asarray(dataset.data).reshape(result.image.shape), rtol=2e-5, atol=2e-4
    )
    assert all(hashlib.sha256(p.read_bytes()).digest() == value for p, value in before.items())


def test_package_selection_sets_orientation_and_hides_unmeasured_slices(tmp_path):
    session = workspace(tmp_path).session
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


def test_comparison_keeps_each_packages_native_plane_including_singletons(tmp_path):
    from mrax.gui.services.comparison import default_comparison, render_comparison
    from mrax.gui.services.scene import scene_settings

    work = workspace(tmp_path)
    session = work.session
    scene_settings(work)["comparison"] = {**default_comparison(work), "enabled": True}
    for package in range(3):
        session.set_active_index("slice_package", package)
        images, metadata = render_comparison(work)
        for image, plane in zip(images, metadata["planes"]):
            np.testing.assert_array_equal(image, session.frame_image())
            assert plane["plane_axis_directions"] == list(session.image_plane_axis_directions)
            assert plane["plane_axis_labels"] == list(session.image_plane_axis_labels)


@pytest.mark.parametrize("package, plane", [(1, "coronal"), (2, "sagittal")])
def test_two_dimensional_comparisons_use_the_native_plane(tmp_path, package, plane):
    from mrax.gui.services.comparison import (
        default_comparison,
        render_comparison,
        validate_comparison,
    )
    from mrax.gui.services.scene import experiment_catalog, scene_settings

    scan, *_ = _scan(tmp_path, counts=(1, 1, 1))
    experiment, dataset, _, metadata = import_bruker_experiment(
        scan, slice_package=package, load_raw_kspace=False
    )
    session = GuiSession()
    session.add_experiment(
        "Native plane",
        dataset.data,
        dims=experiment.dims,
        extra_metadata={**metadata, "spatial_geometry": dataset.spatial_geometry.to_metadata()},
    )
    work = StudyWorkspace("native", "Native plane", session)
    assert session.native_image_plane == plane
    for requested in (None, plane):
        comparison = {**default_comparison(work), "enabled": True, "plane": requested}
        validate_comparison(work, comparison, experiment_catalog(work))
        scene_settings(work)["comparison"] = comparison
        images, _ = render_comparison(work)
        for image in images:
            np.testing.assert_array_equal(image, session.frame_image())


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


def test_native_mask_import_preview_and_gui_export_follow_the_package(tmp_path):
    nib = pytest.importorskip("nibabel")
    work = workspace(tmp_path / "input")
    session = work.session
    assert session.image_data_is_original()
    session.set_active_index("slice_package", 2)
    imported = session.imported_roi(ROI("Imported", np.ones((7, 5), dtype=np.uint8)))
    assert imported.mask.shape == (3, 7, 5, 3)
    assert not imported.mask[:2].any()
    assert imported.mask[2, ..., 0].all()
    assert not imported.mask[2, ..., 1:].any()
    session.add_roi(imported)
    preview = native_mask_preview(session, "Imported")
    assert preview["shape"] == [7, 5, 1]
    assert preview["slice_count"] == 1
    record = next(
        row for row in _export_object_records_for_session(session) if row["kind"] == "roi"
    )
    files = _write_single_export_object(record, tmp_path / "gui-mask.nii.gz", "nifti")
    assert len(files) == 3
    for package, file in enumerate(files):
        image = nib.load(file)
        assert image.shape == ((3, 2, 1)[package], 5, 7)
        np.testing.assert_allclose(
            image.affine,
            session.build_dataset().spatial_geometry.affine_for_frame(package * 2),
            atol=1e-6,
        )


def test_scalar_fit_maps_keep_the_package_axis_for_viewing_and_processing(tmp_path):
    work = workspace(tmp_path)
    session = work.session
    maps = np.stack([np.full((7, 5, 3), 40 * (index + 1), dtype=float) for index in range(3)])
    session.current_experiment["fitted_maps"] = {"t2": maps}
    session.active_map = "t2"
    for package in range(3):
        session.set_active_index("slice_package", package)
        np.testing.assert_array_equal(session.map_image(), np.full((5, 7), 40 * (package + 1)))
    apply_processing(
        work,
        {"method": "gaussian", "input": "map:t2", "parameters": {"sigma": 1}},
        lambda *args: None,
    )
    assert session.experiment_dims["slice_package"] == 3
    assert len(session.slice_packages) == 3
    for package in range(3):
        session.set_active_index("slice_package", package)
        np.testing.assert_allclose(session.frame_image(), 40 * (package + 1), rtol=1e-6)


def test_denoising_and_normalization_keep_packages_independent(tmp_path):
    work = workspace(tmp_path)
    session = work.session
    source = session.image_array.copy()
    apply_processing(
        work,
        {"method": "gaussian", "input": "image", "parameters": {"sigma": 1}},
        lambda *args: None,
    )
    filtered = session.image_array
    for package, count in enumerate((3, 2, 1)):
        assert np.isnan(filtered[package, ..., count:]).all()
        assert np.isfinite(filtered[package, ..., :count]).all()
        assert np.mean(filtered[package, ..., :count]) == pytest.approx(
            np.mean(source[package, ..., :count]), rel=2e-6
        )
    # The reference echo is selected independently in each package.
    apply_processing(
        work,
        {
            "method": "normalize",
            "input": "image",
            "reference_experiment": "1",
            "reference_frame": 0,
        },
        lambda *args: None,
    )
    for package, count in enumerate((3, 2, 1)):
        np.testing.assert_allclose(session.image_array[package, 0, ..., :count], 1, rtol=1e-5)


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


def test_package_dimension_and_raw_layout_survive_workspace_roundtrip(tmp_path):
    work = workspace(tmp_path / "input")
    session = work.session
    session.load_raw_data_for_experiment("1", require_acquisition=True)
    session.set_active_index("slice_package", 1)
    session.set_image_plane("axial")
    saved = save_workspace_bundle(work, tmp_path / "packages.mrax")
    loaded = load_workspace_bundle(saved, tmp_path / "loaded", key="reopened")
    assert loaded.session.experiment_names == ["1"]
    assert loaded.session.experiment_dims["slice_package"] == 3
    assert loaded.session.active_indices["slice_package"] == 1
    assert loaded.session.active_image_plane == "axial"
    acquisition = loaded.session.current_experiment["acquisition"]
    result = reconstruct_acquisition(acquisition)
    np.testing.assert_allclose(
        result.image, loaded.session.image_array.reshape(result.image.shape), rtol=2e-5, atol=2e-4
    )
    assert acquisition.geometry.frame_dims == (("slice_package", 3), ("frame", 2))


def test_variable_package_reconstruction_batches_and_inverse_layout(tmp_path):
    work = workspace(tmp_path)
    session = work.session
    session.load_raw_data_for_experiment("1", require_acquisition=True)
    acquisition = session.current_experiment["acquisition"]
    expected = session.image_array.reshape((6, 7, 5, 3))
    batched = reconstruct_acquisition(acquisition, frame_batch_size=1)
    np.testing.assert_allclose(batched.image, expected, rtol=2e-5, atol=2e-4)
    chosen = acquisition.select_frames(np.asarray([4, 1, 3]))
    np.testing.assert_allclose(
        reconstruct_acquisition(chosen).image, expected[[4, 1, 3]], rtol=2e-5, atol=2e-4
    )
    assert chosen.recipe.output_slice_counts == (1, 3, 2)
    assert chosen.geometry.shape_for_frame(0) == (7, 5, 1)
    encoded = image_to_encoding(expected, acquisition)
    formatted = format_coil_images(
        encoded[:, None], replace(acquisition.recipe, output_scale=1), spatial_ndim=2
    )
    np.testing.assert_allclose(formatted[:, 0], expected)
    assert np.bincount(encoding_frame_map(acquisition)).tolist() == [3, 3, 2, 2, 1, 1]
    repeated, first = [], 0
    for count in acquisition.recipe.output_slice_counts:
        repeated.append(np.tile(np.arange(first, first + count), 2))
        first += count
    indices = np.concatenate(repeated)
    averages = replace(
        acquisition,
        samples=acquisition.samples[indices],
        frame_coordinates={
            key: np.asarray(value)[indices] for key, value in acquisition.frame_coordinates.items()
        },
        recipe=replace(
            acquisition.recipe,
            average_repeats=2,
            frame_permutations=tuple(acquisition.recipe.frame_permutations[i] for i in indices),
            frame_image_shifts=tuple(acquisition.recipe.frame_image_shifts[i] for i in indices)
            if acquisition.recipe.frame_image_shifts
            else (),
        ),
    )
    averaged = reconstruct_acquisition(averages, frame_batch_size=1)
    np.testing.assert_allclose(averaged.image, expected, rtol=2e-5, atol=2e-4)
    for package in range(3):
        session.set_active_index("slice_package", package)
        session.set_active_index("z", 0)
        assert session.kspace_image().shape == (5, 7)


def test_fitting_has_independent_parameters_for_each_package(tmp_path):
    scan, *_ = _scan(tmp_path, counts=(3, 2, 1), echoes=4)
    with (scan / "pdata/1/visu_pars").open("a") as stream:
        stream.write("\n##$VisuAcqEchoTime=( 4 )\n12 24 36 48\n")
    experiment, dataset, _, _ = import_bruker_experiment(
        scan, override_type="mse", load_raw_kspace=False
    )
    levels = np.broadcast_to(np.asarray([1.0, 2.0, 3.0])[:, None, None, None], (3, 7, 5, 3)).copy()
    data = np.broadcast_to(levels[:, None], dataset.shape).copy()
    measured = dataset.spatial_geometry.measured_mask().reshape(data.shape)
    dataset.data = jnp.asarray(np.where(measured, data, np.nan))
    model = Model(
        name="independent_package_level",
        parameter_units={"level": "1"},
        reg_weights={"level": 0},
        parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda values: values),
        model_apply=lambda fields, offsets, observables=None: jnp.broadcast_to(
            fields["level"], (experiment.frame_count,) + experiment.spatial_shape
        ),
    )
    loss, parameters, _ = build_loss(experiment, model, dataset)
    assert parameters["level"].shape == (3, 7, 5, 3)
    truth = {"level": jnp.asarray(levels)}
    assert float(loss(truth)) == pytest.approx(0, abs=1e-7)
    changed = {"level": truth["level"].at[1].add(0.5)}
    gradient = np.asarray(jax.grad(loss)(changed)["level"])
    assert np.count_nonzero(gradient[1]) > 0
    np.testing.assert_allclose(gradient[[0, 2]], 0, atol=1e-7)
    assert not gradient[1, ..., 2:].any()
    regularized_loss, _, _ = build_loss(
        experiment, replace(model, reg_weights={"level": 1}), dataset
    )
    assert float(regularized_loss(truth)) == pytest.approx(0, abs=1e-7)


def test_echo_coordinates_are_repeated_within_each_package_for_t2_fitting(tmp_path):
    scan, *_ = _scan(tmp_path, counts=(3, 2, 1), echoes=4)
    with (scan / "pdata/1/visu_pars").open("a") as stream:
        stream.write("\n##$VisuAcqEchoTime=( 4 )\n12 24 36 48\n")
    experiment, dataset, _, _ = import_bruker_experiment(
        scan, override_type="mse", load_raw_kspace=False
    )
    t2 = np.broadcast_to(np.asarray([40.0, 80.0, 120.0])[:, None, None, None], (3, 7, 5, 3))
    fields = {
        "t2": jnp.asarray(t2),
        "amplitude": jnp.full(t2.shape, 1.2),
        "bias": jnp.full(t2.shape, 0.03),
    }
    prediction = np.asarray(make_model_callable(experiment, make_t2_se_model(scale=1))(fields))
    expected = (
        1.2 * np.exp(-np.asarray([12, 24, 36, 48])[None, :, None, None, None] / t2[:, None]) + 0.03
    )
    np.testing.assert_allclose(prediction.reshape(dataset.shape), expected, rtol=2e-6, atol=2e-7)
