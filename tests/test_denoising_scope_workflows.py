"""Denoising targets are independent, replayable selections of synthetic data."""

import copy

import numpy as np
import pytest
from test_bruker_geometry_workflows import _scan

from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
from mrax.gui.services.pipeline import change_pipeline, current_pipeline
from mrax.gui.services.processing import processing_catalog, run_processing
from mrax.gui.services.reusable_workflows import (
    apply_workflow,
    capture_workflow,
)
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession


def package_workspace(path):
    _scan(path, counts=(3, 2, 1))
    session = GuiSession()
    session.import_bruker_study(path, field_strength_mhz=100)
    return StudyWorkspace("packages", "Packages", session)


def filter_data(work, **options):
    settings = {
        "method": "gaussian",
        "input": "image",
        "axes": ["x", "y"],
        "parameters": {"sigma": 1},
        **options,
    }
    return run_processing(work, settings, lambda *_: None)


def test_package_dimensions_follow_native_orientation_and_measured_extent(tmp_path):
    work = package_workspace(tmp_path)
    expected = [
        (("x", 7), ("y", 5), ("z", 3)),
        (("x", 7), ("z", 5), ("y", 2)),
        (("y", 7), ("z", 5), ("x", 1)),
    ]
    for package, spatial in enumerate(expected):
        work.session.set_active_index("slice_package", package)
        axes = [row for row in processing_catalog(work)["axis_info"] if row["spatial"]]
        assert [(row["label"], row["size"]) for row in axes] == list(spatial)
    catalog = processing_catalog(work, package=0)
    assert next(row for row in catalog["axis_info"] if row["name"] == "z")["size"] == 3
    assert work.session.active_indices["slice_package"] == 2


def test_each_package_keeps_its_denoising_step_and_replacement_does_not_erase_others(tmp_path):
    work = package_workspace(tmp_path)
    original = work.session.image_array.copy()
    filter_data(work, scope="package")
    first = work.session.image_array.copy()
    assert not np.array_equal(first[0], original[0], equal_nan=True)
    np.testing.assert_array_equal(first[1:], original[1:])
    work.session.set_active_index("slice_package", 1)
    filter_data(work, scope="package")
    second = work.session.image_array.copy()
    np.testing.assert_array_equal(second[0], first[0])
    np.testing.assert_array_equal(second[2], original[2])
    assert len(current_pipeline(work)["steps"]) == 2
    work.session.set_active_index("slice_package", 0)
    filter_data(work, scope="package", parameters={"sigma": 2})
    assert len(current_pipeline(work)["steps"]) == 2
    np.testing.assert_array_equal(work.session.image_array[1], second[1])
    assert not np.array_equal(work.session.image_array[0], first[0], equal_nan=True)
    change_pipeline(
        work,
        {"action": "remove", "step_id": current_pipeline(work)["steps"][0]["id"]},
        lambda *_: None,
    )
    np.testing.assert_array_equal(work.session.image_array[0], original[0])
    np.testing.assert_array_equal(work.session.image_array[1], second[1])
    np.testing.assert_array_equal(work.session.original_image_array, original)


@pytest.mark.parametrize(
    "plane, target", [("axial", (1, 2)), ("coronal", (1, 3)), ("sagittal", (1, 4))]
)
def test_viewed_image_filters_one_plane_without_changing_other_samples(plane, target):
    session = GuiSession()
    data = np.random.default_rng(33).normal(size=(3, 12, 14, 4)).astype(np.float32)
    session.add_experiment("Independent image", data, dims={"frame": 3, "x": 12, "y": 14, "z": 4})
    work = StudyWorkspace("image", "Image", session)
    session.set_image_plane(plane)
    session.set_active_index("frame", target[0])
    session.set_active_index(session.image_slice_dim, target[1])
    original = session.image_array.copy()
    axes = list(session.image_plane_dims)
    indices = dict(session.active_indices)
    selected = tuple(
        indices[name] if name not in axes else slice(None) for name in session.experiment_dims
    )
    mask = np.zeros(original.shape, bool)
    mask[selected] = True
    filter_data(work, scope="image", axes=axes)
    result = work.session.image_array
    np.testing.assert_array_equal(result[~mask], original[~mask])
    assert not np.array_equal(result[mask], original[mask])
    assert current_pipeline(work)["steps"][0]["options"]["plane_dims"] == axes


def test_current_image_selection_survives_navigation_bundle_and_workflow_replay(tmp_path):
    work = package_workspace(tmp_path / "source")
    work.session.set_active_index("slice_package", 1)
    work.session.set_active_index("frame", 1)
    work.session.set_active_index("z", 1)
    filter_data(work, scope="image")
    original = work.session.original_image_array.copy()
    processed = work.session.image_array.copy()
    options = copy.deepcopy(current_pipeline(work)["steps"][0]["options"])
    document, bindings = capture_workflow(work, "processing", name="Scoped denoising")
    work.session.set_active_index("slice_package", 2)
    path = save_workspace_bundle(work, tmp_path / "scoped.mrax")
    restored = load_workspace_bundle(path, tmp_path / "restored", key="restored")
    filter_data(restored, **options, step_id=current_pipeline(restored)["steps"][0]["id"])
    np.testing.assert_array_equal(restored.session.image_array, processed)
    fresh = package_workspace(tmp_path / "fresh")
    fresh.session.set_active_index("slice_package", 0)
    apply_workflow(fresh, document, ["1"], bindings, lambda *_: None)
    np.testing.assert_array_equal(fresh.session.original_image_array, original)
    np.testing.assert_array_equal(fresh.session.image_array, processed)
