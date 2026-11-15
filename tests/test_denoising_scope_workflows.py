"""Denoising targets are independent, replayable selections of synthetic data."""

import numpy as np
from test_bruker_geometry_workflows import _scan

from mrax.gui.services.pipeline import change_pipeline, current_pipeline
from mrax.gui.services.processing import run_processing
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


def test_viewed_image_filters_one_plane_without_changing_other_samples():
    session = GuiSession()
    data = np.random.default_rng(33).normal(size=(3, 12, 14, 4)).astype(np.float32)
    session.add_experiment("Independent image", data, dims={"frame": 3, "x": 12, "y": 14, "z": 4})
    work = StudyWorkspace("image", "Image", session)
    session.set_image_plane("coronal")
    session.set_active_index("frame", 1)
    session.set_active_index(session.image_slice_dim, 3)
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
