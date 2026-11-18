"""Read-only comparisons of independently generated images and pipeline states."""

import numpy as np

from mrax.data_struct import AcquisitionData, SpatialGeometry
from mrax.gui.services.analysis_workflows import run_operation
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.pipeline import change_pipeline
from mrax.gui.services.pipeline_blocks import block_options
from mrax.gui.services.pipeline_editing import apply_pipeline_draft
from mrax.gui.services.scene import scene_settings, update_scene
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession
from mrax.reco import fftnc


def synthetic():
    data = np.random.default_rng(19).normal(10, 2, (3, 7, 9)).astype(np.float32)
    session = GuiSession()
    session.create_empty_study("Independent comparison")
    session.add_experiment("Experiment", data, dims={"frame": 3, "x": 7, "y": 9})
    return StudyWorkspace(key="comparison", label=session.study_name, session=session), data


def test_comparison_requires_matching_physical_geometry():
    w, source = synthetic()
    first = SpatialGeometry(voxel_sizes=(1.0, 1.0), affine=np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 4
    second = SpatialGeometry(voxel_sizes=(1.0, 1.0), affine=shifted)
    w.session.add_experiment(
        "Reference",
        source,
        dims={"frame": 3, "x": 7, "y": 9},
        extra_metadata={"spatial_geometry": first.to_metadata()},
    )
    w.session.add_experiment(
        "Displaced",
        source,
        dims={"frame": 3, "x": 7, "y": 9},
        extra_metadata={"spatial_geometry": second.to_metadata()},
    )
    w.session.select_experiment("Reference")
    config = scene_settings(w)["comparison"]
    config["mode"] = "split"
    config["right"]["experiment"] = "Displaced"
    update_scene(w, {"comparison": config})
    arrays, info = render_comparison(w)
    assert arrays[0].shape == arrays[1].shape
    assert not info["geometry_compatible"]

    # Comparison preserves volume navigation and package coordinates.
    volume = np.random.default_rng(31).uniform(1, 2, (2, 3, 7, 9, 5)).astype(np.float32)
    session = GuiSession()
    session.add_experiment("Volumes", volume, dims={"slice_package": 2, "frame": 3, "x": 7, "y": 9, "z": 5})
    samples = np.asarray(fftnc(volume.reshape(6, 1, 7, 9, 5).astype(np.complex64), axes=(-3, -2, -1)))
    acquisition = AcquisitionData(samples, (7, 9, 5), ("x", "y", "z"))
    session.current_experiment.update(acquisition=acquisition, raw_kspace=samples)
    work = StudyWorkspace("volumes", "Volumes", session)
    options = block_options(work, "reconstruct", {})
    apply_pipeline_draft(work, {"base_generation": 0, "steps": [
        {"id": "first", "kind": "reconstruct", "options": options, "enabled": True},
    ]}, lambda *_: None)
    expected = {}
    for package in (0, 1):
        work.session.set_active_index("slice_package", package)
        work.session.set_active_index("frame", 2)
        for plane in ("axial", "coronal", "sagittal"):
            work.session.set_image_plane(plane)
            config = scene_settings(work)["comparison"]
            config.update(enabled=True, mode="overlay", plane=plane)
            for side, kind in (("left", "original"), ("right", "processed")):
                config[side] = {"experiment": "Volumes", "kind": kind}
            update_scene(work, {"comparison": config})
            arrays, info = render_comparison(work)
            assert info["geometry_compatible"]
            np.testing.assert_allclose(arrays[0], arrays[1], rtol=1e-5)
            np.testing.assert_allclose(arrays[0], work.session.frame_image(), rtol=1e-5)
            expected[package, plane] = arrays[0].copy()
    change_pipeline(work, {"action": "duplicate"}, lambda *_: None)
    identifier = work.pipelines["Volumes"]["id"]
    run_operation(work, "process", {"method": "gaussian", "input": "image", "axes": ["x", "y", "z"], "parameters": {"sigma": 0.8}}, lambda *_: None)
    for package in (0, 1):
        work.session.set_active_index("slice_package", package)
        work.session.set_active_index("frame", 2)
        for plane in ("axial", "coronal", "sagittal"):
            work.session.set_image_plane(plane)
            config = scene_settings(work)["comparison"]
            config.update(enabled=True, mode="overlay", plane=plane)
            for side, selected in (("left", "main"), ("right", identifier)):
                config[side] = {"experiment": "Volumes", "pipeline_id": selected, "kind": "processed"}
            update_scene(work, {"comparison": config})
            arrays, info = render_comparison(work)
            assert info["geometry_compatible"]
            np.testing.assert_allclose(arrays[0], expected[package, plane], rtol=1e-5)
            np.testing.assert_array_equal(arrays[1], work.session.frame_image())
