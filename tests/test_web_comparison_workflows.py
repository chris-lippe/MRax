"""Read-only comparisons of independently generated images and pipeline states."""

import numpy as np

from mrax.data_struct import SpatialGeometry
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.scene import scene_settings, update_scene
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession


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
