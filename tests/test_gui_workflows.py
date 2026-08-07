import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from mrax import __version__
from mrax.analysis import ROI
from mrax.core.loss_builder import make_model_callable
from mrax.data_struct import DataSet, Quantity, Study, save_study_results
from mrax.gui import GuiSession, import_mask, load_roi_set, save_roi_set
from mrax.gui.app import (
    StudyWorkspace,
    _apply_local_runtime_env,
    _apply_observable_dim_compatibility,
    _capture_pyqtgraph_image_view_state,
    _centered_top_left,
    _experiment_metadata_template_from_session,
    _export_object_records_for_session,
    _load_experiment_metadata_template_from_file,
    _rectangle_overlap_area,
    _resize_edge_names,
    _restore_gui_state,
    _restore_pyqtgraph_image_view_state,
    _roi_nifti_reference_summary,
    _roi_overlay_mask_for_display,
    _row_major_view_aspect_ratio,
    _screen_with_largest_overlap,
    _serialize_gui_state,
    _workspace_experiment_metadata_template_items,
    _write_single_export_object,
)
from mrax.io.nifti import load_nifti
from mrax.library import build_default_experiment, make_t2_se_model


def _write_small_manifest(tmp_path):
    te = jnp.asarray([12.0, 36.0], dtype=jnp.float32)
    model = make_t2_se_model(scale=1.0)
    experiment = build_default_experiment(
        "mse",
        frame_count=int(te.size),
        spatial_shape=(2, 2),
        observables={"TE": Quantity(te, "ms")},
    )
    truth = {
        "amplitude": jnp.full((2, 2), 1.0, dtype=jnp.float32),
        "t2": jnp.full((2, 2), 60.0, dtype=jnp.float32),
        "bias": jnp.zeros((2, 2), dtype=jnp.float32),
    }
    data = make_model_callable(experiment, model)(truth)
    study = Study(
        [experiment],
        [DataSet(data=data, name="gui_demo", kspace=jnp.fft.fft2(data))],
        [model],
        names=["gui_demo"],
        study_globals={"field_strength_mhz": Quantity(jnp.asarray(400.0), "MHz")},
    )
    return save_study_results(
        study,
        [{"name": "gui_demo", "model": model.name, "fields": {"t2": Quantity(truth["t2"], "ms")}}],
        tmp_path / "run",
        study_name="gui_smoke",
    )


def test_gui_window_geometry_prefers_majority_screen_and_centers_within_it():
    left_screen = (0, 0, 1920, 1080)
    right_screen = (1920, 0, 2560, 1440)
    mostly_right = (1500, 120, 1400, 900)

    assert _rectangle_overlap_area(mostly_right, left_screen) == 420 * 900
    assert _rectangle_overlap_area(mostly_right, right_screen) == 980 * 900
    assert _screen_with_largest_overlap(mostly_right, [left_screen, right_screen]) is right_screen
    assert _centered_top_left((1320, 860), right_screen) == (2540, 290)
    assert _resize_edge_names((16, 16), (16, 16, 800, 600)) == ("left", "top")
    assert _resize_edge_names((815, 615), (16, 16, 800, 600)) == ("right", "bottom")
    assert _resize_edge_names((400, 300), (16, 16, 800, 600)) == ()


@pytest.mark.parametrize(
    ("wayland_display", "display", "expected_platform"),
    [("wayland-test", ":1", "wayland"), (None, ":1", "xcb")],
)
def test_gui_runtime_selects_native_display_backend(
    monkeypatch,
    tmp_path,
    wayland_display,
    display,
    expected_platform,
):
    monkeypatch.setenv("MRAX_GUI_LOCAL_LIB_DIR", str(tmp_path / "missing-local-libs"))
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    if wayland_display is None:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    else:
        monkeypatch.setenv("WAYLAND_DISPLAY", wayland_display)
    monkeypatch.setenv("DISPLAY", display)

    _apply_local_runtime_env()

    assert os.environ["QT_QPA_PLATFORM"] == expected_platform


def test_gui_session_loads_manifest_and_extracts_roi_analysis(tmp_path):
    session = GuiSession.from_manifest(_write_small_manifest(tmp_path))

    assert session.experiment_names == ["gui_demo"]
    assert session.frame_count == 2
    assert session.frame_image().shape == (2, 2)
    assert session.kspace_image(mode="magnitude").shape == (2, 2)
    assert session.map_names == ["t2"]
    assert session.map_unit("t2") == "ms"

    roi = ROI("all", np.ones((2, 2), dtype=bool), dims=("x", "y"))
    session.add_roi(roi)
    rows = session.roi_stats(map_names=["t2"], stats=("mean",))
    timecourse = session.extract_timecourse("all")

    assert rows[0]["map"] == "t2"
    assert rows[0]["roi"] == "all"
    assert float(rows[0]["value"]) == 60.0
    assert np.asarray(timecourse["timecourse"]).shape == (2,)


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


@pytest.mark.parametrize(
    ("axis_labels", "native_plane", "expected_dims"),
    [
        (
            ["z", "x", "y"],
            "coronal",
            {"axial": ("y", "z"), "coronal": ("y", "x"), "sagittal": ("z", "x")},
        ),
        (
            ["y", "z", "x"],
            "sagittal",
            {"axial": ("z", "x"), "coronal": ("z", "y"), "sagittal": ("x", "y")},
        ),
    ],
)
def test_gui_session_orders_anatomical_planes_from_geometry(
    axis_labels,
    native_plane,
    expected_dims,
):
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape((2, 3, 4, 5))
    session = GuiSession()
    session.create_empty_study("native_orientation")
    session.add_experiment(
        "volume",
        data,
        dims={"frame": 2, "x": 3, "y": 4, "z": 5},
        extra_metadata={
            "spatial_geometry": {
                "spatial_shape": [3, 4, 5],
                "voxel_sizes": [0.2, 0.3, 1.1],
                "axis_anatomical_labels": axis_labels,
                "source": "synthetic",
                "status": "unknown",
            },
        },
    )

    assert session.active_image_plane == native_plane
    assert session.native_image_plane == native_plane
    for plane, dims in expected_dims.items():
        session.set_image_plane(plane)
        assert session.image_plane_dims == dims


def test_gui_session_uses_geometry_spacing_for_ordered_anatomical_planes():
    session = GuiSession()
    session.create_empty_study("physical_spacing")
    session.add_experiment(
        "volume",
        np.zeros((3, 4, 5), dtype=np.float32),
        dims={"x": 3, "y": 4, "z": 5},
        extra_metadata={
            "spatial_geometry": {
                "version": 1,
                "axis_convention": "native_v1",
                "affine_convention": "nifti_xyz_ras",
                "status": "unknown",
                "source": "synthetic",
                "spatial_shape": [3, 4, 5],
                "voxel_sizes": [0.2, 0.3, 1.1],
                "axis_anatomical_labels": ["z", "x", "y"],
            }
        },
    )

    assert session.active_image_plane == "coronal"
    assert session.image_plane_dims == ("y", "x")
    assert session.image_plane_voxel_sizes == pytest.approx((0.3, 0.2))
    session.set_image_plane("axial")
    assert session.image_plane_dims == ("y", "z")
    assert session.image_plane_voxel_sizes == pytest.approx((0.3, 1.1))
    session.set_image_plane("sagittal")
    assert session.image_plane_dims == ("z", "x")
    assert session.image_plane_voxel_sizes == pytest.approx((1.1, 0.2))


def test_row_major_image_aspect_uses_column_over_row_spacing():
    shape = (28, 256)
    spacings = (0.75, 0.15625)

    ratio = _row_major_view_aspect_ratio(spacings)

    assert ratio == pytest.approx(spacings[1] / spacings[0])
    displayed_width_to_height = shape[1] * ratio / shape[0]
    physical_width_to_height = (shape[1] * spacings[1]) / (shape[0] * spacings[0])
    assert displayed_width_to_height == pytest.approx(physical_width_to_height)


def test_gui_state_saves_mrax_version_and_spatial_geometry():
    session = GuiSession()
    session.create_empty_study("versioned_state")
    session.add_experiment(
        "volume",
        np.zeros((3, 4), dtype=np.float32),
        dims={"x": 3, "y": 4},
        extra_metadata={
            "spatial_geometry": {
                "version": 1,
                "axis_convention": "native_v1",
                "affine_convention": "nifti_xyz_ras",
                "status": "unknown",
                "source": "synthetic",
                "spatial_shape": [3, 4],
                "voxel_sizes": [0.2, 0.3],
            }
        },
    )
    payload, arrays = _serialize_gui_state(
        session,
        preview_results={},
        fit_results={},
    )

    assert payload["mrax_version"] == __version__
    assert payload["spatial_geometry"]["volume"]["voxel_sizes"] == [0.2, 0.3]

    incompatible = dict(payload)
    major, minor, _patch = (int(value) for value in __version__.split(".")[:3])
    incompatible["mrax_version"] = (
        f"0.{minor + 1}.0" if major == 0 else f"{major + 1}.0.0"
    )
    with pytest.raises(ValueError, match="MRax version incompatibility"):
        _restore_gui_state(session, incompatible, arrays)

    legacy = dict(payload)
    legacy.pop("mrax_version")
    warnings_list: list[str] = []
    _restore_gui_state(session, legacy, arrays, warnings=warnings_list)
    assert any("no 'mrax_version' tag" in message for message in warnings_list)


def test_gui_roi_overlay_follows_selected_3d_plane():
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[1, 2, 4] = True
    roi = ROI("point", mask, dims=("x", "y", "z"))

    axial = _roi_overlay_mask_for_display(
        roi,
        display_shape=(3, 4),
        indices={"z": 4},
        plane_dims=("x", "y"),
    )
    coronal = _roi_overlay_mask_for_display(
        roi,
        display_shape=(3, 5),
        indices={"y": 2},
        plane_dims=("x", "z"),
    )
    sagittal = _roi_overlay_mask_for_display(
        roi,
        display_shape=(4, 5),
        indices={"x": 1},
        plane_dims=("y", "z"),
    )

    assert axial is not None and axial[1, 2]
    assert coronal is not None and coronal[1, 4]
    assert sagittal is not None and sagittal[2, 4]


def test_gui_roi_mask_round_trip_and_raw_import(tmp_path):
    mask = np.asarray([[True, False], [False, True]])
    roi = ROI("diag", mask, dims=("x", "y"), metadata={"color": "#abcdef"})

    npz_path, json_path = save_roi_set([roi], tmp_path / "rois.npz", metadata={"study": "demo"})
    loaded = load_roi_set(json_path)
    imported = import_mask(mask.T, name="transposed", spatial_shape=(2, 2), dims=("x", "y"))

    assert npz_path.exists()
    assert json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["mrax_version"] == __version__
    assert loaded[0].name == "diag"
    assert loaded[0].dims == ("x", "y")
    assert loaded[0].metadata["color"] == "#abcdef"
    np.testing.assert_array_equal(loaded[0].mask, mask)
    np.testing.assert_array_equal(imported.mask, mask)


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

    roi_record = next(record for record in _export_object_records_for_session(session) if record["kind"] == "roi")
    output = _write_single_export_object(roi_record, tmp_path / "mask.nii", "nifti")[0]
    loaded_mask, loaded_affine = load_nifti(output)

    np.testing.assert_array_equal(loaded_mask, mask.astype(np.float32))
    np.testing.assert_allclose(loaded_affine, affine, atol=1e-6)
    assert roi_record["nifti_reference"]["identity_fallback"] is False
    assert "reference_scan: 1 mask(s), reference affine" in _roi_nifti_reference_summary(session)


def test_gui_roi_nifti_export_keeps_identity_fallback_without_reference_geometry():
    session = GuiSession()
    session.create_empty_study("identity_export")
    session.add_experiment("reference_scan", np.zeros((2, 2)), dims={"x": 2, "y": 2})
    session.add_roi(ROI("mask", np.ones((2, 2), dtype=bool), dims=("x", "y")))

    roi_record = next(record for record in _export_object_records_for_session(session) if record["kind"] == "roi")

    np.testing.assert_array_equal(roi_record["nifti_reference"]["affine"], np.eye(4))
    assert roi_record["nifti_reference"]["identity_fallback"] is True
    assert "reference_scan: 1 mask(s), identity fallback" in _roi_nifti_reference_summary(session)


def test_pyqtgraph_image_view_state_round_trip_preserves_range_and_levels():
    class FakeViewBox:
        def __init__(self):
            self._view_range = [[1.0, 5.0], [2.0, 8.0]]
            self.restored = None

        def viewRange(self):
            return [list(self._view_range[0]), list(self._view_range[1])]

        def setRange(self, *, xRange=None, yRange=None, padding=None):
            self.restored = (tuple(xRange), tuple(yRange), padding)

    class FakeHistogram:
        def __init__(self):
            self._levels = (0.25, 0.75)
            self.restored = None

        def getLevels(self):
            return self._levels

        def setLevels(self, lo, hi):
            self.restored = (float(lo), float(hi))

    class FakeImageView:
        def __init__(self):
            self.view = FakeViewBox()
            self.hist = FakeHistogram()
            self.restored_levels = None

        def getView(self):
            return self.view

        def getHistogramWidget(self):
            return self.hist

        def setLevels(self, lo, hi):
            self.restored_levels = (float(lo), float(hi))

    image_view = FakeImageView()
    state = _capture_pyqtgraph_image_view_state(image_view)

    assert state == {"x_range": (1.0, 5.0), "y_range": (2.0, 8.0), "levels": (0.25, 0.75)}

    image_view.view._view_range = [[0.0, 10.0], [0.0, 10.0]]
    image_view.hist._levels = (0.0, 1.0)
    _restore_pyqtgraph_image_view_state(image_view, state)

    assert image_view.restored_levels == (0.25, 0.75)
    assert image_view.hist.restored == (0.25, 0.75)
    assert image_view.view.restored == ((1.0, 5.0), (2.0, 8.0), 0)


def test_experiment_metadata_template_from_loaded_session():
    session = GuiSession()
    session.create_empty_study("template_demo", field_strength_mhz=400.0)
    session.add_experiment(
        "source_scan",
        np.zeros((2, 3, 4), dtype=np.float32),
        dims={"frame": 2, "x": 3, "y": 4},
        exp_type="t2_se",
        model_name="identity",
        observables={"TE": Quantity(jnp.asarray([10.0, 20.0]), "ms")},
        extra_metadata={"method": {"EchoTime": [10.0, 20.0]}},
    )

    template = _experiment_metadata_template_from_session(session, "source_scan")

    assert template["exp_type"] == "t2_se"
    assert template["model"] == "identity"
    assert template["dims"] == {"frame": 2, "x": 3, "y": 4}
    assert template["extra_metadata"]["method"]["EchoTime"] == [10.0, 20.0]
    np.testing.assert_allclose(np.asarray(template["observables"]["TE"].values), [10.0, 20.0])
    assert template["observables"]["TE"].si_unit == "ms"


def test_workspace_metadata_template_choices_include_all_loaded_studies():
    session_a = GuiSession()
    session_a.create_empty_study("study_a", field_strength_mhz=400.0)
    session_a.add_experiment(
        "scan_a",
        np.zeros((1, 2, 2), dtype=np.float32),
        dims={"frame": 1, "x": 2, "y": 2},
        exp_type="t2_se",
        model_name="identity",
    )
    session_b = GuiSession()
    session_b.create_empty_study("study_b", field_strength_mhz=400.0)
    session_b.add_experiment(
        "scan_b",
        np.zeros((1, 3, 3), dtype=np.float32),
        dims={"frame": 1, "x": 3, "y": 3},
        exp_type="t1_ir",
        model_name="identity",
    )
    workspaces = {
        "a": StudyWorkspace(key="a", label="Study A", session=session_a),
        "b": StudyWorkspace(key="b", label="Study B", session=session_b),
    }

    items = _workspace_experiment_metadata_template_items(workspaces, active_key="a")

    assert ("Current study: Study A / scan_a", ("workspace", "a", "scan_a")) in items
    assert ("Study B / scan_b", ("workspace", "b", "scan_b")) in items


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


def test_observable_compatibility_trims_and_extends_observables():
    dims = {"frame": 3, "x": 2, "y": 2}
    trimmed, trim_lines = _apply_observable_dim_compatibility(
        {"TE": Quantity(jnp.asarray([10.0, 20.0, 30.0, 40.0]), "ms")},
        dims,
        exp_type="t2_se",
        model_name="identity",
        policy="trim",
    )
    extended, extend_lines = _apply_observable_dim_compatibility(
        {"TE": Quantity(jnp.asarray([10.0, 20.0]), "ms")},
        dims,
        exp_type="t2_se",
        model_name="identity",
        policy="extend",
    )

    np.testing.assert_allclose(np.asarray(trimmed["TE"].values), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(np.asarray(extended["TE"].values), [10.0, 20.0, 20.0])
    assert any("trimmed" in line for line in trim_lines)
    assert any("extended" in line for line in extend_lines)


def test_observable_compatibility_resizes_frame_tables():
    dims = {"frame": 3, "x": 2, "y": 2}
    bvecs = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    adjusted, _lines = _apply_observable_dim_compatibility(
        {"bvecs": Quantity(bvecs, "unitless")},
        dims,
        exp_type="dti",
        model_name="identity",
        policy="trim",
    )

    assert np.asarray(adjusted["bvecs"].values).shape == (3, 3)
    np.testing.assert_allclose(np.asarray(adjusted["bvecs"].values), np.asarray(bvecs)[:3])


def test_experiment_metadata_template_from_metadata_file_and_manifest(tmp_path):
    metadata = {
        "mrax_version": __version__,
        "model": "identity",
        "exp_type": "t2_se",
        "dataset_name": "scan_a",
        "dims": {"frame": 2, "x": 3, "y": 4},
        "observable_values": {"TE": {"unit": "ms", "values": [10.0, 20.0]}},
        "observable_units": {"TE": "ms"},
        "extra_metadata": {"source": "saved_metadata"},
    }
    metadata_path = tmp_path / "experiments" / "scan_a" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest_path = tmp_path / "study_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mrax_version": __version__,
                "experiments": {
                    "scan_a": {"metadata": "experiments/scan_a/metadata.json"}
                },
            }
        ),
        encoding="utf-8",
    )

    direct = _load_experiment_metadata_template_from_file(metadata_path)
    from_manifest = _load_experiment_metadata_template_from_file(manifest_path)

    assert direct["dims"] == {"frame": 2, "x": 3, "y": 4}
    assert direct["extra_metadata"] == {"source": "saved_metadata"}
    np.testing.assert_allclose(np.asarray(direct["observables"]["TE"].values), [10.0, 20.0])
    assert from_manifest["name"] == "scan_a"
    assert from_manifest["observables"]["TE"].si_unit == "ms"
