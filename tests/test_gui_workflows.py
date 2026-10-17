import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

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
    _fit_settings_from_dict,
    _load_experiment_metadata_template_from_file,
    _MainWindowMixin,
    _optimizer_choices_for_regularizers,
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


@pytest.mark.parametrize("preallocate", [None, "true", "false"])
def test_gui_allocator_default_precedes_session_import(preallocate):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    if preallocate is not None:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = preallocate
    script = """
import importlib.abc
import os
import sys

expected = sys.argv[1]
class ImportGuard(importlib.abc.MetaPathFinder):
    seen = False
    def find_spec(self, fullname, path, target=None):
        if fullname == 'mrax.gui.session':
            assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] == expected
            self.seen = True
guard = ImportGuard()
sys.meta_path.insert(0, guard)
import mrax.gui
assert guard.seen
assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] == expected
"""
    subprocess.run([sys.executable, "-c", script, preallocate or "false"], env=env, check=True, capture_output=True, timeout=45)


@pytest.fixture
def roi_editor():
    # Exercise controller edits and real session masks without a Qt event loop.
    editor = object.__new__(_MainWindowMixin)
    editor.session = GuiSession()
    editor.session.create_empty_study("synthetic", field_strength_mhz=100)
    editor.session.add_experiment("synthetic", np.ones((2, 7, 11, 3)), dims={"frame": 2, "x": 7, "y": 11, "z": 3})
    editor.view_combo = SimpleNamespace(currentText=lambda: "Image")
    editor._displayed_image = editor.session.frame_image()
    editor._displayed_view_key = "image"
    editor._displayed_image_label = "intensity"
    editor._displayed_image_unit = None
    editor.brush_mode = None
    editor._last_brush_stamp = None
    editor.brush_size_spin = SimpleNamespace(value=lambda: 1)
    editor.brush_label_spin = SimpleNamespace(value=lambda: 1)
    editor.brush_auto_subtract_check = None
    editor._task_running = False
    editor._busy_message = None
    editor.statusBar = Mock(return_value=Mock())
    for method in ("_begin_brush_stroke", "refresh_roi_list", "update_view", "mark_dirty", "_clear_brush_preview",
                   "_update_brush_preview", "_finish_brush_stroke"):
        setattr(editor, method, Mock())
    editor._left_mouse_pressed = lambda: True
    editor.image_view = SimpleNamespace(getView=lambda: SimpleNamespace(
        mapSceneToView=lambda point: point,
        sceneBoundingRect=lambda: SimpleNamespace(contains=lambda point: True),
    ))
    return editor


@pytest.mark.parametrize("plane", ["axial", "coronal", "sagittal"])
@pytest.mark.parametrize("label", [1, 7])
def test_brush_and_erase_edit_native_voxels_on_each_plane(roi_editor, plane, label):
    editor = roi_editor
    editor.session.active_image_plane = plane
    editor._displayed_image = editor.session.frame_image()
    shape = editor._displayed_image.shape
    editor.brush_label_spin.value = lambda: label
    editor.brush_mode = "brush"
    editor._apply_brush_at(shape[1] - 1, shape[0] - 1, shape)
    roi = editor.session.roi()
    assert roi is not None
    expected = np.zeros(editor.session.spatial_shape, dtype=int)
    index = tuple(editor._current_slice_indices().get(dim, 0) if dim not in editor._current_plane_dims()
                  else editor.session.spatial_shape[axis] - 1
                  for axis, dim in enumerate(editor.session.spatial_dim_names))
    expected[index] = label
    np.testing.assert_array_equal(roi.mask, expected)
    editor.brush_mode = "erase"
    editor._apply_brush_at(shape[1] - 1, shape[0] - 1, shape)
    assert not np.any(editor.session.roi().mask)


@pytest.mark.parametrize("view", ["Vendor vs MRax", "Vendor MRax Diff", "K Space", "MRax Reconstruction"])
def test_roi_tools_reject_non_image_views_without_editing_masks(roi_editor, view):
    editor = roi_editor
    editor.view_combo.currentText = lambda: view
    editor._displayed_image = np.zeros((7, 24))
    editor._displayed_view_key = view.lower().replace(" ", "_")
    for tool in (editor.enable_brush, editor.enable_erase, editor.enable_wand,
                 editor.add_polygon_roi, editor.add_lasso_roi, editor.add_flexible_shape_roi,
                 lambda: editor._add_editable_roi("rectangle"), editor._finish_polygon_roi):
        tool()
        assert editor.brush_mode is None
    # Switching views during a stroke must also prevent click/drag edits.
    editor.brush_mode = "brush"
    editor._apply_brush_at(1, 1, (7, 24))
    editor.on_image_clicked(None)
    editor.on_image_hovered(SimpleNamespace(x=lambda: 1, y=lambda: 1))
    assert not editor.session.rois
    editor._begin_brush_stroke.assert_not_called()
    editor._update_brush_preview.assert_not_called()
    assert "image and map views" in editor.statusBar().showMessage.call_args.args[0]


def test_roi_brush_rejects_stale_or_resampled_display_grid(roi_editor):
    editor = roi_editor
    editor.brush_mode = "brush"
    for displayed, key, shape in ((np.zeros((9, 13)), "image", (9, 13)),
                                  (editor.session.frame_image(), "registration_mask_preview", (7, 11)),
                                  (editor.session.frame_image(), "image", (7, 24))):
        editor._displayed_image = displayed
        editor._displayed_view_key = key
        editor._apply_brush_at(1, 1, shape)
        assert not editor.session.rois
    editor._begin_brush_stroke.assert_not_called()


def test_roi_brush_edits_native_mask_from_colored_map_view(roi_editor):
    editor = roi_editor
    editor.session.replace_fitted_maps({"level": np.ones(editor.session.spatial_shape)})
    editor.view_combo.currentText = lambda: "Map"
    editor._displayed_image = np.zeros((7, 11, 3))
    editor._displayed_view_key = "map"
    editor.brush_mode = "brush"
    editor._apply_brush_at(5, 3, (7, 11))
    roi = editor.session.roi()
    assert np.count_nonzero(roi.mask) == 1
    assert roi.mask[3, 5, 0]


def test_gui_bruker_detection_uses_file_based_experiment_discovery(tmp_path):
    pytest.importorskip("PySide6")
    from test_import_workflows import _write_minimal_bruker_scan

    from mrax.gui.app import _detect_bruker_import_summary, _looks_like_bruker_study

    _write_minimal_bruker_scan(tmp_path)
    exported = tmp_path / "exported_scan"
    (tmp_path / "1").rename(exported)
    for root in [tmp_path, exported]:
        assert _looks_like_bruker_study(root)
        assert [row["scan"] for row in _detect_bruker_import_summary(root)] == ["exported_scan"]
        session = GuiSession()
        session.import_bruker_study(root, field_strength_mhz=100, load_raw_kspace=False)
        assert list(session.data["experiments"]) == ["exported_scan"]


def test_gui_optimizer_defaults_preserve_auto_and_explicit_adam():
    assert GuiSession().fit_settings.optimizer_name == "auto"
    assert _fit_settings_from_dict({}).optimizer_name == "auto"
    assert _fit_settings_from_dict({"optimizer_name": "adam"}).optimizer_name == "adam"
    assert _optimizer_choices_for_regularizers({"tv"})[:3] == ["auto", "fista", "split_bregman"]
    assert "adam" in _optimizer_choices_for_regularizers({"tv"})
    assert _optimizer_choices_for_regularizers(set())[0] == "auto"


def test_roi_preview_renders_fair_axes_and_preserves_frame_selection():
    ti = np.asarray([70., 190., 410., 930., 1800.], dtype=np.float32)
    labels = np.asarray([0., 1.], dtype=np.float32)
    amp = np.asarray([1.3, 1.1])
    t1 = np.asarray([900., 1400.])
    signal = np.abs(amp - 2 * amp * np.exp(-ti[:, None] / t1)) + 1e-6
    mask = np.zeros((3, 4), dtype=bool)
    mask[1:, 1:3] = True
    data = np.where(mask, signal[:, :, None, None], 20.).astype(np.float32)
    session = GuiSession()
    session.create_empty_study("synthetic", field_strength_mhz=100)
    session.add_experiment("synthetic", data, dims={"ti": 5, "fair": 2, "x": 3, "y": 4},
                           exp_type="fair_asl", model_name="fair_asl", observables={
                               "TI": Quantity(ti, "ms"), "fair_label": Quantity(labels, "1"),
                           })
    session.add_roi(ROI("signal", mask, dims=("x", "y")))
    session.frame_mask.exclude(1, 6)
    result = session.run_preview_fit(model_name="fair_asl", roi_name="signal", num_steps=5)
    np.testing.assert_allclose(result["observed"], signal.reshape(-1), rtol=2e-6)
    np.testing.assert_array_equal(result["included"], session.frame_mask.as_array())
    fields = result["fields"]
    fitted_amp = np.asarray([np.asarray(fields[key].values).item() for key in ("amplitude_sel", "amplitude_non")])
    fitted_t1 = np.asarray([np.asarray(fields[key].values).item() for key in ("t1_sel", "t1_non")])
    expected = np.abs(fitted_amp - 2 * fitted_amp * np.exp(-ti[:, None] / fitted_t1)) + 1e-6
    np.testing.assert_allclose(result["predicted"], expected.reshape(-1), rtol=2e-6, atol=2e-7)
    assert len(result["x"]) == len(result["predicted"]) == 10
    assert np.isfinite(result["loss_history"]).all()


def test_gui_prepares_reconstruction_as_a_separate_requested_view():
    from mrax.data_struct import AcquisitionData
    from mrax.reco import fftnc

    image = np.arange(20, dtype=np.float32).reshape((1, 4, 5)) / 20 + 1
    acquisition = AcquisitionData(np.asarray(fftnc(jnp.asarray(image[:, None]))), (4, 5), ("phase", "read"))
    session = GuiSession()
    session.create_empty_study("synthetic")
    session.add_experiment("synthetic", image, dims={"frame": 1, "x": 4, "y": 5})
    session.current_experiment["acquisition"] = acquisition
    session.current_experiment["raw_kspace"] = acquisition.samples
    assert session.can_reconstruct_raw_data()
    assert session.reconstruction_array is None
    session.prepare_raw_view("k_space")
    assert session.reconstruction_array is None
    assert session.prepare_raw_view("mrax_reconstruction")
    np.testing.assert_allclose(session.reconstruction_array, image, atol=1e-6)
    np.testing.assert_array_equal(session.image_array, image)


def test_frequency_reconstruction_components_preserve_signed_values():
    from mrax.reco.components import reconstruction_variant

    frequency = np.asarray([[[-20., 0., 30.], [-5., 4., 12.]]])
    reconstructed = reconstruction_variant({"raw_reconstruction_frequency_hz": frequency}, component="magnitude")
    np.testing.assert_array_equal(reconstructed, frequency)


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
    [
        ("wayland-test", ":1", "wayland"),
        ("vscode-wayland-test.sock", ":1", "xcb"),
        ("vscode-wayland-test.sock", None, "wayland"),
        (None, ":1", "xcb"),
    ],
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
    if display is None:
        monkeypatch.delenv("DISPLAY", raising=False)
    else:
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


def test_gui_reconstruction_views_and_exports_combined_and_individual_coils(tmp_path):
    coil_1 = np.full((2, 2, 3), 2.0 + 2.0j, dtype=np.complex64)
    coil_2 = np.full((2, 2, 3), 1.0 - 1.0j, dtype=np.complex64)
    coil_1[1] = 2.0 - 2.0j
    coil_2[1] = 1.0 + 1.0j
    coil_images = np.stack([coil_1, coil_2], axis=1)
    combined_magnitude = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=1)).astype(np.float32)

    session = GuiSession()
    session.create_empty_study("reconstruction_components")
    session.add_experiment(
        "multi_coil",
        np.zeros((2, 2, 3), dtype=np.float32),
        dims={"frame": 2, "x": 2, "y": 3},
        processed={
            "raw_reconstruction": combined_magnitude,
            "raw_reconstruction_coils": coil_images,
        },
    )

    assert session.reconstruction_coil_count == 2
    np.testing.assert_allclose(
        session.reconstruction_image(component="magnitude", combined=True),
        combined_magnitude[0],
    )
    np.testing.assert_allclose(
        session.reconstruction_image(component="phase", combined=True),
        np.angle(coil_1[0] + coil_2[0]),
    )
    np.testing.assert_allclose(
        session.reconstruction_image(
            frame=1,
            component="phase",
            combined=True,
            unwrap_phase=False,
        ),
        np.angle(coil_1[1] + coil_2[1]),
    )
    session.set_reconstruction_coil(1)
    session.set_reconstruction_component("phase")
    session.set_reconstruction_phase_unwrap(False)
    np.testing.assert_allclose(session.reconstruction_image(), np.angle(coil_2[0]))
    assert session.dimension_controls("mrax_reconstruction") == [
        {"key": "frame", "label": "frame", "axis": 0, "size": 2}
    ]
    session.add_experiment(
        "other",
        np.zeros((2, 3), dtype=np.float32),
        dims={"x": 2, "y": 3},
    )
    session.select_experiment("multi_coil")
    assert session.active_reconstruction_component == "phase"
    assert session.active_reconstruction_coil == 1
    assert not session.active_reconstruction_unwrap_phase

    reconstruction_records = [
        record
        for record in _export_object_records_for_session(session)
        if record["kind"] == "reconstruction"
    ]
    assert [record["name"] for record in reconstruction_records] == [
        "combined_magnitude",
        "combined_phase",
        "combined_phase_wrapped",
        "coil_001_magnitude",
        "coil_001_phase",
        "coil_001_phase_wrapped",
        "coil_002_magnitude",
        "coil_002_phase",
        "coil_002_phase_wrapped",
    ]
    combined_phase_record = reconstruction_records[1]
    assert combined_phase_record["unit"] == "rad"
    saved = _write_single_export_object(
        combined_phase_record,
        tmp_path / "combined_phase.npy",
        "npy",
    )
    np.testing.assert_allclose(np.load(saved[0]), np.angle(coil_1 + coil_2))


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

    untagged = dict(payload)
    untagged.pop("mrax_version")
    warnings_list: list[str] = []
    _restore_gui_state(session, untagged, arrays, warnings=warnings_list)
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
