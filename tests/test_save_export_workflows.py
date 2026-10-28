"""Synthetic save export contracts and numerical references."""

import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from mrax import MRAX_VERSION, MRAX_VERSION_TAG, __version__
from mrax.analysis.group import discover_study_manifests
from mrax.data_struct import (
    DataSet,
    Experiment,
    Quantity,
    SpatialGeometry,
    Study,
    load_study_results,
    resolve_spatial_geometry,
    save_study_results,
)
from mrax.data_struct.dimensions import build_experiment_dims
from mrax.data_struct.storage import _resolve_manifest_path
from mrax.gui import GuiSession
from mrax.gui.app import (
    _export_object_records_for_session,
    _MainWindowMixin,
    _mrax_package_entries,
    _single_export_file_types,
    _write_single_export_object,
)
from mrax.gui.plotting import axis_coordinates, image_profile, profile_selection
from mrax.io.bundle import extract_bundle, write_bundle
from mrax.io.image_export import export_image_files
from mrax.io.nifti import (
    export_nifti_4d,
    export_nifti_series,
    load_nifti,
    mrax_to_nifti_volume,
    nifti_spatial_geometry,
    save_nifti,
)
from mrax.library import make_t2_se_model
from mrax.versioning import versioned_payload


def small_study():
    experiment = Experiment(dims={"te": 3, "x": 2, "y": 4}, exp_type="mse",
                            observables={"TE": Quantity([10., 30., 70.], "ms")})
    data = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    return Study([experiment], [DataSet(data)], [make_t2_se_model()], names=["synthetic"],
                 study_globals={"field_strength_mhz": Quantity(100., "MHz")})


def test_package_version_citation_and_distribution():
    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    assert MRAX_VERSION == __version__
    assert MRAX_VERSION_TAG == 'mrax_version'
    assert project['project']['dynamic'] == ['version']
    assert project['tool']['setuptools']['dynamic']['version'] == {'attr': 'mrax._version.MRAX_VERSION'}
    assert versioned_payload({}) == {MRAX_VERSION_TAG: MRAX_VERSION}
    root = Path(__file__).parents[1]
    citation = (root / 'CITATION.cff').read_text()
    assert '\ndoi: 10.5281/zenodo.21723327\n' in citation
    assert not any((line.startswith(('version:', 'date-released:')) for line in citation.splitlines()))
    assert 'version-specific DOI' in citation
    manifest = (root / 'MANIFEST.in').read_text()
    assert 'prune data\n' in manifest
    assert 'prune local_testing\n' in manifest
    exclusions = (root / '.dockerignore').read_text().splitlines()
    assert 'data/' in exclusions and 'local_testing/' in exclusions


def test_storage_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match='outside manifest directory'):
        _resolve_manifest_path(tmp_path / 'study', '../secret.npy', label='image')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'mrax_package.json').write_text(json.dumps({'format': 'mrax.gui.package', 'version': 1, 'mrax_version': __version__, 'entries': {'gui_state': '../outside.json'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='outside package directory'):
        _mrax_package_entries(package)


def test_nifti_single_volume_and_series_round_trip(tmp_path):
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    affine = np.diag([1.5, 2.0, 1.0, 1.0])
    path = tmp_path / "image.nii.gz"
    save_nifti(path, image, affine)
    loaded, loaded_affine = load_nifti(path)
    np.testing.assert_allclose(loaded, image)
    np.testing.assert_allclose(loaded_affine, affine, atol=1e-6)
    geometry = nifti_spatial_geometry(loaded_affine, spatial_shape=image.shape)
    assert geometry.voxel_sizes == (2.0, 1.5)
    assert geometry.axis_anatomical_labels == ("y", "x")

    experiment = Experiment(
        dims={"echo": 2, "x": 3, "y": 4},
        observables={"TE": Quantity(jnp.asarray([10.0, 20.0]), "ms")},
        exp_type="mse",
    )
    dataset = DataSet(jnp.stack([jnp.asarray(image), jnp.asarray(image + 1)]), name="nifti")
    outputs = export_nifti_series(experiment, dataset, tmp_path / "series", affine=affine)
    output_4d = export_nifti_4d(experiment, dataset, tmp_path / "series_4d.nii.gz", affine=affine)

    assert len(outputs) == 2
    assert all(path.is_file() for path in outputs)
    assert output_4d.is_file()


def test_nifti_export_resolves_dataset_reference_and_identity_geometry(tmp_path):
    affine = np.asarray(
        [[0.0, -0.7, 0.0, 12.0], [0.8, 0.0, 0.0, -4.0], [0.0, 0.0, 1.2, 3.0], [0, 0, 0, 1]]
    )
    known = SpatialGeometry(
        affine=affine,
        spatial_shape=(2, 3),
        voxel_sizes=(0.8, 0.7),
        source="synthetic_reference",
        status="known",
    )
    unknown = SpatialGeometry(spatial_shape=(2, 3), source="synthetic", status="unknown")
    resolved = resolve_spatial_geometry(unknown, reference=known)
    np.testing.assert_allclose(resolved.affine, affine)
    assert resolved.source == "reference:synthetic_reference"
    assert resolved.identity_fallback is False

    incompatible = SpatialGeometry(spatial_shape=(3, 2), source="synthetic", status="unknown")
    fallback = resolve_spatial_geometry(incompatible, reference=known)
    np.testing.assert_array_equal(fallback.affine, np.eye(4))
    assert fallback.identity_fallback is True

    experiment = Experiment({"x": 2, "y": 3}, exp_type="anatomical")
    dataset = DataSet(jnp.ones((2, 3)), "geometry", spatial_geometry=known)
    output = export_nifti_series(experiment, dataset, tmp_path / "geometry")
    _, loaded_affine = load_nifti(output[0])
    np.testing.assert_allclose(loaded_affine, affine, atol=1e-6)


def test_default_study_save_is_self_contained_bundle_without_gui(tmp_path):
    study = small_study()
    events = []
    output = study.save(tmp_path / "synthetic.mrax", progress_callback=lambda *args: events.append(args))
    assert output.suffix == ".mrax"
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "study_manifest.json" in names
        assert all("gui" not in name for name in names)
    restored = load_study_results(output)
    np.testing.assert_array_equal(restored["experiments"]["synthetic"]["mr_image"], study.datasets[0].data)
    assert GuiSession.from_manifest(output).frame_count == 3
    assert discover_study_manifests(tmp_path)["synthetic"] == output
    assert len(events) > 3 and events[-1][0] == events[-1][1]
    fractions = [current / total for current, total, _ in events]
    assert np.all(np.diff(fractions) >= 0)


def test_directory_save_remains_explicit(tmp_path):
    path = save_study_results(small_study(), [{"name": "synthetic", "fields": {}}], tmp_path, bundle=False)
    assert path.name == "study_manifest.json"
    assert load_study_results(path)["experiments"]


def test_bundle_failure_preserves_previous_file_and_rejects_traversal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "array.bin").write_bytes(b"x" * 2048)
    target = tmp_path / "study.mrax"
    target.write_bytes(b"previous contents")
    def cancel(*_args):
        raise RuntimeError("cancel")
    with pytest.raises(RuntimeError, match="cancel"):
        write_bundle(target, source, progress_callback=cancel)
    assert target.read_bytes() == b"previous contents"
    assert not list(tmp_path.glob("*.partial"))
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("../outside", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(target, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("frames", [1, 2])
def test_nifti_export_splits_frames_and_squeezes_only_nonspatial_singletons(tmp_path, frames):
    dims = {"t": frames, "ti": 1, "x": 3, "y": 5, "z": 1}
    data = np.arange(frames * 15, dtype=np.float32).reshape(tuple(dims.values()))
    affine = np.diag([2., 3., 4., 1.])
    record = {"array": data, "dims": dims, "kind": "image", "nifti_reference": {"affine": affine}}
    assert {row["id"] for row in _single_export_file_types(record)} >= {"nifti", "dicom"}
    paths = _write_single_export_object(record, tmp_path / "image.nii", "nifti")
    assert len(paths) == frames
    for index, path in enumerate(paths):
        restored, matrix = load_nifti(path)
        np.testing.assert_array_equal(restored, data[index, 0])
        np.testing.assert_array_equal(matrix, affine)
    assert paths[0].name == ("image.nii" if frames == 1 else "image_t-0000.nii")


def test_dicom_export_preserves_calibrated_pixels_units_and_geometry(tmp_path):
    import pydicom
    data = np.linspace(-12.5, 93.7, 2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    affine = np.diag([1.2, 2.3, 3.4, 1.])
    affine[:3, 3] = [10., -20., 30.]
    paths = export_image_files(tmp_path / "image.dcm", data, dims={"t": 2, "x": 3, "y": 4, "z": 5},
                               affine=affine, format="dicom", unit="Hz")
    assert len(paths) == 6
    series = [pydicom.dcmread(path) for path in paths]
    assert len({ds.StudyInstanceUID for ds in series}) == 1
    assert len({ds.FrameOfReferenceUID for ds in series}) == 1
    assert len({ds.SeriesInstanceUID for ds in series}) == 2
    for index in range(2):
        expected = mrax_to_nifti_volume(data[index])
        for z, ds in enumerate(series[index * 3:(index + 1) * 3]):
            pixels = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            np.testing.assert_allclose(pixels, expected[:, :, z].T, atol=float(ds.RescaleSlope) * .51)
            np.testing.assert_allclose(ds.ImagePositionPatient, [-10., 20., 30. + z * 3.4])
            np.testing.assert_allclose(ds.PixelSpacing, [2.3, 1.2])
            assert ds.RescaleType == "Hz" and not ds.PatientID


def test_image_export_uses_per_frame_affines(tmp_path):
    matrices = [np.eye(4), np.eye(4)]
    matrices[1][0, 3] = 12.
    geometry = {"spatial_shape": [3, 5], "frame_affines": [matrix.tolist() for matrix in matrices]}
    outputs = export_image_files(tmp_path / "image.nii", np.ones((2, 3, 5)), dims={"t": 2, "x": 3, "y": 5}, geometry=geometry)
    for index, path in enumerate(outputs):
        np.testing.assert_array_equal(load_nifti(path)[1], matrices[index])


def test_plot_choices_follow_sliders_and_profiles_follow_other_selected_axes():
    session = GuiSession()
    session.create_empty_study("synthetic", field_strength_mhz=100)
    dims = {"ti": 3, "fair": 2, "x": 3, "y": 5, "z": 4}
    data = np.arange(np.prod(list(dims.values()))).reshape(tuple(dims.values()))
    session.add_experiment("synthetic", data, dims=dims, exp_type="fair_asl", model_name="fair_asl",
                           observables={"TI": Quantity([100., 300., 900.], "ms"),
                                        "fair_label": Quantity([0., 1.], "1"), "TR": Quantity(np.ones(6) * 1000, "ms")})
    editor = object.__new__(_MainWindowMixin)
    editor.session = session
    editor.view_combo = SimpleNamespace(currentText=lambda: "Image")
    assert editor._plot_x_axis_choices() == [("Auto", "auto"), ("ti", "ti"), ("fair", "fair"), ("z", "z")]
    indices = {"fair": 1, "z": 2}
    values = image_profile(data, session.build_experiment(), "ti", indices)
    np.testing.assert_allclose(values, np.mean(data[:, 1, :, :, 2], axis=(1, 2)))
    assert axis_coordinates(session.build_experiment(), "ti")[0] == "TI (ms)"
    _, frame_ids = profile_selection(session.build_experiment(), "ti", indices)
    np.testing.assert_array_equal(frame_ids, [1, 3, 5])


def test_elapsed_time_is_cumulative_but_variable_tr_axis_retains_parameter_values():
    obs = {"TR": Quantity([5., 7., 9.], "ms")}
    dynamic = Experiment(dims={"t": 3, "x": 2, "y": 4}, observables=obs, exp_type="dce")
    label, times = axis_coordinates(dynamic, "t")
    assert label == "Elapsed time (s)"
    np.testing.assert_allclose(times, [0., .005, .012])
    vtr = Experiment(dims={"tr": 3, "x": 2, "y": 4}, observables=obs, exp_type="vtr")
    label, values = axis_coordinates(vtr, "tr")
    assert label == "TR (ms)"
    np.testing.assert_array_equal(values, [5., 7., 9.])
    assert list(build_experiment_dims(3, (2, 4), obs, exp_type="dce")) == ["t", "x", "y"]


def test_guarded_save_enters_busy_state_before_work_and_clears_on_error():
    editor = object.__new__(_MainWindowMixin)
    editor._task_running = False
    editor._busy_message = None
    events = []
    def begin(message):
        editor._task_running = True
        events.append("busy")
    def end():
        editor._task_running = False
        events.append("idle")
    editor._begin_busy, editor._end_busy = begin, end
    editor.on_task_progress = Mock()
    editor.show_error = Mock()
    editor.statusBar = Mock(return_value=Mock())
    def fail(progress):
        assert editor.is_busy
        events.append("write")
        raise RuntimeError("write failed")
    editor.run_guarded_task(fail, Mock(), "Saving study")
    assert events == ["busy", "write", "idle"] and not editor.is_busy
    editor.show_error.assert_called_once()


def test_nifti_integer_label_export_preserves_values(tmp_path):
    data = np.arange(15, dtype=np.int64).reshape((1, 3, 5))
    paths = export_image_files(tmp_path / "labels.nii", data, dims={"t": 1, "x": 3, "y": 5})
    np.testing.assert_array_equal(load_nifti(paths[0])[0], data[0])


def test_gui_image_record_exports_reference_geometry(tmp_path):
    session = GuiSession()
    session.create_empty_study("synthetic", field_strength_mhz=100)
    affine = np.diag([2., 3., 4., 1.])
    session.add_experiment("synthetic", np.ones((2, 3, 5)), dims={"t": 2, "x": 3, "y": 5},
                           extra_metadata={"spatial_geometry": {"affine": affine.tolist(), "spatial_shape": [3, 5]}})
    record = next(row for row in _export_object_records_for_session(session) if row["kind"] == "image")
    paths = _write_single_export_object(record, tmp_path / "image.nii", "nifti")
    assert len(paths) == 2
    for path in paths:
        np.testing.assert_array_equal(load_nifti(path)[1], affine)


@pytest.mark.parametrize("ti_count", [2, 4])
def test_repeat_parameter_map_export_omits_explained_acquisition_axes(tmp_path, ti_count):
    data = np.arange(2 * 3 * 5).reshape((2, 3, 5))
    paths = export_image_files(tmp_path / "map.nii", data, dims={"t": 2, "ti": ti_count, "x": 3, "y": 5})
    assert len(paths) == 2
    for index, path in enumerate(paths):
        np.testing.assert_array_equal(load_nifti(path)[0], data[index])


def test_bundle_optional_nifti_exports_repeated_parameter_maps(tmp_path):
    study = small_study()
    experiment = study.experiments[0]
    experiment.dims = {"t": 2, **experiment.dims}
    study.datasets[0].data = np.ones(tuple(experiment.dims.values()), dtype=np.float32)
    results = [{"name": "synthetic", "fields": {"amplitude": Quantity(np.ones((2, 2, 4)), "1")}}]
    output = save_study_results(study, results, tmp_path / "study.mrax", save_nifti=True)
    restored = load_study_results(output)
    assert not restored["warnings"]
    with zipfile.ZipFile(output) as archive:
        assert len([name for name in archive.namelist() if name.endswith(".nii")]) == 8


def test_gui_snapshot_saves_current_data_without_copying_unrelated_files(tmp_path):
    study = small_study()
    original = study.save(tmp_path / "original", bundle=False)
    (original.parent / "unrelated.bin").write_bytes(b"unrelated")
    session = GuiSession.from_manifest(original)
    updated = np.full_like(session.image_array, 7)
    session.replace_image_data(updated)
    session.replace_fitted_maps({"amplitude": Quantity(np.ones((2, 4)), "1")})
    editor = object.__new__(_MainWindowMixin)
    editor.session = session
    editor.on_task_progress = Mock()
    output = editor._save_gui_state_data_snapshot(tmp_path / "snapshot")
    restored = load_study_results(output)["experiments"]["synthetic"]
    np.testing.assert_array_equal(restored["mr_image"], updated)
    np.testing.assert_array_equal(restored["fitted_maps"]["amplitude"].values, np.ones((2, 4)))
    assert not (output.parent / "unrelated.bin").exists()


def test_dicom_unsigned_integer_images_are_lossless(tmp_path):
    import pydicom
    data = np.asarray([[0, 65535], [32768, 42]], dtype=np.uint16)
    path = export_image_files(tmp_path / "image.dcm", data, format="dicom")[0]
    restored = pydicom.dcmread(path)
    np.testing.assert_array_equal(restored.pixel_array, data)
    assert restored.PixelRepresentation == 0 and float(restored.RescaleSlope) == 1.


def test_bulk_exports_include_dicom_and_multiframe_nifti_maps(tmp_path):
    session = GuiSession()
    session.create_empty_study("synthetic", field_strength_mhz=100)
    session.add_experiment("synthetic", np.ones((2, 3, 5)), dims={"t": 2, "x": 3, "y": 5})
    session.replace_fitted_maps({"level": Quantity(np.full((2, 3, 5), 7.), "1")})
    editor = object.__new__(_MainWindowMixin)
    editor.session = session
    editor.record_diagnostic_event = Mock()
    paths = editor.export_current_study(tmp_path, include_images=True, include_kspace=False,
                                        include_maps=True, include_rois=False, export_numpy=False,
                                        export_nifti=True, export_dicom=True, include_reconstructions=False)
    assert len([path for path in paths if path.suffix == ".dcm"]) == 4
    assert len([path for path in paths if path.suffix == ".nii"]) == 4
    assert all(path.is_file() for path in paths)


def test_spatial_plot_axis_uses_spacing_in_millimeters():
    experiment = Experiment(dims={"frame": 1, "x": 3, "y": 5, "z": 4})
    label, values = axis_coordinates(experiment, "z", geometry={"spatial_shape": [3, 5, 4], "voxel_sizes": [1., 2., 3.]})
    assert label == "z distance (mm)"
    np.testing.assert_array_equal(values, [0., 3., 6., 9.])


def test_headless_gui_save_feedback_and_map_axis_refresh(tmp_path):
    script = """
import sys
from pathlib import Path
import numpy as np
try:
    from mrax.gui._deps import import_qt, import_pyqtgraph
    QtCore, QtGui, QtWidgets = import_qt()
    pg = import_pyqtgraph()
except Exception as error:
    print(error)
    sys.exit(77)
from mrax.gui.app import _build_main_window
pg.setConfigOptions(imageAxisOrder='row-major')
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
window = _build_main_window(QtCore, QtGui, QtWidgets, pg)()
sys.excepthook = sys.__excepthook__
window.session.create_empty_study('synthetic', field_strength_mhz=100)
window.session.add_experiment('synthetic', np.ones((3, 7, 11)), dims={'t': 3, 'x': 7, 'y': 11})
window.show()
app.processEvents()
output = Path(sys.argv[1])
def accept(dialog):
    for edit in dialog.findChildren(QtWidgets.QLineEdit):
        edit.setText(str(output))
    return QtWidgets.QDialog.Accepted
window._exec_dialog = accept
errors = []
window.show_error = lambda *args, **kwargs: errors.append(args)
updates = []
original = window.on_task_progress
def progress(*args):
    assert window.is_busy and window.progress_bar.isVisible()
    assert not window._central_splitter.isEnabled()
    assert window._central_splitter.graphicsEffect().opacity() == .45
    updates.append(args)
    original(*args)
window.on_task_progress = progress
assert window.save_dialog() and output.is_file() and updates and not errors
assert not window.is_busy and not window.progress_bar.isVisible()
assert window._central_splitter.isEnabled() and window._central_splitter.graphicsEffect() is None
window.session.replace_fitted_maps({'static': np.ones((7, 11)), 'dynamic': np.ones((3, 7, 11))})
window.refresh_map_list()
window.view_combo.setCurrentText('Map')
for name, expected in [('static', ['auto']), ('dynamic', ['auto', 't']), ('static', ['auto'])]:
    window.on_map_changed(name)
    choices = [window.plot_x_combo.itemData(i) for i in range(window.plot_x_combo.count())]
    assert choices == expected, (name, choices)
assert not errors
window.clear_dirty()
window.close()
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", JAX_PLATFORMS="cpu",
               MRAX_GUI_CONFIG=str(tmp_path / "gui_config.json"), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "study.mrax")],
                            env=env, capture_output=True, text=True, timeout=45)
    if result.returncode == 77:
        pytest.skip("Native Qt dependencies are unavailable")
    assert result.returncode == 0, result.stderr
