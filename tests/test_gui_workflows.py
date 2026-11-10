"""Synthetic gui contracts and numerical references."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import nibabel as nib
import numpy as np
import pytest

from mrax import __version__
from mrax.analysis import ROI
from mrax.core.loss_builder import make_model_callable, run_fit
from mrax.data_struct import DataSet, Quantity, Study, save_study_results
from mrax.gui import GuiSession, import_mask, load_roi_set, save_roi_set
from mrax.gui.pulseq_import import plan_pulseq_import
from mrax.gui.services.export import _export_object_records_for_session, _write_single_export_object
from mrax.gui.services.importing import (
    _apply_observable_dim_compatibility,
    _detect_import_vendor,
    _load_experiment_metadata_template_from_file,
)
from mrax.gui.services.state import _restore_gui_state, _serialize_gui_state
from mrax.io.nifti import load_nifti
from mrax.library import build_default_experiment, make_t2_se_model
from mrax.versioning import versioned_payload


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


def _sequence(path):
    path.write_text('[DEFINITIONS]\nName inversion_recovery\nTI 0.03 0.1 0.4\nFREQ 120\n')
    return path


def _images(path):
    curve = np.abs(1000 * (1 - 2 * np.exp(-np.array([30., 100., 400.]) / 180.)))
    data = np.broadcast_to(curve, (4, 3, 2, 3)).copy()
    nib.save(nib.Nifti1Image(data, np.diag([2., 3., 4., 1.])), path)


@pytest.mark.parametrize('choose_file', [False, True])
def test_pulseq_neighbours_import_to_gui_and_fit(tmp_path, choose_file):
    sequence = _sequence(tmp_path / 'relaxation.seq')
    _images(tmp_path / 'images.nii.gz')
    source = sequence if choose_file else tmp_path
    assert _detect_import_vendor(source) == 'Pulseq'
    plan = plan_pulseq_import(source)
    assert len(plan.entries) == 1
    assert plan.summary[0]['frame_count'] == 3
    assert plan.summary[0]['detected_type'] == 'ir'
    session = GuiSession()
    session.import_pulseq_study(plan)
    assert session.source_kind == 'pulseq'
    assert session.image_array.shape == (3, 2, 3, 4)
    assert session.data['manifest']['source']['kind'] == 'pulseq'
    experiment, model, dataset = next(iter(session.study.entries_by_name().values()))
    experiment.validate(dataset)
    np.testing.assert_array_equal(experiment.observables['TI'].values, [30., 100., 400.])
    np.testing.assert_allclose(dataset.spatial_geometry.voxel_sizes, [4., 3., 2.])
    fields, _ = run_fit(experiment, model, dataset, num_steps=1500)
    np.testing.assert_allclose(fields['t1'].values, 180., atol=.1)
    prediction = make_model_callable(experiment, model)(fields)
    np.testing.assert_allclose(prediction, dataset.data, rtol=1e-4, atol=.02)
    session.save_imported_study(tmp_path / 'result.mrax')
    reopened = GuiSession.from_manifest(tmp_path / 'result.mrax')
    np.testing.assert_array_equal(reopened.image_array, session.image_array)


def test_pulseq_external_file_selection_uses_portable_experiment_names(tmp_path):
    sequences = tmp_path / 'sequences'
    sequences.mkdir()
    sequence = _sequence(sequences / 'relaxation.seq')
    images = tmp_path / 'images.nii.gz'
    _images(images)
    plan = plan_pulseq_import(sequence, images)
    session = GuiSession()
    session.import_pulseq_study(plan)
    assert all(not Path(name).is_absolute() and ':' not in name for name in session.experiment_names)
    np.testing.assert_array_equal(session.study.experiments[0].observables['TI'].values, [30., 100., 400.])


def test_pulseq_multiple_sequences_require_unambiguous_pairs(tmp_path):
    first = _sequence(tmp_path / 'first.seq')
    _sequence(tmp_path / 'second.seq')
    _images(tmp_path / 'unpaired.nii')
    plan = plan_pulseq_import(tmp_path)
    assert not plan.entries and plan.issues
    _images(tmp_path / 'first.nii.gz')
    plan = plan_pulseq_import(tmp_path)
    assert len(plan.entries) == 1
    assert plan.entries[0]['sequence'] == first.resolve()


def test_pulseq_dicom_preparation_import_preserves_geometry_and_scanner_frequency(tmp_path):
    from _synthetic import _write_classic_dicom_slice
    from pydicom import dcmread
    from pydicom.uid import generate_uid

    (tmp_path / 'preparation.seq').write_text('[DEFINITIONS]\nName cest\noffsets_ppm -1 0 1\nTsat 0.004\nB1pa 2.4\nFREQ 110\n')
    uid = generate_uid()
    for frame in range(3):
        for plane in range(2):
            path = tmp_path / f'image_{frame}_{plane}.IMA'
            _write_classic_dicom_slice(path, series_uid=uid,
                echo_index=frame, slice_index=plane, description='Synthetic preparation')
            header = dcmread(path)
            header.MagneticFieldStrength = 3.
            header.save_as(path, enforce_file_format=True)
    plan = plan_pulseq_import(tmp_path)
    assert len(plan.entries) == 1
    assert plan.summary[0]['frame_count'] == 3
    assert plan.summary[0]['field_strength_mhz'] == pytest.approx(3. * 42.57747892)
    session = GuiSession()
    session.import_pulseq_study(plan)
    experiment, _model, dataset = next(iter(session.study.entries_by_name().values()))
    experiment.validate(dataset)
    assert dataset.spatial_geometry.status == 'known'
    np.testing.assert_array_equal(experiment.observables['freq_offsets'].values, [-1., 0., 1.])
    assert float(experiment.observables['field_strength_mhz'].values) == pytest.approx(3. * 42.57747892)


def test_pulseq_raw_only_gui_import_uses_generic_reconstruction_and_retains_raw(tmp_path):
    from _synthetic import _write_mrd

    (tmp_path / 'sequence.seq').write_text('[DEFINITIONS]\nName multi_spin_echo\nEchoTimes 0.012 0.036\n')
    _write_mrd(tmp_path / 'raw.mrd')
    session = GuiSession()
    session.import_pulseq_study(tmp_path)
    assert len(session.experiment_names) == 1
    assert session.raw_data_loaded()
    assert session.can_reconstruct_raw_data()
    assert session.study.datasets[0].acquisition is not None
    np.testing.assert_allclose(session.study.experiments[0].observables['TE'].values, [12., 36.])
    assert float(session.study.study_globals['field_strength_mhz'].values) == 128.
    previous = session.study
    with pytest.raises(ValueError, match='No selected'):
        session.import_pulseq_study(tmp_path, options={'load_raw_kspace': False})
    assert session.study is previous


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


def test_gui_bruker_detection_uses_file_based_experiment_discovery(tmp_path):
    from _synthetic import _write_minimal_bruker_scan

    from mrax.gui.services.importing import _detect_bruker_import_summary, _looks_like_bruker_study

    _write_minimal_bruker_scan(tmp_path)
    exported = tmp_path / "exported_scan"
    (tmp_path / "1").rename(exported)
    for root in [tmp_path, exported]:
        assert _looks_like_bruker_study(root)
        assert [row["scan"] for row in _detect_bruker_import_summary(root)] == ["exported_scan"]
        session = GuiSession()
        session.import_bruker_study(root, field_strength_mhz=100, load_raw_kspace=False)
        assert list(session.data["experiments"]) == ["exported_scan"]


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
            {"axial": ("z", "y"), "coronal": ("x", "y"), "sagittal": ("x", "z")},
        ),
        (
            ["y", "z", "x"],
            "sagittal",
            {"axial": ("x", "z"), "coronal": ("y", "z"), "sagittal": ("y", "x")},
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
    assert session.image_plane_dims == ("x", "y")
    assert session.image_plane_voxel_sizes == pytest.approx((0.2, 0.3))
    session.set_image_plane("axial")
    assert session.image_plane_dims == ("z", "y")
    assert session.image_plane_voxel_sizes == pytest.approx((1.1, 0.3))
    session.set_image_plane("sagittal")
    assert session.image_plane_dims == ("x", "z")
    assert session.image_plane_voxel_sizes == pytest.approx((0.2, 1.1))


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


def test_gui_roi_nifti_export_keeps_identity_fallback_without_reference_geometry():
    session = GuiSession()
    session.create_empty_study("identity_export")
    session.add_experiment("reference_scan", np.zeros((2, 2)), dims={"x": 2, "y": 2})
    session.add_roi(ROI("mask", np.ones((2, 2), dtype=bool), dims=("x", "y")))

    roi_record = next(record for record in _export_object_records_for_session(session) if record["kind"] == "roi")

    np.testing.assert_array_equal(roi_record["nifti_reference"]["affine"], np.eye(4))
    assert roi_record["nifti_reference"]["identity_fallback"] is True


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


def test_gui_state_accepts_only_the_current_spatial_schema():
    from mrax.gui import GuiSession
    from mrax.gui.services.state import _restore_gui_state

    payload = versioned_payload({'format': 'mrax.gui.state', 'version': 2})
    with pytest.raises(ValueError, match='not supported'):
        _restore_gui_state(GuiSession(), payload, {})
