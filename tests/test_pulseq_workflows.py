"""Synthetic pulseq contracts and numerical references."""

from dataclasses import replace

import h5py
import ismrmrd
import nibabel as nib
import numpy as np
import pytest
from _synthetic import (
    _header,
)
from ismrmrd import xsd

from mrax.core import run_fit
from mrax.io import (
    read_measurement,
    read_pulseq_measurement,
)
from mrax.io.measurement import build_measurement_experiment
from mrax.io.pulseq_import import PulseqSequenceInfo, _metadata_from_pulseq
from mrax.reco import reconstruct_acquisition


@pytest.fixture
def quantitative_recording(tmp_path, monkeypatch):
    nx, ny = 8, 5
    times = np.asarray([11.0, 63.0, 137.0])
    yy, xx = np.mgrid[:ny, :nx]
    truth = (1.0 + xx / nx + yy / ny)[None] * np.exp(-times[:, None, None] / 65.0)
    kspace = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(truth, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )
    definitions = {
        "Name": "synthetic_tse",
        "TE": 0.011,
        "TR": 0.8,
        "echo_train_length": 3,
        "encoding_dim": [nx, ny, 1],
        "FOV": [0.07, 0.09, 0.004],
        "channel_order": ["y", "x", "z"],
    }
    sequence = tmp_path / "sequence.seq"
    sequence.write_text("[DEFINITIONS]\nName synthetic_tse\n")
    path = tmp_path / "raw.mrd"
    dataset = ismrmrd.Dataset(str(path), "dataset")
    header = _header(nx, ny)
    header.sequenceParameters = None
    for space in (header.encoding[0].encodedSpace, header.encoding[0].reconSpace):
        space.fieldOfView_mm = xsd.fieldOfViewMm(x=90.0, y=70.0, z=4.0)
    dataset.write_xml_header(xsd.ToXML(header))
    trajectories, echo_times = [], []
    for line in (2, 0, 4, 1, 3):
        for contrast in (1, 2, 0):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(nx, 1, 3)
            acquisition.idx.kspace_encode_step_1 = line
            acquisition.idx.contrast = contrast
            acquisition.center_sample = nx // 2
            acquisition.read_dir[:] = [1, 0, 0]
            acquisition.phase_dir[:] = [0, 1, 0]
            acquisition.slice_dir[:] = [0, 0, 1]
            acquisition.data[:] = kspace[contrast, line]
            physical = np.column_stack(
                [
                    np.full(nx, (line - ny // 2) / 0.07),
                    (np.arange(nx) - nx // 2) / 0.09,
                    np.zeros(nx),
                ]
            )
            acquisition.traj[:] = physical
            trajectories.append(physical[:, :2])
            echo_times.append(times[contrast])
            dataset.append_acquisition(acquisition)
    dataset.close()
    info = PulseqSequenceInfo(
        sequence,
        definitions=definitions,
        trajectory_physical=np.concatenate(trajectories),
        adc_counts=(nx,) * (ny * 3),
        adc_echo_times_ms=np.asarray(echo_times),
        spatial_shape=(ny, nx),
        normalized_metadata=_metadata_from_pulseq(definitions, 1, sequence),
    )
    monkeypatch.setattr("mrax.io.pulseq_import.read_pulseq", lambda *args, **kwargs: info)
    monkeypatch.setattr("mrax.gui.pulseq_import.read_pulseq", lambda *args, **kwargs: info)
    order = [2, 0, 1]
    image = tmp_path / "image_mag.nii"
    nib.save(
        nib.Nifti1Image(
            truth[order].transpose(2, 1, 0), np.diag([0.09 / nx, 0.07 / ny, 0.004, 1.0])
        ),
        image,
    )
    return path, image, sequence, info, truth, kspace, times


def test_physical_cartesian_coordinates_echo_timing_and_geometry(quantitative_recording):
    raw, _, sequence, _, truth, expected_kspace, times = quantitative_recording
    measured = read_pulseq_measurement(raw, sequence_path=sequence)
    assert measured.acquisition.kind == "cartesian"
    assert measured.metadata["cartesian_trajectory_validated"]
    assert measured.metadata["trajectory_units"] == "cycles/m"
    np.testing.assert_allclose(measured.acquisition.samples[:, 0], expected_kspace, atol=1e-6)
    geometry = measured.spatial_geometry
    assert geometry.status == "partial"
    np.testing.assert_allclose(np.asarray(geometry.affine)[:3, 0], [0.0, -11.25, 0.0])
    image = abs(reconstruct_acquisition(measured.acquisition).image)
    np.testing.assert_allclose(image, truth, atol=1e-6)
    experiment, dataset, model, _ = build_measurement_experiment(measured, image_data=image)
    assert experiment.dims == {"te": 3, "x": 5, "y": 8}
    assert model.name == "t2_se"
    np.testing.assert_allclose(experiment.observables["TE"].values, times)
    experiment.validate(dataset)


def test_fractional_sequence_coordinates_remain_sampled(quantitative_recording, monkeypatch):
    raw, _, sequence, info, *_ = quantitative_recording
    physical = info.trajectory_physical.copy()
    physical[:, 1] += 0.25 / 0.09
    updated = replace(info, trajectory_physical=physical)
    monkeypatch.setattr("mrax.io.pulseq_import.read_pulseq", lambda *args, **kwargs: updated)
    with h5py.File(raw, "r+") as file:
        records = file["dataset/data"][:]
        for row in records:
            row["traj"].reshape(-1, 3)[:, 1] += 0.25 / 0.09
        file["dataset/data"][:] = records
    measured = read_pulseq_measurement(
        raw, sequence_path=sequence, reader_options={"density": None}
    )
    assert measured.acquisition.kind == "sampled"
    assert not measured.metadata["cartesian_trajectory_validated"]
    np.testing.assert_allclose(
        measured.acquisition.trajectory[:8, 1], (np.arange(8) - 4 + 0.25) / 8, atol=1e-6
    )


def test_nifti_explicit_layout_validates_order_and_preserves_standard_volumes(tmp_path):
    path = tmp_path / "volume.nii"
    array = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2))
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    assert read_measurement(path).image_data.shape == (1, 2, 3, 4)
    explicit = read_measurement(path, spatial_ndim=2, frame_order=[1, 0])
    np.testing.assert_array_equal(explicit.image_data, array.transpose(2, 1, 0)[::-1])
    for order in ([0, 0], [0, 1, 2], [0.5, 1.5]):
        with pytest.raises(ValueError, match="permutation"):
            read_measurement(path, spatial_ndim=2, frame_order=order)
    with pytest.raises(ValueError, match="spatial_ndim"):
        read_measurement(path, spatial_ndim=1)


def test_slice_volumes_contrast_order_geometry_and_inverse_encoding(tmp_path):
    from mrax.reco.layout import image_to_encoding

    path = tmp_path / "volume.h5"
    header = _header(4, 3)
    header.sequenceParameters.TE = [10.0, 30.0]
    file = ismrmrd.Dataset(str(path), "dataset")
    file.write_xml_header(xsd.ToXML(header))
    truth = np.zeros((2, 2, 3, 3, 4))  # echo, average, slice, row, column
    for average in range(2):
        for contrast, te in enumerate([10.0, 30.0]):
            for label, physical_slice in enumerate([0, 2, 1]):
                image = np.full(
                    (3, 4), (1 + average * 0.1) * (1 + physical_slice * 0.2) * np.exp(-te / 70.0)
                )
                truth[contrast, average, physical_slice] = image
                raw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm="ortho"))
                for line in [2, 0, 1]:
                    record = ismrmrd.Acquisition()
                    record.resize(4, 1)
                    record.idx.average = average
                    record.idx.contrast = contrast
                    record.idx.slice = label
                    record.idx.kspace_encode_step_1 = line
                    record.center_sample = 2
                    record.position[:] = [0, 0, physical_slice * 4]
                    record.read_dir[:] = [1, 0, 0]
                    record.phase_dir[:] = [0, 1, 0]
                    record.slice_dir[:] = [0, 0, 1]
                    record.data[0] = raw[line]
                    file.append_acquisition(record)
    file.close()
    measured = read_pulseq_measurement(path)
    assert measured.acquisition.recipe.stack_slices == 3
    image = reconstruct_acquisition(measured.acquisition).image
    experiment, dataset, _model, _metadata = build_measurement_experiment(
        measured, image_data=np.abs(image), override_type="mse"
    )
    experiment.validate(dataset)
    np.testing.assert_allclose(dataset.data, truth, rtol=1e-6)
    np.testing.assert_allclose(dataset.spatial_geometry.voxel_sizes, [4, 20, 20])
    np.testing.assert_allclose(np.asarray(dataset.spatial_geometry.affine)[:3, 3], [30, 20, 0])
    predicted = image_to_encoding(dataset.data.reshape((4, 3, 3, 4)), dataset.acquisition)
    original = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(measured.acquisition.samples[:, 0], axes=(-2, -1)), norm="ortho"
        ),
        axes=(-2, -1),
    )
    np.testing.assert_allclose(predicted, original, rtol=1e-6, atol=1e-7)


def test_nifti_declared_time_and_space_units(tmp_path):
    image = nib.Nifti1Image(np.ones((4, 5, 2, 3), np.float32), np.diag([0.001, 0.002, 0.003, 1.0]))
    image.header.set_xyzt_units("meter", "msec")
    image.header.set_zooms((0.001, 0.002, 0.003, 250.0))
    path = tmp_path / "time.nii"
    nib.save(image, path)
    measurement = read_pulseq_measurement(path)
    np.testing.assert_allclose(measurement.metadata["frame_coordinates"]["time"], [0, 0.25, 0.5])
    np.testing.assert_allclose(measurement.spatial_geometry.voxel_sizes, [3, 2, 1])
    exp, data, _model, _meta = build_measurement_experiment(measurement)
    np.testing.assert_allclose(exp.observables["time"].values.ravel(), [0, 0.25, 0.5])
    assert exp.observables["time"].si_unit == "s"


def test_quantitative_mrd_fft_variational_and_model_fit_agree(tmp_path):
    """Known relaxation signals test units and the complete public fitting path."""
    nx, ny = 4, 3
    te = np.asarray([8.0, 24.0, 48.0, 96.0, 160.0])
    curve = 1.2 * np.exp(-te / 70.0) + 0.03
    images = np.broadcast_to(curve[:, None, None], (len(te), ny, nx))
    kspace = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(images, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )
    header = _header(nx, ny)
    header.sequenceParameters.TE = te.tolist()
    path = tmp_path / "relaxation.h5"
    file = ismrmrd.Dataset(str(path), "dataset")
    file.write_xml_header(xsd.ToXML(header))
    for contrast in range(len(te)):
        for line in range(ny):
            record = ismrmrd.Acquisition()
            record.resize(nx, 1)
            record.idx.contrast = contrast
            record.idx.kspace_encode_step_1 = line
            record.center_sample = nx // 2
            record.data[0] = kspace[contrast, line]
            file.append_acquisition(record)
    file.close()
    measurement = read_pulseq_measurement(path)
    acquisition = measurement.acquisition
    fft = reconstruct_acquisition(acquisition)
    np.testing.assert_allclose(fft.image, images, rtol=2e-6)
    variational = reconstruct_acquisition(acquisition, method="variational", num_steps=8)
    np.testing.assert_allclose(variational.image, fft.image, atol=2e-5)
    experiment, dataset, model, _ = build_measurement_experiment(
        measurement, image_data=np.abs(fft.image), override_type="mse"
    )
    fields, _ = run_fit(experiment, model, dataset, num_steps=2000, regularizer_weight=0.0)
    direct = reconstruct_acquisition(
        acquisition,
        method="model_based",
        experiment=experiment,
        model=model,
        num_steps=2000,
        regularizer_weight=0.0,
    )
    for key in fields:
        np.testing.assert_allclose(
            fields[key].values, direct.parameters[key].values, rtol=3e-3, atol=2e-4
        )
    np.testing.assert_allclose(fields["t2"].values, 70.0, rtol=0.04)
    from mrax.data_struct import Study, load_study_results
    from mrax.gui import GuiSession

    study = Study([experiment], [dataset], [model], names=["relaxation"])
    bundle = study.save(tmp_path / "study.mrax")
    restored = load_study_results(bundle)
    assert (
        restored["experiments"]["relaxation"]["model_parameter_specs"]["amplitude"]["scale"] < 2.0
    )
    session = GuiSession.from_manifest(bundle)
    _experiment, restored_model, _dataset, _study = session.build_fit_inputs()
    params, unpack = model.parametrization_factory((ny, nx))
    restored_params, restored_unpack = restored_model.parametrization_factory((ny, nx))
    for key, value in unpack(params).items():
        np.testing.assert_array_equal(restored_unpack(restored_params)[key], value)
