"""Synthetic pulseq contracts and numerical references."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import ismrmrd
import nibabel as nib
import numpy as np
import optax
import pytest
from _synthetic import (
    _header,
    _write_mrd,
)
from ismrmrd import xsd

from mrax.core import run_fit
from mrax.gui import GuiSession
from mrax.gui.pulseq_import import plan_pulseq_import
from mrax.io import (
    MeasurementData,
    import_pulseq_experiment,
    import_pulseq_study,
    read_measurement,
    read_pulseq_measurement,
    register_measurement_reader,
)
from mrax.io.measurement import build_measurement_experiment
from mrax.io.pulseq_import import PulseqSequenceInfo, _metadata_from_pulseq, _normalize_trajectory
from mrax.io.pulseq_measurement import discover_pulseq_measurements
from mrax.io.pulseq_samples import prepare_pulseq_samples
from mrax.reco import reconstruct_acquisition


@pytest.fixture
def quantitative_recording(tmp_path, monkeypatch):
    nx, ny = 8, 5
    times = np.asarray([11., 63., 137.])
    yy, xx = np.mgrid[:ny, :nx]
    truth = (1. + xx / nx + yy / ny)[None] * np.exp(-times[:, None, None] / 65.)
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(truth, axes=(-2, -1)),
        axes=(-2, -1), norm='ortho'), axes=(-2, -1))
    definitions = {'Name': 'synthetic_tse', 'TE': .011, 'TR': .8, 'echo_train_length': 3,
        'encoding_dim': [nx, ny, 1], 'FOV': [.07, .09, .004], 'channel_order': ['y', 'x', 'z']}
    sequence = tmp_path / 'sequence.seq'
    sequence.write_text('[DEFINITIONS]\nName synthetic_tse\n')
    path = tmp_path / 'raw.mrd'
    dataset = ismrmrd.Dataset(str(path), 'dataset')
    header = _header(nx, ny)
    header.sequenceParameters = None
    for space in (header.encoding[0].encodedSpace, header.encoding[0].reconSpace):
        space.fieldOfView_mm = xsd.fieldOfViewMm(x=90., y=70., z=4.)
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
            physical = np.column_stack([np.full(nx, (line - ny // 2) / .07), (np.arange(nx) - nx // 2) / .09, np.zeros(nx)])
            acquisition.traj[:] = physical
            trajectories.append(physical[:, :2])
            echo_times.append(times[contrast])
            dataset.append_acquisition(acquisition)
    dataset.close()
    info = PulseqSequenceInfo(sequence, definitions=definitions, trajectory_physical=np.concatenate(trajectories),
        adc_counts=(nx,) * (ny * 3), adc_echo_times_ms=np.asarray(echo_times), spatial_shape=(ny, nx),
        normalized_metadata=_metadata_from_pulseq(definitions, 1, sequence))
    monkeypatch.setattr('mrax.io.pulseq_import.read_pulseq', lambda *args, **kwargs: info)
    monkeypatch.setattr('mrax.gui.pulseq_import.read_pulseq', lambda *args, **kwargs: info)
    order = [2, 0, 1]
    image = tmp_path / 'image_mag.nii'
    nib.save(nib.Nifti1Image(truth[order].transpose(2, 1, 0), np.diag([.09 / nx, .07 / ny, .004, 1.])), image)
    return path, image, sequence, info, truth, kspace, times






def test_physical_cartesian_coordinates_echo_timing_and_geometry(quantitative_recording):
    raw, _, sequence, _, truth, expected_kspace, times = quantitative_recording
    measured = read_pulseq_measurement(raw, sequence_path=sequence)
    assert measured.acquisition.kind == 'cartesian'
    assert measured.metadata['cartesian_trajectory_validated']
    assert measured.metadata['trajectory_units'] == 'cycles/m'
    np.testing.assert_allclose(measured.acquisition.samples[:, 0], expected_kspace, atol=1e-6)
    geometry = measured.spatial_geometry
    assert geometry.status == 'partial'
    np.testing.assert_allclose(np.asarray(geometry.affine)[:3, 0], [0., -11.25, 0.])
    image = abs(reconstruct_acquisition(measured.acquisition).image)
    np.testing.assert_allclose(image, truth, atol=1e-6)
    experiment, dataset, model, _ = build_measurement_experiment(measured, image_data=image)
    assert experiment.dims == {'te': 3, 'x': 5, 'y': 8}
    assert model.name == 't2_se'
    np.testing.assert_allclose(experiment.observables['TE'].values, times)
    experiment.validate(dataset)


@pytest.mark.parametrize('selection', [None, [2, 0]])
def test_compact_nifti_uses_raw_energy_order_and_explicit_units(quantitative_recording, selection):
    _, image, sequence, _, truth, _, times = quantitative_recording
    with pytest.warns(RuntimeWarning, match='metres'):
        measured = read_pulseq_measurement(image, sequence_path=sequence, reader_options={'frame_selection': selection})
    indices = np.arange(3) if selection is None else np.asarray(selection)
    np.testing.assert_allclose(measured.image_data, truth[indices])
    np.testing.assert_allclose(measured.normalized_metadata.echo_times, times[indices])
    np.testing.assert_allclose(measured.spatial_geometry.voxel_sizes, [14., 11.25])
    experiment, dataset, model, _ = build_measurement_experiment(measured)
    assert tuple(experiment.non_spatial_dim_names) == ('te',)
    assert model.name == 't2_se'
    np.testing.assert_allclose(dataset.data, truth[np.sort(indices)])


def test_compact_nifti_rejects_ambiguous_order_and_accepts_override(quantitative_recording):
    _, image, sequence, _, truth, _, _ = quantitative_recording
    pixels = np.ones((8, 5, 3), np.float32)
    nib.save(nib.Nifti1Image(pixels, np.eye(4)), image)
    with pytest.raises(ValueError, match='ambiguous'):
        read_pulseq_measurement(image, sequence_path=sequence)
    explicit = read_pulseq_measurement(image, sequence_path=sequence,
        reader_options={'spatial_ndim': 2, 'frame_order': [1, 2, 0], 'spatial_unit': 'mm'})
    assert explicit.image_data.shape == truth.shape


def test_fractional_sequence_coordinates_remain_sampled(quantitative_recording, monkeypatch):
    raw, _, sequence, info, *_ = quantitative_recording
    physical = info.trajectory_physical.copy()
    physical[:, 1] += .25 / .09
    updated = replace(info, trajectory_physical=physical)
    monkeypatch.setattr('mrax.io.pulseq_import.read_pulseq', lambda *args, **kwargs: updated)
    with h5py.File(raw, 'r+') as file:
        records = file['dataset/data'][:]
        for row in records:
            row['traj'].reshape(-1, 3)[:, 1] += .25 / .09
        file['dataset/data'][:] = records
    measured = read_pulseq_measurement(raw, sequence_path=sequence, reader_options={'density': None})
    assert measured.acquisition.kind == 'sampled'
    assert not measured.metadata['cartesian_trajectory_validated']
    np.testing.assert_allclose(measured.acquisition.trajectory[:8, 1], (np.arange(8) - 4 + .25) / 8, atol=1e-6)


def test_explicit_sampled_mode_retains_verified_physical_units(quantitative_recording):
    raw, _, sequence, *_ = quantitative_recording
    measured = read_pulseq_measurement(raw, sequence_path=sequence,
        reader_options={'trajectory_mode': 'sampled', 'density': None})
    assert measured.acquisition.kind == 'sampled'
    assert measured.metadata['trajectory_units'] == 'cycles/m'
    assert measured.metadata['cartesian_trajectory_validated']


def test_inconsistent_trajectory_never_silently_uses_cartesian_counters(quantitative_recording):
    raw, _, sequence, *_ = quantitative_recording
    with h5py.File(raw, 'r+') as file:
        records = file['dataset/data'][:]
        records[0]['traj'][0] += 10.
        file['dataset/data'][:] = records
    with pytest.warns(RuntimeWarning, match='units inferred'), pytest.raises(ValueError, match='cycles/pixel bounds'):
        read_pulseq_measurement(raw, sequence_path=sequence)


def test_compact_nifti_without_acquisition_evidence_requires_explicit_axes(quantitative_recording):
    raw, image, sequence, *_ = quantitative_recording
    raw.unlink()
    with pytest.raises(ValueError, match='third axis conflicts'):
        read_pulseq_measurement(image, sequence_path=sequence)


def test_phase_images_use_magnitude_order_but_do_not_choose_t2(quantitative_recording):
    _, image, sequence, _, truth, *_ = quantitative_recording
    phase = image.with_name('image_phase.nii')
    source = nib.load(image)
    pixels = np.broadcast_to(np.asarray([.3, .1, .2]), source.shape)
    nib.save(nib.Nifti1Image(pixels, source.affine), phase)
    with pytest.warns(RuntimeWarning, match='metres'):
        measured = read_pulseq_measurement(phase, sequence_path=sequence)
    assert measured.metadata['image_component'] == 'phase'
    assert measured.image_data.shape == truth.shape
    np.testing.assert_allclose(measured.image_data[:, 0, 0], [.1, .2, .3])
    _, _, model, _ = build_measurement_experiment(measured)
    assert model.name == 'identity'


def test_auxiliary_hdf5_frequency_and_gui_import(quantitative_recording):
    raw, image, sequence, info, _, _, times = quantitative_recording
    with h5py.File(raw.parent / 'auxiliary.h5', 'w') as file:
        file.create_dataset('receive_data/processed', data=np.ones(7))
    definitions = {**info.definitions, 'channel_order': 'y x z'}
    (raw.parent / 'console.json').write_text(json.dumps({'sequence': {'definitions': definitions},
        'acquisition_parameter': {'larmor_frequency': 127700000.}}))
    assert set(discover_pulseq_measurements(raw.parent)) == {raw, image}
    plan = plan_pulseq_import(raw.parent)
    assert not plan.issues
    assert all(entry['frame_count'] == 3 and entry['detected_type'] == 'mse' for entry in plan.summary)
    with pytest.warns(RuntimeWarning):
        session = GuiSession()
        session.import_pulseq_study(plan)
    assert not session.import_metadata['import_issues']
    assert float(session.study.study_globals['field_strength_mhz'].values) == pytest.approx(127.7)
    for experiment, model, dataset in session.study.entries_by_name().values():
        np.testing.assert_allclose(experiment.observables['TE'].values, times)
        experiment.validate(dataset)
        assert model.name == 't2_se'


def test_imported_quantitative_model_fits_synthetic_truth(quantitative_recording):
    _, image, sequence, *_ = quantitative_recording
    with pytest.warns(RuntimeWarning, match='metres'):
        measured = read_pulseq_measurement(image, sequence_path=sequence)
    experiment, dataset, model, _ = build_measurement_experiment(measured)
    fields, _ = run_fit(experiment, model, dataset, num_steps=4000,
        step_size=optax.exponential_decay(.03, 1000, .4))
    np.testing.assert_allclose(fields['t2'].values, 65., atol=.03)


def test_nifti_explicit_layout_validates_order_and_preserves_standard_volumes(tmp_path):
    path = tmp_path / 'volume.nii'
    array = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2))
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    assert read_measurement(path).image_data.shape == (1, 2, 3, 4)
    explicit = read_measurement(path, spatial_ndim=2, frame_order=[1, 0])
    np.testing.assert_array_equal(explicit.image_data, array.transpose(2, 1, 0)[::-1])
    for order in ([0, 0], [0, 1, 2], [.5, 1.5]):
        with pytest.raises(ValueError, match='permutation'):
            read_measurement(path, spatial_ndim=2, frame_order=order)
    with pytest.raises(ValueError, match='spatial_ndim'):
        read_measurement(path, spatial_ndim=1)


def test_executed_echo_times_use_readout_centres_and_preceding_excitation(tmp_path, monkeypatch):
    import pypulseq

    from mrax.io.pulseq_import import read_pulseq

    sequence_path = tmp_path / 'timing.seq'
    sequence_path.write_text('[DEFINITIONS]\nFOV .04 .04\nMatrix 3 2\n\n'
        '[ADC]\n1 3\n\n[BLOCKS]\n1 0 0 0 0 0 1\n2 0 0 0 0 0 1\n')
    adc_times = np.asarray([.0109, .011, .0111, 2.0629, 2.063, 2.0631])
    trajectory = np.vstack([np.tile([-1., 0., 1.], 2), np.repeat([0., 1.], 3), np.zeros(6)])
    sequence = SimpleNamespace(definitions={},
        calculate_kspace=lambda: (trajectory, None, [0., 2.], [], adc_times))
    monkeypatch.setattr(pypulseq, 'Sequence', lambda: sequence)
    monkeypatch.setattr('mrax.io.pulseq_compat.read_sequence', lambda *args: None)
    info = read_pulseq(sequence_path)
    np.testing.assert_allclose(info.adc_echo_times_ms, [11., 63.], atol=1e-9)


@pytest.mark.parametrize('selection', [[4, 1], [4, 3, 2, 1, 0]])
def test_nifti_selected_pixels_and_timing_are_selected_once(tmp_path, selection):
    sequence = tmp_path / 'recovery.seq'
    sequence.write_text('[DEFINITIONS]\nName inversion_recovery\nTI .01 .03 .09 .27 .81\nFREQ 120\n')
    pixels = np.broadcast_to(np.arange(5, dtype=np.float32), (3, 4, 2, 5)).copy()
    image = nib.Nifti1Image(pixels, np.eye(4))
    image.header.set_xyzt_units('mm', 'sec')
    image.header.set_zooms((1., 1., 1., .5))
    path = tmp_path / 'recovery.nii.gz'
    nib.save(image, path)
    measured = read_pulseq_measurement(path, reader_options={'frame_selection': selection})
    np.testing.assert_array_equal(measured.image_data[:, 0, 0, 0], selection)
    np.testing.assert_array_equal(measured.metadata['frame_coordinates']['time'], np.asarray(selection) * .5)
    np.testing.assert_allclose(measured.normalized_metadata.inversion_times, np.asarray([10., 30., 90., 270., 810.])[selection])
    experiment, dataset, model, _ = build_measurement_experiment(measured)
    assert model.name == 't1_ir'
    experiment.validate(dataset)
    np.testing.assert_array_equal(np.asarray(dataset.data)[:, 0, 0, 0], np.sort(selection))


@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('selection', [[1], [1, 0]])
def test_mrd_selected_frames_keep_declared_sequence_timing(tmp_path, raw, selection):
    from _synthetic import _write_mrd

    sequence = tmp_path / 'relaxation.seq'
    sequence.write_text('[DEFINITIONS]\nName multi_spin_echo\nEchoTimes .009 .041\n')
    path = tmp_path / 'relaxation.mrd'
    if raw:
        _write_mrd(path)
    else:
        file = ismrmrd.Dataset(str(path), 'dataset')
        file.write_xml_header(xsd.ToXML(_header(4, 3)))
        for i in range(2):
            image = ismrmrd.Image.from_array(np.full((3, 4), i + 1, np.float32), transpose=False)
            image.contrast = i
            image.field_of_view[:] = [40., 30., 4.]
            image.read_dir[:] = [1., 0., 0.]
            image.phase_dir[:] = [0., 1., 0.]
            image.slice_dir[:] = [0., 0., 1.]
            file.append_image('images', image)
        file.close()
    measured = read_pulseq_measurement(path, sequence_path=sequence,
        reader_options={'frame_selection': selection})
    np.testing.assert_allclose(measured.normalized_metadata.echo_times, np.asarray([9., 41.])[selection])
    if not raw:
        np.testing.assert_array_equal(measured.image_data[:, 0, 0], np.asarray(selection) + 1)


def test_parameter_map_does_not_inherit_quantitative_sweep_axes(tmp_path):
    sequence = tmp_path / 'sequence.seq'
    sequence.write_text('[DEFINITIONS]\nName multi_spin_echo\nEchoTimes .01 .03 .06 .12\nFREQ 120\n')
    path = tmp_path / 'relaxation_map.nii'
    nib.save(nib.Nifti1Image(np.ones((3, 4)), np.eye(4)), path)
    measured = read_pulseq_measurement(path, sequence_path=sequence)
    experiment, dataset, model, _ = build_measurement_experiment(measured)
    assert measured.normalized_metadata.echo_times is None
    assert model.name == 'identity'
    assert experiment.frame_count == 1
    experiment.validate(dataset)


@pytest.mark.parametrize('mode', ['auto', 'sampled'])
def test_missing_trajectory_axis_requires_sequence_and_counter_agreement(quantitative_recording, mode):
    raw, _, sequence, _, truth, *_ = quantitative_recording
    with h5py.File(raw, 'r+') as file:
        records = file['dataset/data'][:]
        for row in records:
            row['traj'].reshape(-1, 3)[:, 0] = 0.
        file['dataset/data'][:] = records
    with pytest.warns(RuntimeWarning, match='axes recovered'):
        measured = read_pulseq_measurement(raw, sequence_path=sequence,
            reader_options={'trajectory_mode': mode, 'density': None})
    assert measured.metadata['trajectory_axes_from_sequence'] == (0,)
    assert measured.metadata['cartesian_trajectory_validated']
    if mode == 'auto':
        np.testing.assert_allclose(abs(reconstruct_acquisition(measured.acquisition).image), truth, atol=1e-6)
    with h5py.File(raw, 'r+') as file:
        records = file['dataset/data'][:]
        records['head']['idx']['kspace_encode_step_1'] = 0
        file['dataset/data'][:] = records
    with pytest.raises(ValueError, match='cannot be verified'):
        read_pulseq_measurement(raw, sequence_path=sequence)


def test_mrd_metadata_template_preserves_samples_and_checks_alignment(tmp_path):
    raw = tmp_path / 'raw.h5'
    template = tmp_path / 'metadata.h5'
    _write_mrd(raw)
    expected = read_pulseq_measurement(raw).acquisition
    file = ismrmrd.Dataset(str(raw), 'dataset', create_if_needed=False)
    metadata = ismrmrd.Dataset(str(template), 'dataset')
    header = _header(8, 6)
    header.encoding[0].encodingLimits.kspace_encoding_step_1.center = 0
    metadata.write_xml_header(xsd.ToXML(header))
    for i in range(file.number_of_acquisitions()):
        original = file.read_acquisition(i)
        skeleton = ismrmrd.Acquisition()
        skeleton.idx = original.idx
        metadata.append_acquisition(skeleton)
    file.close()
    metadata.close()
    import h5py

    with h5py.File(raw, 'r+') as file:
        del file['dataset/xml']
        records = file['dataset/data'][:]
        records['head']['idx']['kspace_encode_step_1'] = 100
        records['head']['idx']['contrast'] = 0
        file['dataset/data'][:] = records
    with pytest.warns(RuntimeWarning, match='metadata template'):
        measured = read_pulseq_measurement(raw, reader_options={'metadata_template': 'metadata.h5'})
    acquisition = measured.acquisition
    np.testing.assert_array_equal(acquisition.samples, np.roll(expected.samples, 3, axis=-2))
    assert acquisition.samples.shape == expected.samples.shape
    assert measured.metadata['metadata_template_used']
    with h5py.File(template, 'r+') as file:
        file['dataset/data'].resize((1,))
    with pytest.raises(ValueError, match='acquisition count'):
        read_pulseq_measurement(raw, reader_options={'metadata_template': template})


def test_noise_whitening_preserves_shared_receiver_encoding():
    from dataclasses import replace

    from mrax.data_struct import AcquisitionData
    from mrax.reco import apply_receiver_transform, estimate_noise_whitener, prewhiten_acquisition

    rng = np.random.default_rng(28)
    independent = rng.normal(size=(2, 4000)) + 1j * rng.normal(size=(2, 4000))
    noise = np.asarray([[2., 0.], [.4 + .3j, .7]]) @ independent
    noise = noise.reshape((2, 10, 400)).transpose(1, 0, 2)
    whitener = estimate_noise_whitener(noise)
    white = apply_receiver_transform(noise, whitener, coil_axis=1).transpose(1, 0, 2).reshape((2, -1))
    np.testing.assert_allclose(np.cov(white), np.eye(2), atol=1e-12)
    signal = rng.normal(size=(3, 2, 4, 6)) + 1j * rng.normal(size=(3, 2, 4, 6))
    acquisition = AcquisitionData(signal, (4, 6), ('phase', 'read'),
        auxiliary_samples={'noise': noise, 'calibration_grid': signal[0],
                           'epi_navigators': np.stack([signal] * 3, axis=1)})
    prepared, transform = prewhiten_acquisition(acquisition)
    np.testing.assert_allclose(prepared.samples, np.einsum('cd,fdij->fcij', transform, signal))
    np.testing.assert_allclose(prepared.auxiliary_samples['calibration_grid'], prepared.samples[0])
    np.testing.assert_allclose(prepared.auxiliary_samples['epi_navigators'][:, 0], prepared.samples)
    np.testing.assert_array_equal(acquisition.samples, signal)
    fft = reconstruct_acquisition(prepared)
    variational = reconstruct_acquisition(prepared, method='variational', optimizer_name='fista',
        num_steps=4, regularizer_weight=0.)
    np.testing.assert_allclose(variational.image, fft.image, rtol=2e-5, atol=2e-5)
    mask = np.ones(signal.shape, bool)
    mask[:, 1, 0, 0] = False
    with pytest.raises(ValueError, match='common receiver sampling'):
        prewhiten_acquisition(replace(acquisition, acquired=mask))
    with pytest.raises(ValueError, match='zero covariance'):
        estimate_noise_whitener(np.zeros((3, 2, 7)))
    with pytest.warns(RuntimeWarning, match='poorly conditioned'):
        singular = estimate_noise_whitener(np.tile(noise[:, :1], (1, 2, 1)))
    assert np.isfinite(singular).all()


def test_mrd_order_masks_geometry_and_shared_reconstruction(tmp_path):
    path = tmp_path / 'acquisition.h5'
    _write_mrd(path)
    measurement = read_pulseq_measurement(path)
    acq = measurement.acquisition
    assert acq.samples.shape == (2, 2, 6, 8)
    assert acq.acquired.all()
    np.testing.assert_array_equal(acq.samples[0, 0, 0], 0)
    np.testing.assert_array_equal(acq.samples[1, 1, 5], 12)
    assert acq.geometry.status == 'known'
    np.testing.assert_allclose(acq.geometry.voxel_sizes, [10, 10])
    result = reconstruct_acquisition(acq)
    expected = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(acq.samples, axes=(-2, -1)), axes=(-2, -1), norm='ortho'), axes=(-2, -1))
    np.testing.assert_allclose(result.image, np.sqrt(np.sum(np.abs(expected)**2, axis=1)), rtol=1e-6, atol=1e-6)
    exp, data, model, _ = import_pulseq_experiment(path, image_data=np.abs(result.image), override_type='mse')
    exp.validate(data)
    np.testing.assert_allclose(exp.observables['TE'].values.reshape(-1), [12, 36])
    assert model.name == 't2_se'


def test_slice_volumes_contrast_order_geometry_and_inverse_encoding(tmp_path):
    from mrax.reco.layout import image_to_encoding

    path = tmp_path / 'volume.h5'
    header = _header(4, 3)
    header.sequenceParameters.TE = [10., 30.]
    file = ismrmrd.Dataset(str(path), 'dataset')
    file.write_xml_header(xsd.ToXML(header))
    truth = np.zeros((2, 2, 3, 3, 4))  # echo, average, slice, row, column
    for average in range(2):
        for contrast, te in enumerate([10., 30.]):
            for label, physical_slice in enumerate([0, 2, 1]):
                image = np.full((3, 4), (1 + average * .1) * (1 + physical_slice * .2) * np.exp(-te / 70.))
                truth[contrast, average, physical_slice] = image
                raw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm='ortho'))
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
    experiment, dataset, _model, _metadata = build_measurement_experiment(measured, image_data=np.abs(image), override_type='mse')
    experiment.validate(dataset)
    np.testing.assert_allclose(dataset.data, truth, rtol=1e-6)
    np.testing.assert_allclose(dataset.spatial_geometry.voxel_sizes, [4, 20, 20])
    np.testing.assert_allclose(np.asarray(dataset.spatial_geometry.affine)[:3, 3], [30, 20, 0])
    predicted = image_to_encoding(dataset.data.reshape((4, 3, 3, 4)), dataset.acquisition)
    original = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(measured.acquisition.samples[:, 0], axes=(-2, -1)), norm='ortho'), axes=(-2, -1))
    np.testing.assert_allclose(predicted, original, rtol=1e-6, atol=1e-7)


def test_mrd_sampled_units_frame_selection_and_varying_trajectories(tmp_path):
    path = tmp_path / 'radial.h5'
    _write_mrd(path, sampled=True)
    m = read_pulseq_measurement(path, reader_options={'trajectory_units': 'cycles/fov'})
    assert m.acquisition.samples.shape == (2, 2, 48)
    assert m.acquisition.trajectory.shape == (48, 2)
    assert np.max(np.abs(m.acquisition.trajectory)) <= .75
    _write_mrd(tmp_path / 'varying.h5', sampled=True, bad_trajectory=True)
    with pytest.raises(ValueError, match='Trajectory changes'):
        read_pulseq_measurement(tmp_path / 'varying.h5', reader_options={'trajectory_units': 'cycles/fov'})
    selected = read_pulseq_measurement(path, reader_options={'frame_selection': [1], 'trajectory_units': 'cycles/fov'})
    assert selected.acquisition.frame_count == 1
    np.testing.assert_allclose(selected.normalized_metadata.echo_times, [36])


def test_mrd_extension_and_optional_header_information(tmp_path):
    import h5py

    from mrax.io.pulseq_measurement import discover_pulseq_measurements

    path = tmp_path / 'measurement.mrd'
    _write_mrd(path)
    with h5py.File(path, 'r+') as file:
        xml = file['dataset/xml'][0].decode()
        xml = xml.replace('</ismrmrdHeader>', '<measurementInformation><protocolName>synthetic</protocolName></measurementInformation></ismrmrdHeader>')
        file['dataset/xml'][0] = xml
    assert discover_pulseq_measurements(tmp_path) == [path]
    with pytest.warns(RuntimeWarning, match='optional MRD measurementInformation'):
        measurement = read_pulseq_measurement(path)
    assert measurement.acquisition.samples.shape == (2, 2, 6, 8)
    study, metadata = import_pulseq_study(path)
    assert not study.names
    assert len(metadata['raw_measurements']) == 1


def test_mrd_explicit_record_roles_and_cartesian_trajectory_override(tmp_path):
    path = tmp_path / 'measurement.mrd'
    _write_mrd(path, sampled=True)
    file = ismrmrd.Dataset(str(path), 'dataset')
    header = _header()
    file.write_xml_header(xsd.ToXML(header))
    file.close()
    with pytest.warns(RuntimeWarning, match='explicit Cartesian counters'):
        measurement = read_pulseq_measurement(path, reader_options={
            'trajectory_mode': 'cartesian', 'noise_records': [0], 'exclude_records': [1]})
    acquisition = measurement.acquisition
    assert acquisition.kind == 'cartesian'
    assert acquisition.auxiliary_samples['noise'].shape == (1, 2, 8)
    assert not acquisition.acquired[0, 0, 3].any()
    assert not acquisition.acquired[0, 0, 0].any()
    assert acquisition.acquired[1].all()
    with pytest.raises(ValueError, match='record indices'):
        read_pulseq_measurement(path, reader_options={'exclude_records': [-1]})
    _write_mrd(tmp_path / 'radial.mrd', sampled=True)
    with pytest.raises(ValueError, match='declared Cartesian'):
        read_pulseq_measurement(tmp_path / 'radial.mrd', reader_options={'trajectory_mode': 'cartesian'})


def test_mrd_explicit_sample_window_and_center(tmp_path):
    path = tmp_path / 'measurement.mrd'
    _write_mrd(path)
    full = read_pulseq_measurement(path).acquisition
    windowed = read_pulseq_measurement(path, reader_options={'sample_window': [1, 7], 'center_sample': 4}).acquisition
    np.testing.assert_array_equal(windowed.samples[..., 1:7], full.samples[..., 1:7])
    assert not windowed.acquired[..., (0, 7)].any()
    with pytest.raises(ValueError, match='sample_window'):
        read_pulseq_measurement(path, reader_options={'sample_window': [0, 9]})
    with pytest.raises(ValueError, match='center_sample'):
        read_pulseq_measurement(path, reader_options={'center_sample': 8})


@pytest.mark.parametrize('echoes,expected', [(np.array([8., 24., 40.]), 'mse'), (8., 'anatomical')])
def test_spin_echo_train_typing_requires_echo_variation(echoes, expected):
    from mrax.io.import_definitions import (
        NormalizedMetadata,
        detect_experiment_type,
        extract_observables_from_metadata,
    )

    meta = NormalizedMetadata(method_name='tse_cartesian', echo_times=echoes, inversion_times=np.array([30., 70., 140.]))
    assert detect_experiment_type(meta) == expected
    echo_observables = extract_observables_from_metadata('mse', meta, 3)
    inversion_observables = extract_observables_from_metadata('ir', meta, 3)
    np.testing.assert_array_equal(echo_observables['TE'].values, np.atleast_1d(echoes))
    np.testing.assert_array_equal(inversion_observables['TI'].values, [30., 70., 140.])


def test_nifti_dynamic_axes_scaling_and_raw_study_separation(tmp_path):
    array = np.arange(4 * 5 * 2 * 3, dtype=np.float32).reshape(4, 5, 2, 3)
    affine = np.diag([1.5, 2., 3., 1.])
    nib.save(nib.Nifti1Image(array, affine), tmp_path / 'dynamic.nii.gz')
    measurement = read_measurement(tmp_path / 'dynamic.nii.gz')
    assert measurement.image_data.shape == (3, 2, 5, 4)
    np.testing.assert_array_equal(measurement.image_data, array.transpose(3, 2, 1, 0))
    np.testing.assert_array_equal(measurement.spatial_geometry.affine, affine)
    _write_mrd(tmp_path / 'raw.h5')
    study, metadata = import_pulseq_study(tmp_path)
    assert len(study.experiments) == 1
    assert len(metadata['raw_measurements']) == 1
    assert not metadata['import_issues']
    with pytest.raises(ValueError, match='Raw-only'):
        build_measurement_experiment(metadata['raw_measurements']['raw.h5'])


def test_nifti_declared_time_and_space_units(tmp_path):
    image = nib.Nifti1Image(np.ones((4, 5, 2, 3), np.float32), np.diag([.001, .002, .003, 1.]))
    image.header.set_xyzt_units('meter', 'msec')
    image.header.set_zooms((.001, .002, .003, 250.))
    path = tmp_path / 'time.nii'
    nib.save(image, path)
    measurement = read_pulseq_measurement(path)
    np.testing.assert_allclose(measurement.metadata['frame_coordinates']['time'], [0, .25, .5])
    np.testing.assert_allclose(measurement.spatial_geometry.voxel_sizes, [3, 2, 1])
    exp, data, _model, _meta = build_measurement_experiment(measurement)
    np.testing.assert_allclose(exp.observables['time'].values.ravel(), [0, .25, .5])
    assert exp.observables['time'].si_unit == 's'


def test_volumetric_dicom_import_preserves_physical_voxel_locations(tmp_path):
    from mrax.io import export_image_files, load_dicom_series

    volume = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [7., -11., 19.]
    paths = export_image_files(tmp_path / 'spatial.dcm', volume,
        dims={'z': 3, 'y': 4, 'x': 5}, affine=affine, format='dicom')
    _experiment, dataset, _metadata = load_dicom_series(paths)
    np.testing.assert_array_equal(np.asarray(dataset.data)[0], volume.transpose(1, 2, 0))
    np.testing.assert_allclose(dataset.spatial_geometry.affine, affine[:, [2, 0, 1, 3]])
    original_world = affine @ np.array([3., 1., 2., 1.])
    imported_world = np.asarray(dataset.spatial_geometry.affine) @ np.array([2., 3., 1., 1.])
    np.testing.assert_allclose(imported_world, original_world)


def test_pulseq_native_seconds_and_origin_are_preserved():
    meta = _metadata_from_pulseq({'TE': [.012, .036], 'TR': 1., 'TI': .25, 'Flipangle': 15}, 2, Path('synthetic.seq'))
    assert meta.echo_times == [12., 36.]
    assert meta.repetition_times == [1000.]
    assert meta.inversion_times == [250.]
    physical = np.array([[-40., 0.], [30., 0.]])
    normalized = _normalize_trajectory(physical, spatial_shape=(6, 8), definitions={'FOV': [.1, .06]}, normalize=True)
    np.testing.assert_allclose(normalized[:, 0], [-.5, .375])


def test_serial_cartesian_centred_and_repeated_visits():
    ny, nx = 6, 8
    yy, xx = np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing='ij')
    physical = np.column_stack([xx.ravel() / .08, yy.ravel() / .06])
    info = PulseqSequenceInfo(Path('synthetic.seq'), definitions={'FOV': [.08, .06]}, trajectory_physical=np.tile(physical, (2, 1)), spatial_shape=(ny, nx))
    values = np.arange(2 * ny * nx, dtype=np.float32).astype(complex)[None]
    acq = prepare_pulseq_samples(info, values)
    np.testing.assert_array_equal(acq.samples[:, 0], values.reshape(2, ny, nx))
    assert acq.acquired.all()


def test_array_import_requires_explicit_receiver_axis(monkeypatch):
    from mrax.io import read_pulseq_acquisition

    ny, nx = 3, 4
    yy, xx = np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing='ij')
    physical = np.column_stack([xx.ravel() / .04, yy.ravel() / .03])
    info = PulseqSequenceInfo(Path('synthetic.seq'), definitions={'FOV': [.04, .03]},
        trajectory_physical=physical, adc_counts=(nx,) * ny, spatial_shape=(ny, nx), sequence_kind='cartesian')
    monkeypatch.setattr('mrax.io.pulseq_import.read_pulseq', lambda path: info)
    samples = np.stack([np.ones(ny * nx), np.full(ny * nx, 2.)])
    acq = read_pulseq_acquisition('synthetic.seq', samples, coil_axis=0)
    assert acq.samples.shape == (1, 2, ny, nx)
    np.testing.assert_array_equal(acq.samples[0, 1], 2.)


def test_reader_registration_and_archives_are_not_inputs(tmp_path):
    def reader(path):
        return MeasurementData(np.ones((1, 2, 3)))
    register_measurement_reader('synthetic-test', ('.synthetic',), reader)
    path = tmp_path / 'test.synthetic'
    path.touch()
    assert read_measurement(path).image_data.shape == (1, 2, 3)
    with pytest.raises(ValueError, match='already registered'):
        register_measurement_reader('synthetic-test', ('.synthetic',), reader)
    archive = tmp_path / 'archive.zip'
    archive.touch()
    with pytest.raises(ValueError, match='Unknown or ambiguous'):
        read_measurement(archive)


@pytest.mark.parametrize('t1', [420., 1420.])
def test_imported_ir_initialization_recovers_both_null_branches(t1):
    from mrax.io.import_definitions import NormalizedMetadata
    from mrax.io.measurement import MeasurementData

    times = np.asarray([20., 100., 240., 480., 800., 1300., 2100.])
    image = np.broadcast_to(np.abs(1.7 - 3.4 * np.exp(-times[:, None, None] / t1)), (7, 3, 4))
    measured = MeasurementData(image_data=image, normalized_metadata=NormalizedMetadata(
        inversion_times=times, frame_count=len(times)))
    experiment, dataset, model, _ = build_measurement_experiment(measured, override_type='ir')
    fields, _ = run_fit(experiment, model, dataset, num_steps=2000)
    np.testing.assert_allclose(fields['t1'].values, t1, rtol=2e-3)


def test_quantitative_mrd_fft_variational_and_model_fit_agree(tmp_path):
    """Known relaxation signals test units and the complete public fitting path."""
    nx, ny = 4, 3
    te = np.asarray([8., 24., 48., 96., 160.])
    curve = 1.2 * np.exp(-te / 70.) + .03
    images = np.broadcast_to(curve[:, None, None], (len(te), ny, nx))
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(images, axes=(-2, -1)), axes=(-2, -1), norm='ortho'), axes=(-2, -1))
    header = _header(nx, ny)
    header.sequenceParameters.TE = te.tolist()
    path = tmp_path / 'relaxation.h5'
    file = ismrmrd.Dataset(str(path), 'dataset')
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
    variational = reconstruct_acquisition(acquisition, method='variational', num_steps=8)
    np.testing.assert_allclose(variational.image, fft.image, atol=2e-5)
    experiment, dataset, model, _ = build_measurement_experiment(measurement, image_data=np.abs(fft.image), override_type='mse')
    fields, _ = run_fit(experiment, model, dataset, num_steps=2000, regularizer_weight=0.)
    direct = reconstruct_acquisition(acquisition, method='model_based', experiment=experiment, model=model,
        num_steps=2000, regularizer_weight=0.)
    for key in fields:
        np.testing.assert_allclose(fields[key].values, direct.parameters[key].values, rtol=3e-3, atol=2e-4)
    np.testing.assert_allclose(fields['t2'].values, 70., rtol=.04)
    from mrax.data_struct import Study, load_study_results
    from mrax.gui import GuiSession

    study = Study([experiment], [dataset], [model], names=['relaxation'])
    bundle = study.save(tmp_path / 'study.mrax')
    restored = load_study_results(bundle)
    assert restored['experiments']['relaxation']['model_parameter_specs']['amplitude']['scale'] < 2.
    session = GuiSession.from_manifest(bundle)
    _experiment, restored_model, _dataset, _study = session.build_fit_inputs()
    params, unpack = model.parametrization_factory((ny, nx))
    restored_params, restored_unpack = restored_model.parametrization_factory((ny, nx))
    for key, value in unpack(params).items():
        np.testing.assert_array_equal(restored_unpack(restored_params)[key], value)


def test_observable_library_and_boolean_spin_lock(tmp_path):
    from mrax.io.pulseq_metadata import sequence_metadata

    extension = tmp_path / 'observables.toml'
    extension.write_text('schema_version = 1\n[observables.echo_times]\naliases = ["echo_ms"]\nscale = 1.0\n')
    metadata = sequence_metadata({'echo_ms': [11, 29], 'SpinLock': 'False'}, 2, 'synthetic.seq', definition_paths=[extension])
    assert metadata.echo_times == [11., 29.]
    assert metadata.spin_lock is False
    assert sequence_metadata({'SpinLock': 'True'}, 1, 'synthetic.seq').spin_lock is True


def test_density_preserves_repeated_sample_quadrature():
    from mrax.reco.density import triangulated_density

    points = np.array([[-.25, -.25], [-.25, .25], [.25, -.25], [.25, .25]])
    weights = triangulated_density(points, spatial_shape=(4, 4))
    repeated = triangulated_density(np.repeat(points, 2, axis=0), spatial_shape=(4, 4))
    np.testing.assert_allclose(repeated.reshape(-1, 2).sum(1), weights)
    assert np.sum(weights) == pytest.approx(4.)


def test_calibration_grid_and_sense_unfolding(tmp_path):
    from mrax.reco import estimate_cartesian_sensitivities

    nx, ny = 32, 24
    y, x = np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing='ij')
    image = np.exp(-(x / 6.)**2 - (y / 4.)**2)
    coils = np.stack([np.ones_like(image), np.exp(2j * np.pi * y / ny)]) / np.sqrt(2.)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image[None] * coils, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))
    path = tmp_path / 'accelerated.h5'
    file = ismrmrd.Dataset(str(path), 'dataset')
    file.write_xml_header(xsd.ToXML(_header(nx, ny)))
    for calibration, lines in [(False, range(0, ny, 2)), (True, range(8, 16))]:
        for line in lines:
            record = ismrmrd.Acquisition()
            record.resize(nx, 2)
            record.center_sample = nx // 2
            record.idx.kspace_encode_step_1 = line
            record.data[:] = kspace[:, line]
            if calibration:
                record.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
            file.append_acquisition(record)
    file.close()
    with pytest.warns(RuntimeWarning, match='undersampled'):
        measured = read_pulseq_measurement(path)
    acq = measured.acquisition
    assert acq.auxiliary_samples['calibration_grid'].shape == (2, 8, nx)
    sensitivity = estimate_cartesian_sensitivities(acq, smoothing_sigma=1.)
    preview = reconstruct_acquisition(acq).image[0]
    solved = reconstruct_acquisition(acq, method='variational', sensitivities=sensitivity,
        optimizer_name='fista', regularizer='none', regularizer_weight=0., num_steps=100).image[0]
    preview_error = np.linalg.norm(preview - image) / np.linalg.norm(image)
    solved_error = np.linalg.norm(solved - image) / np.linalg.norm(image)
    assert solved_error < .05
    assert solved_error < preview_error / 5


@pytest.mark.parametrize('complex_image', [False, True])
@pytest.mark.parametrize('frames', [False, True])
def test_streamed_sense_objective_and_derivatives_match_dense(complex_image, frames):
    import jax
    import jax.numpy as jnp

    from mrax.reco.cartesian_loss import cartesian_sense_loss
    from mrax.reco.variational import fftnc

    rng = np.random.default_rng(143)
    shape = (4, 5)
    image = rng.normal(size=((2,) if frames else ()) + shape).astype(np.float32)
    if complex_image:
        image = image + 1j * rng.normal(size=image.shape).astype(np.float32)
    sensitivity = rng.normal(size=(3,) + shape) + 1j * rng.normal(size=(3,) + shape)
    sensitivity = jnp.asarray(sensitivity, jnp.complex64)
    data_shape = ((2,) if frames else ()) + sensitivity.shape
    data = jnp.asarray(rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape), jnp.complex64)
    weights = jnp.asarray(rng.uniform(size=((2, 1) if frames else (1,)) + shape), jnp.float32)
    weights = jnp.where(weights < .3, 0, weights)

    def dense(value):
        predicted = fftnc(jnp.expand_dims(value, -3) * sensitivity, axes=(-2, -1))
        weight = jnp.broadcast_to(weights, data.shape)
        return .5 * jnp.sum(weight * jnp.abs(predicted - data)**2) / jnp.sum(weight)

    streamed = cartesian_sense_loss(data, weights, sensitivity, shape)
    np.testing.assert_allclose(streamed(image), dense(image), rtol=2e-6)
    np.testing.assert_allclose(jax.grad(streamed)(image), jax.grad(dense)(image), rtol=3e-6, atol=2e-7)
    direction = jnp.ones_like(image)
    np.testing.assert_allclose(jax.jvp(streamed, (image,), (direction,))[1],
        jax.jvp(dense, (image,), (direction,))[1], rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(jax.jvp(jax.grad(streamed), (image,), (direction,))[1],
        jax.jvp(jax.grad(dense), (image,), (direction,))[1], rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize('segmented', [False, True])
def test_siemens_adc_transforms_are_disabled_and_samples_stay_measured(tmp_path, monkeypatch, segmented):
    import sys
    import types

    from mrax.io.siemens_measurement import read_siemens_measurement

    class Raw:
        NCol, NAcq = 4, 3
        Lin = np.arange(3)
        scancounter = np.arange(3)
        memPos = np.arange(3)
        Ave = Eco = Phs = Rep = Sli = np.zeros(3)
        Set = np.arange(3) if segmented else np.zeros(3)
        flagRemoveOS = flagRampSampRegrid = flagDoAverage = True

        def readData(self, positions, cIxToRaw=None):
            assert not self.flagRemoveOS and not self.flagRampSampRegrid
            return np.arange(12).reshape(4, 1, 3).astype(np.complex64)[:, :, cIxToRaw]

    def read(path, **options):
        assert options['disableReflect'] and options['ignoreROoffcenter']
        return {'image': Raw()}

    monkeypatch.setitem(sys.modules, 'mapvbvd', types.SimpleNamespace(mapVBVD=read))
    info = PulseqSequenceInfo(Path('synthetic.seq'), adc_counts=(12,) if segmented else (4, 4, 4),
        adc_times=np.arange(12) * 2e-5)
    measurement = read_siemens_measurement(tmp_path / 'adc.dat', sequence_info=info)
    assert measurement.acquisition.domain == 'adc'
    np.testing.assert_allclose(measurement.acquisition.frame_coordinates['time'], [11e-5])
    assert measurement.metadata['frame_coordinate_units']['time'] == 's'
    np.testing.assert_array_equal(measurement.acquisition.samples[0, 0], np.arange(12).reshape(4, 3).T.ravel())


def test_compressed_shape_reader_does_not_patch_pypulseq(tmp_path):
    import importlib

    import pypulseq

    from mrax.io.pulseq_compat import read_sequence

    module = importlib.import_module('pypulseq.Sequence.read_seq')
    original = module.__dict__['__read_shapes']
    path = tmp_path / 'compressed.seq'
    path.write_text('[VERSION]\nmajor 1\nminor 3\nrevision 1\n\n[SHAPES]\nshape_id 1\nnum_samples 3\n0.1\n0.1\n1\n\n')
    sequence = pypulseq.Sequence()
    with pytest.warns(UserWarning):
        read_sequence(sequence, path)
    np.testing.assert_allclose(sequence.shape_library.data[1], [3., .1, .2, .3])
    assert module.__dict__['__read_shapes'] is original


@pytest.mark.parametrize('explicit', [False, True])
def test_runtime_constant_buffers_respect_explicit_environment(monkeypatch, explicit):
    import types

    import mrax.devices as devices

    calls = []
    config = types.SimpleNamespace(jax_use_simplified_jaxpr_constants=False, update=lambda *args: calls.append(args))
    monkeypatch.setattr(devices, 'jax', types.SimpleNamespace(config=config))
    monkeypatch.setattr(devices, '_constant_buffers_configured', False)
    monkeypatch.delenv('JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS', raising=False)
    if explicit:
        monkeypatch.setenv('JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS', 'false')
    devices._configure_constant_buffers()
    devices._configure_constant_buffers()
    assert calls == ([] if explicit else [('jax_use_simplified_jaxpr_constants', True)])


def test_line_factorization_preserves_fractional_readout_and_oversampling():
    from mrax.io.line_sampling import prepare_line_acquisition
    from mrax.reco.variational import build_operator

    ny, nx = 6, 16
    read = (np.arange(nx) - nx // 2 + .25) / nx
    phase = (np.arange(ny) - ny // 2) / ny
    points = np.stack(np.meshgrid(read, phase), axis=-1).reshape((-1, 2))
    # FOV determines physical frequencies, including a quarter-sample offset.
    fov = np.array([.16, .06])
    physical = points * np.array([nx, ny]) / fov
    rng = np.random.default_rng(21)
    image = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    positions = np.stack(np.meshgrid(np.arange(nx) - nx // 2, np.arange(ny) - ny // 2), axis=-1).reshape((-1, 2))
    measured = (np.exp(-2j * np.pi * (points @ positions.T)) @ image.ravel() / np.sqrt(nx * ny)).astype(np.complex64)
    acquisition = prepare_line_acquisition(measured[None], physical, fov, (ny, nx), (ny, nx // 2), (nx,) * ny)
    assert acquisition.kind == 'cartesian' and acquisition.readout_coordinates is not None
    assert acquisition.recipe.output_scale == pytest.approx(np.sqrt(.5))
    op = build_operator(spatial_shape=(ny, nx), readout_coordinates=acquisition.readout_coordinates,
        readout_polarity=acquisition.readout_polarity, independent_coils=True)
    np.testing.assert_allclose(op.forward(image[None, None]), acquisition.samples, rtol=2e-5, atol=5e-6)


def test_siemens_scanner_planes_use_phase_read_slice_quaternion_columns():
    import types

    from mrax.data_struct import AcquisitionData, ReconstructionRecipe
    from mrax.io.siemens_measurement import _attach_geometry

    acquisition = AcquisitionData(np.ones((1, 1, 6, 8), np.complex64), (6, 8), ('phase', 'read'),
        recipe=ReconstructionRecipe(output_shape=(6, 8)))
    image = types.SimpleNamespace(slicePos=np.tile([10., 20., 30., 1., 0., 0., 0.], (6, 1)))
    info = PulseqSequenceInfo(Path('test.seq'), definitions={'FOV': [.08, .06, .004], 'SliceThickness': .004})
    result = _attach_geometry(acquisition, image, np.arange(6), np.zeros(6, int), info, None, None)
    affine = np.asarray(result.geometry.affine)
    # Quaternion identity: physical read is LPS Y; phase is LPS X.
    np.testing.assert_allclose(affine[:3, :3], [[0, -10, 0], [-10, 0, 0], [0, 0, 4]])
    np.testing.assert_allclose(affine @ [3.5, 2.5, 0, 1], [-10, -20, 30, 1])


def test_line_factorization_accepts_positive_nyquist_phase_boundary():
    from mrax.io.line_sampling import prepare_line_acquisition

    phase = np.array([2., -1., 0., 1.])
    read = np.arange(8) - 4
    physical = np.stack([np.tile(read, 4), np.repeat(phase, 8)], axis=1)
    samples = np.repeat(np.arange(4) + 1j, 8)[None].astype(np.complex64)
    acquisition = prepare_line_acquisition(samples, physical, (1., 1.), (4, 8), (4, 8), (8,) * 4)
    assert np.all(acquisition.acquired)
    np.testing.assert_array_equal(acquisition.samples[0, 0, :, 0], np.arange(4) + 1j)


def test_sequence_slice_offsets_resolve_prototype_planes_without_overwriting_known_positions(tmp_path):
    from mrax.io.ismrmrd_measurement import read_ismrmrd_measurement
    from mrax.io.measurement_layout import sequence_slice_positions

    path = tmp_path / 'slices.h5'
    header = _header(4, 3)
    file = ismrmrd.Dataset(str(path), 'dataset')
    file.write_xml_header(xsd.ToXML(header))
    for label in (1, 0):
        for line in range(3):
            record = ismrmrd.Acquisition()
            record.resize(4, 1)
            record.idx.slice = label
            record.idx.kspace_encode_step_1 = line
            record.center_sample = 2
            record.position[:] = [10., 20., 30.]
            record.read_dir[:] = [1., 0., 0.]
            record.phase_dir[:] = [0., 1., 0.]
            record.slice_dir[:] = [0., 0., 1.]
            record.data[:] = 1
            file.append_acquisition(record)
    file.close()
    with pytest.warns(RuntimeWarning, match='share one scanner plane'):
        unresolved = read_ismrmrd_measurement(path)
    assert unresolved.spatial_geometry.status == 'partial'
    info = PulseqSequenceInfo(Path('slices.seq'), definitions={'SlicePositions': [-.003, .003]})
    measured = read_ismrmrd_measurement(path, sequence_info=info)
    assert measured.acquisition.recipe.stack_slices == 2
    assert measured.spatial_geometry.status == 'known'
    np.testing.assert_allclose(measured.spatial_geometry.voxel_sizes, [6., 20., 20.])
    selected = read_ismrmrd_measurement(path, sequence_info=info, frame_selection=slice(0, 1))
    np.testing.assert_allclose(np.asarray(selected.spatial_geometry.affine)[2, 3], 27.)
    known = np.array([[0., 0., 3.], [0., 0., 9.]])
    np.testing.assert_array_equal(sequence_slice_positions([0, 1], known, [[0, 0, 1]] * 2,
        {'SlicePositions': [-.003, .003]}), known)


def test_study_import_combines_dicom_nifti_and_skips_invalid_measurements(tmp_path):
    from mrax.data_struct import SpatialGeometry
    from mrax.io import export_image_files

    geometry = SpatialGeometry(affine=np.diag([2., 3., 4., 1.]), spatial_shape=(3, 4, 5), status='known')
    export_image_files(tmp_path / 'image.dcm', np.ones((3, 4, 5), np.int16),
        format='dicom', geometry=geometry)
    nib.save(nib.Nifti1Image(np.ones((5, 4, 3)), np.eye(4)), tmp_path / 'image.nii')
    (tmp_path / 'invalid.h5').write_bytes(b'invalid')
    with pytest.warns(RuntimeWarning):
        study, metadata = import_pulseq_study(tmp_path)
    assert len(study.experiments) == 2
    assert len(metadata['import_issues']) == 1
    for experiment, dataset in zip(study.experiments, study.datasets):
        experiment.validate(dataset)


def test_unresolved_siemens_orientation_retains_known_voxel_spacing():
    import types

    from mrax.data_struct import AcquisitionData
    from mrax.io.siemens_measurement import _attach_geometry

    acquisition = AcquisitionData(np.ones((1, 1, 4, 6), np.complex64), (4, 6), ('phase', 'read'))
    image = types.SimpleNamespace(slicePos=np.tile([0., 0., 0., 1., 0., 0., 0.], (4, 1)))
    info = PulseqSequenceInfo(Path('test.seq'), definitions={'FOV': [.06, .08], 'OrientationMapping': 'CUSTOM'})
    with pytest.warns(RuntimeWarning, match='gradient_directions'):
        result = _attach_geometry(acquisition, image, np.arange(4), np.zeros(4, int), info, None, None)
    assert result.geometry.status == 'partial' and result.geometry.affine is None
    np.testing.assert_allclose(result.geometry.voxel_sizes, [20., 10.])


def test_file_overrides_apply_to_each_mrd_image_series(tmp_path):
    path = tmp_path / 'series.h5'
    file = ismrmrd.Dataset(str(path), 'dataset')
    file.write_xml_header(xsd.ToXML(_header(4, 3)))
    for index in range(2):
        image = ismrmrd.Image.from_array(np.ones((3, 4), np.float32), transpose=False)
        image.contrast = index
        file.append_image(f'image_{index}', image)
    file.close()
    study, metadata = import_pulseq_study(path, type_overrides={'series.h5': 'mse'},
        metadata_overrides={'series.h5': {'echo_times': [20.]}})
    assert len(study.experiments) == 2 and not metadata['import_issues']
    assert float(study.study_globals['field_strength_mhz'].values) == 128.
    for experiment in study.experiments:
        assert experiment.exp_type == 'mse'
        np.testing.assert_allclose(experiment.observables['TE'].values, 20.)


def test_strided_adc_profiles_retain_polarity_without_mutating_source():
    from mrax.io.line_sampling import prepare_line_acquisition

    ny, nx, coils = 4, 8, 2
    read = np.arange(nx) - nx // 2
    phase = np.arange(ny) - ny // 2
    physical = np.stack([np.tile(read, ny), np.repeat(phase, nx)], axis=1).reshape((ny, nx, 2))
    physical[1::2] = physical[1::2, ::-1]
    raw = np.asfortranarray(np.arange(nx * coils * ny).reshape((nx, coils, ny)).astype(np.complex64))
    profiles = raw.transpose(1, 2, 0)
    original = raw.copy()
    acquisition = prepare_line_acquisition(profiles, physical.reshape((-1, 2)), (1., 1.),
        (ny, nx), (ny, nx), (nx,) * ny)
    expected = profiles.copy()
    expected[:, 1::2] = expected[:, 1::2, ::-1]
    np.testing.assert_array_equal(acquisition.samples[0], expected)
    np.testing.assert_array_equal(raw, original)
    info = PulseqSequenceInfo(Path('test.seq'), definitions={'FOV': [1., 1.]}, spatial_shape=(ny, nx),
        trajectory_physical=physical.reshape((-1, 2)), adc_counts=(nx,) * ny)
    prepared = prepare_pulseq_samples(info, profiles)
    np.testing.assert_array_equal(prepared.samples, acquisition.samples)


def test_raw_frame_selection_preserves_requested_order_and_rejects_duplicates(tmp_path):
    path = tmp_path / 'frames.h5'
    _write_mrd(path)
    complete = read_pulseq_measurement(path).acquisition
    selected = read_pulseq_measurement(path, reader_options={'frame_selection': [1, 0]}).acquisition
    np.testing.assert_array_equal(selected.samples, complete.samples[[1, 0]])
    np.testing.assert_array_equal(selected.frame_coordinates['contrast'], [1, 0])
    with pytest.raises(ValueError, match='must not repeat'):
        read_pulseq_measurement(path, reader_options={'frame_selection': [0, 0]})


@pytest.mark.parametrize('combine', ['rss', 'sense', 'none'])
def test_fft_frame_batches_preserve_slice_repeat_order_and_coil_images(combine):
    from mrax.data_struct import AcquisitionData, ReconstructionRecipe

    rng = np.random.default_rng(31)
    shape = (12, 2, 4, 6)
    samples = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    mask = rng.random((12, 1, 4, 6)) > .1
    recipe = ReconstructionRecipe(stack_slices=3, average_repeats=2,
        output_stack_permutation=(2, 0, 1),
        output_frame_indices=tuple(np.arange(12).reshape(4, 3)[:, ::-1].ravel()),
        output_shape=(4, 4), output_flips=(1,), coil_combine=combine,
        frame_image_shifts=tuple((float(i % 2) / 3, .2) for i in range(12)))
    sensitivities = np.broadcast_to(np.array([1., .7j])[None, :, None, None], shape).astype(np.complex64)
    acquisition = AcquisitionData(samples, (4, 6), ('phase', 'read'), acquired=mask,
        statistical_weights=np.broadcast_to(mask, shape).astype(float), recipe=recipe)
    kwargs = {'sensitivities': sensitivities} if combine == 'sense' else {}
    dense = reconstruct_acquisition(acquisition, frame_batch_size=0, **kwargs)
    from mrax.reco import ReconstructionPipeline

    plan = ReconstructionPipeline.from_acquisition(acquisition, frame_batch_size=6)
    formatting = next(stage for stage in plan.stages if stage.kind == 'format')
    assert formatting.options['output_stack_permutation'] == (2, 0, 1)
    batch = plan.run(acquisition, **kwargs)
    np.testing.assert_allclose(batch.image, dense.image, atol=2e-6)
    np.testing.assert_allclose(batch.coil_images, dense.coil_images, atol=2e-6)
    assert batch.diagnostics[0]['stage'] == 'frame_batches'
    assert isinstance(batch.coil_images, np.ndarray)


def test_sampled_fft_frame_batches_match_dense_gridding():
    from mrax.data_struct import AcquisitionData
    from mrax.io.measurement import sampled_reconstruction_recipe

    rng = np.random.default_rng(42)
    trajectory = rng.uniform(-.4, .4, (32, 2)).astype(np.float32)
    samples = (rng.normal(size=(4, 2, 32)) + 1j * rng.normal(size=(4, 2, 32))).astype(np.complex64)
    acquisition = AcquisitionData(samples, (8, 8), ('phase', 'read'), kind='sampled', trajectory=trajectory,
        recipe=sampled_reconstruction_recipe(output_shape=(8, 8)))
    dense = reconstruct_acquisition(acquisition, frame_batch_size=0)
    batched = reconstruct_acquisition(acquisition, frame_batch_size=2)
    np.testing.assert_allclose(batched.image, dense.image, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(batched.coil_images, dense.coil_images, rtol=2e-6, atol=2e-6)


def test_siemens_epi_navigators_enable_generic_phase_stage_without_loading_auxiliary(tmp_path, monkeypatch):
    import sys
    import types

    from mrax.io.siemens_measurement import read_siemens_measurement

    class Raw:
        NCol, NAcq = 4, 3
        Lin = np.arange(3)
        scancounter = np.arange(3)
        memPos = np.arange(3)
        Ave = Eco = Phs = Rep = Set = Sli = np.zeros(3)

        def readData(self, positions, cIxToRaw=None):
            return np.ones((4, 1, 3), np.complex64)[:, :, cIxToRaw]

    monkeypatch.setitem(sys.modules, 'mapvbvd', types.SimpleNamespace(
        mapVBVD=lambda *args, **kwargs: {'image': Raw(), 'phasecor': object()}))
    read = np.tile(np.arange(4) - 2, (3, 1))
    read[1] = read[1, ::-1]
    physical = np.stack([read / .04, np.broadcast_to((np.arange(3) - 1)[:, None] / .03, (3, 4))], axis=-1)
    info = PulseqSequenceInfo(Path('test.seq'), definitions={'FOV': [.04, .03]}, spatial_shape=(3, 4),
        adc_counts=(4, 4, 4), trajectory_physical=physical.reshape((-1, 2)))
    measured = read_siemens_measurement(tmp_path / 'test.dat', sequence_info=info, load_auxiliary=False)
    assert measured.acquisition.recipe.epi_phase_correction == 'entropy'
    assert not measured.acquisition.auxiliary_samples


@pytest.mark.parametrize('outer_sign', [1, -1])
def test_timed_epi_navigators_remove_ghosts_and_survive_frame_batches(outer_sign, tmp_path):
    from mrax.data_struct import AcquisitionData, ReconstructionRecipe
    from mrax.reco.epi import calibrate_epi_navigators, correct_readout_phase
    from mrax.reco.variational import fftnc

    frames, coils, ny, nx = 4, 2, 16, 24
    x = (np.arange(nx) - nx // 2) / nx
    y = (np.arange(ny) - ny // 2) / ny
    image = np.exp(-x[None]**2 / .02 - y[:, None]**2 / .03)
    coil_images = np.broadcast_to(image, (frames, coils, ny, nx)).astype(np.complex64)
    negative = np.broadcast_to((np.arange(ny) % 2 == 1)[None, None, :, None], (frames, 1, ny, 1))
    coefficients = np.array([.3, 1.8])
    clean = np.asarray(fftnc(coil_images))
    corrupted = np.asarray(correct_readout_phase(clean, negative, -coefficients))
    times = np.array([1., 1.003, 1.008])
    polarity = np.array([outer_sign, -outer_sign, outer_sign])
    navigators = []
    signal = np.exp(-x**2 / .02)[None] * np.array([1., .8j])[:, None]
    for sign, time in zip(polarity, times):
        phase = 20. * (time - times[1]) + (coefficients[0] + coefficients[1] * x if sign < 0 else 0)
        navigators.append(np.asarray(fftnc(signal * np.exp(1j * phase), axes=(-1,))))
    aux = {'epi_navigators': np.broadcast_to(navigators, (frames, 3, coils, nx)),
           'epi_navigator_polarity': np.broadcast_to(polarity, (frames, 3)),
           'epi_navigator_times': np.broadcast_to(times, (frames, 3))}
    acquisition = AcquisitionData(corrupted, (ny, nx), ('phase', 'read'), readout_polarity=np.where(negative, -1, 1),
        auxiliary_samples={'epi_navigators': aux['epi_navigators']},
        auxiliary_coordinates={key: value for key, value in aux.items() if key != 'epi_navigators'},
        auxiliary_frame_axes={key: 0 for key in aux}, recipe=ReconstructionRecipe(epi_phase_correction='navigator'))
    from mrax.io.acquisition import load_acquisition, save_acquisition

    saved = tmp_path / 'calibrated.npz'
    save_acquisition(saved, acquisition)
    restored = load_acquisition(saved).select_frames([2, 0])
    np.testing.assert_allclose(calibrate_epi_navigators(restored, 1), coefficients, atol=2e-5)
    np.testing.assert_allclose(calibrate_epi_navigators(acquisition, 0), coefficients, atol=2e-5)
    for batch_size in (0, 2):
        result = reconstruct_acquisition(acquisition, frame_batch_size=batch_size)
        np.testing.assert_allclose(result.image, np.sqrt(coils) * np.abs(coil_images[:, 0]), rtol=2e-5, atol=2e-6)


def test_sequence_adc_times_retain_submillisecond_precision_late_in_scan(tmp_path, monkeypatch):
    import types

    import pypulseq

    from mrax.io import read_pulseq

    times = 3600. + np.arange(4) * 2e-6
    sequence = types.SimpleNamespace(definitions={'FOV': [.04, .04]},
        calculate_kspace=lambda: (np.zeros((3, 4)), None, None, None, times))
    monkeypatch.setattr(pypulseq, 'Sequence', lambda: sequence)
    monkeypatch.setattr('mrax.io.pulseq_compat.read_sequence', lambda *args: None)
    path = tmp_path / 'timing.seq'
    path.write_text('[VERSION]\nmajor 1\nminor 4\nrevision 0\n\n[DEFINITIONS]\nFOV .04 .04\n')
    info = read_pulseq(path)
    np.testing.assert_array_equal(info.adc_times, times)
    assert np.all(np.diff(info.adc_times) > 0)


def test_siemens_navigator_grouping_uses_frame_labels_adc_timing_and_polarity():
    import types

    from mrax.io.siemens_measurement import _group_epi_navigators

    counts = (8,) * 6
    signs = np.array([1, -1, 1] * 2)
    read = np.tile(np.arange(8) - 4, (6, 1))
    read[signs < 0] = read[signs < 0, ::-1]
    physical = np.stack([read, np.zeros_like(read)], axis=-1).reshape((-1, 2))
    starts = np.array([0., .002, .004, 1., 1.002, 1.004])
    times = starts[:, None] + np.arange(8) * 1e-5
    info = PulseqSequenceInfo(Path('test.seq'), adc_counts=counts, trajectory_physical=physical, adc_times=times.ravel())
    reference = types.SimpleNamespace(NCol=8, scancounter=np.arange(6), Rep=np.repeat([0, 1], 3))
    raw = np.arange(6 * 2 * 8).reshape((6, 2, 8)).astype(np.complex64)
    samples, coordinates = _group_epi_navigators(reference, raw, info, np.arange(6),
        np.array([[0], [1]]), np.array([[1], [0]]), ('Rep',))
    np.testing.assert_array_equal(samples['epi_navigators'][0, 0], raw[3])
    np.testing.assert_array_equal(samples['epi_navigators'][0, 1], raw[4, :, ::-1])
    np.testing.assert_array_equal(coordinates['epi_navigator_polarity'], signs.reshape(2, 3))
    np.testing.assert_allclose(coordinates['epi_navigator_times'], times.reshape(2, 3, 8)[[1, 0]].mean(-1))
