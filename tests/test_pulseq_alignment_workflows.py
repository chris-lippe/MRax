"""Synthetic quantitative import conventions, independent of scanner datasets."""
import json
from dataclasses import replace
from types import SimpleNamespace

import h5py
import ismrmrd
import nibabel as nib
import numpy as np
import optax
import pytest
from ismrmrd import xsd
from test_pulseq_measurement_workflows import _header

from mrax.core import run_fit
from mrax.gui import GuiSession
from mrax.gui.pulseq_import import plan_pulseq_import
from mrax.io import read_measurement, read_pulseq_measurement
from mrax.io.measurement import build_measurement_experiment
from mrax.io.pulseq_import import PulseqSequenceInfo, _metadata_from_pulseq
from mrax.io.pulseq_measurement import discover_pulseq_measurements
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
