"""GUI Pulseq import checks with independently generated measurements."""
from pathlib import Path
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from mrax.core.loss_builder import make_model_callable, run_fit
from mrax.gui import GuiSession
from mrax.gui.app import _detect_import_vendor, _MainWindowMixin
from mrax.gui.pulseq_import import plan_pulseq_import


def _sequence(path):
    path.write_text('[DEFINITIONS]\nName inversion_recovery\nTI 0.03 0.1 0.4\nFREQ 120\n')
    return path


def _images(path):
    curve = np.abs(1000 * (1 - 2 * np.exp(-np.array([30., 100., 400.]) / 180.)))
    data = np.broadcast_to(curve, (4, 3, 2, 3)).copy()
    nib.save(nib.Nifti1Image(data, np.diag([2., 3., 4., 1.])), path)


def _controller(source):
    controller = object.__new__(_MainWindowMixin)
    controller._task_running = False
    controller._busy_message = None
    controller._get_import_source = Mock(return_value=str(source))
    controller._show_import_options_dialog = Mock(return_value={})
    controller.run_guarded_task = Mock()
    controller.show_error = Mock()
    return controller


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


def test_pulseq_missing_measurements_prompts_and_cancellation_is_safe(tmp_path):
    sequence = _sequence(tmp_path / 'relaxation.seq')
    data = tmp_path / 'measurements'
    data.mkdir()
    _images(data / 'images.nii')
    controller = _controller(sequence)
    controller._get_import_source.side_effect = [str(sequence), str(data)]
    controller.import_study_dialog()
    assert controller._get_import_source.call_count == 2
    assert controller._get_import_source.call_args.kwargs['measurements']
    args, kwargs = controller._show_import_options_dialog.call_args
    assert args == (sequence, 'Pulseq')
    assert len(kwargs['summary']) == 1
    controller.run_guarded_task.assert_called_once()
    controller.show_error.assert_not_called()
    for answers in ([str(sequence), ''], ['']):
        controller = _controller(sequence)
        controller._get_import_source.side_effect = answers
        controller.import_study_dialog()
        controller.run_guarded_task.assert_not_called()
        controller._show_import_options_dialog.assert_not_called()


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


def test_pulseq_overview_cancellation_and_existing_bruker_routing(tmp_path):
    from test_import_workflows import _write_minimal_bruker_scan

    sequence = _sequence(tmp_path / 'relaxation.seq')
    _images(tmp_path / 'images.nii')
    controller = _controller(sequence)
    controller._show_import_options_dialog.return_value = None
    controller.import_study_dialog()
    controller.run_guarded_task.assert_not_called()
    bruker = tmp_path / 'bruker'
    bruker.mkdir()
    _write_minimal_bruker_scan(bruker)
    # A nearby descriptor must never override a recognized Bruker study.
    _sequence(bruker / 'unrelated.seq')
    assert _detect_import_vendor(bruker) == 'Bruker'
    controller = _controller(bruker)
    controller.import_study_dialog()
    controller._show_import_options_dialog.assert_called_once_with(bruker, 'Bruker')
    assert controller.run_guarded_task.call_args.args[1] == controller.on_bruker_import_finished
    controller.show_error.assert_not_called()


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


def test_pulseq_auto_discovery_uses_overview_without_second_picker(tmp_path):
    sequence = _sequence(tmp_path / 'relaxation.seq')
    _images(tmp_path / 'images.nii')
    controller = _controller(sequence)
    controller.import_study_dialog()
    controller._get_import_source.assert_called_once()
    assert controller._show_import_options_dialog.call_args.args == (sequence, 'Pulseq')
    assert controller.run_guarded_task.call_args.args[1] == controller.on_pulseq_import_finished


def test_pulseq_dicom_preparation_import_preserves_geometry_and_scanner_frequency(tmp_path):
    from pydicom import dcmread
    from pydicom.uid import generate_uid
    from test_import_workflows import _write_classic_dicom_slice

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
    from test_pulseq_measurement_workflows import _write_mrd

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


@pytest.mark.parametrize('vendor', ['Pulseq', 'Bruker'])
def test_import_overview_shows_vendor_and_corresponding_options(tmp_path, monkeypatch, vendor):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    from PySide6 import QtCore

    import mrax.gui.app as app

    monkeypatch.setattr(app, 'QtWidgets', QtWidgets, raising=False)
    monkeypatch.setattr(app, 'QtCore', QtCore, raising=False)
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Window(_MainWindowMixin, QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)

        def closeEvent(self, event):
            QtWidgets.QWidget.closeEvent(self, event)

    window = Window()
    summary = [{'scan': 'images', 'method': 'synthetic', 'detected_type': 'ir'}]

    def inspect(dialog):
        labels = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
        assert ('pdata index' in labels) == (vendor == 'Bruker')
        assert dialog.findChildren(QtWidgets.QComboBox)[0].currentText() == vendor
        return QtWidgets.QDialog.Accepted

    window._exec_dialog = inspect
    options = window._show_import_options_dialog(tmp_path, vendor, summary=summary)
    assert options['vendor'] == vendor
    assert options['load_raw_kspace'] == (vendor == 'Pulseq')
    assert options['selected_scans'] == ['images']
    if vendor == 'Pulseq':
        assert not options['model_overrides_by_name']
        assert not options['type_overrides']
    else:
        assert options['type_overrides'] == {'images': 'ir'}
    window._get_existing_directory = Mock(return_value=str(tmp_path))
    window._get_open_file_name = Mock(return_value=(str(tmp_path / 'sequence.seq'), ''))

    def choose_source(dialog):
        text = 'Choose folder…' if vendor == 'Bruker' else 'Choose Pulseq file…'
        next(button for button in dialog.findChildren(QtWidgets.QPushButton) if button.text() == text).click()
        return QtWidgets.QDialog.Accepted

    window._exec_dialog = choose_source
    selected = window._get_import_source('Import')
    assert selected == str(tmp_path if vendor == 'Bruker' else tmp_path / 'sequence.seq')
    window.close()
    qt_app.processEvents()
