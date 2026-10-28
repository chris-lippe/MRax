"""Pulseq source selection and GUI imports using the shared measurement readers."""
from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from mrax.data_struct import Quantity, Study
from mrax.gui.models import make_model, normalize_model_key
from mrax.io.import_definitions import NormalizedMetadata, detect_experiment_type
from mrax.io.measurement import (
    build_measurement_experiment,
    image_signal_scale,
    measurement_entries,
)
from mrax.io.pulseq_import import read_pulseq, read_pulseq_sidecar_metadata
from mrax.io.pulseq_measurement import discover_pulseq_measurements, read_pulseq_measurement
from mrax.library.default_experiments import default_model_for_type


@dataclass(frozen=True)
class PulseqImportPlan:
    root: Path
    sequences: tuple[Path, ...]
    entries: tuple[dict, ...]
    issues: tuple[dict, ...] = ()

    @property
    def summary(self):
        return [dict(entry['summary']) for entry in self.entries]


def _sequence_values(info):
    return {item.name: value for item in fields(NormalizedMetadata)
        if item.name != 'frame_count' and (value := getattr(info.normalized_metadata, item.name)) is not None
        and not (isinstance(value, (str, tuple)) and not value)}


def plan_pulseq_import(source, measurement_source=None):
    """Inspect extracted neighbours and require unambiguous sequence pairing."""
    source = Path(source)
    root = source if source.is_dir() else source.parent
    sequences = tuple(sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.seq')) if source.is_dir() else (source,)
    if not sequences or any(not p.is_file() or p.suffix.lower() != '.seq' for p in sequences):
        raise ValueError('Select a Pulseq .seq file or a folder containing one')
    infos = {p.resolve(): read_pulseq(p, calculate_trajectory=False) for p in sequences}
    entries, issues = [], []
    folders = sorted({p.parent for p in sequences})
    locations = [Path(measurement_source)] if measurement_source is not None else folders
    candidates = sorted({p for location in locations for p in discover_pulseq_measurements(location, recursive=measurement_source is not None)})

    def paired(path):
        sidecar = read_pulseq_sidecar_metadata(path)
        if sidecar.get('sequence_path'):
            match = (path.parent / sidecar['sequence_path']).resolve()
            return match if match in infos else None
        exact = (path.with_name(path.name[:-7] + '.seq') if path.name.lower().endswith('.nii.gz') else path.with_suffix('.seq')).resolve()
        if exact in infos:
            return exact
        neighbours = [p for p in infos if p.parent == path.parent.resolve()]
        if len(neighbours) == 1:
            # Require a unique sequence before pairing same-folder products.
            other = [p for p in path.parent.iterdir() if p.suffix.lower() == '.seq']
            if len(other) == 1:
                return neighbours[0]
        return next(iter(infos)) if measurement_source is not None and len(infos) == 1 else None

    def add(path, sequence, format, reader_options=None, series_uid=None, frequency=None, frame_count=''):
        info = infos[sequence]
        from mrax.io.pulseq_companions import (
            acquisition_frequency,
            image_product_role,
            measurement_overview,
        )

        normalized = info.normalized_metadata
        if format in ('nifti', 'ismrmrd'):
            normalized, frame_count = measurement_overview(path, info, format, reader_options or {})
        relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
        suffix = series_uid or '/'.join(str(v) for v in (reader_options or {}).values())
        base = re.sub(r'[^\w.-]+', '_', relative + ('_' + suffix if suffix else ''))[:100].strip('._') or 'measurement'
        key = base
        number = 2
        while any(entry['summary']['scan'] == key for entry in entries):
            key = f'{base}_{number}'
            number += 1
        kind = detect_experiment_type(normalized)
        if format == 'nifti' and image_product_role(path) in ('phase', 'parameter_map'):
            kind = 'anatomical'
        name = info.normalized_metadata.method_name
        if frequency is None:
            frequency = acquisition_frequency(sequence, info)
        if frequency is None:
            frequency = next((info.definitions[k] for k in ('FREQ', 'freq', 'FieldStrengthMHz') if k in info.definitions), None)
        if frequency is None:
            tesla = next((info.definitions[k] for k in ('B0', 'b0') if k in info.definitions), None)
            frequency = float(tesla) * 42.57747892 if tesla is not None else None
        entries.append({'path': path, 'sequence': sequence, 'format': format,
            'reader_options': dict(reader_options or {}), 'series_uid': series_uid,
            'summary': {'scan': key, 'method': name, 'protocol': path.name,
                'frame_count': frame_count, 'detected_type': kind,
                'detected_model': 'wasabi' if 'wasabi' in name.lower() else normalize_model_key(kind) or 'identity',
                'field_strength_mhz': frequency}})

    for path in candidates:
        sequence = paired(path)
        if sequence is None:
            issues.append({'path': str(path), 'message': 'No unambiguous Pulseq sequence pairing'})
            continue
        try:
            for variant in measurement_entries(path):
                add(path, sequence, variant['format'], variant['reader_options'])
        except (OSError, ValueError, ImportError) as exc:
            issues.append({'path': str(path), 'message': str(exc)})
    from mrax.io.dicom import discover_dicom_series

    for location in locations:
        files = [location] if location.is_file() and location.suffix.lower() in ('.dcm', '.dicom', '.ima') else sorted(
            p for p in (location.rglob('*') if measurement_source is not None else location.iterdir())
            if p.is_file() and p.suffix.lower() in ('.dcm', '.dicom', '.ima') and not p.name.startswith('._')) if location.is_dir() else []
        by_sequence = {}
        for path in files:
            sequence = paired(path)
            if sequence is not None:
                by_sequence.setdefault(sequence, []).append(path)
        for sequence, paths in by_sequence.items():
            from pydicom import dcmread
            from pydicom.errors import InvalidDicomError

            grouped_headers = defaultdict(list)
            for path in paths:
                try:
                    header = dcmread(path, stop_before_pixels=True, specific_tags=[
                        'SeriesInstanceUID', 'MagneticFieldStrength', 'ImagePositionPatient', 'NumberOfFrames'])
                except (OSError, ValueError, InvalidDicomError) as exc:
                    issues.append({'path': str(path), 'message': str(exc)})
                    continue
                grouped_headers[str(header.get('SeriesInstanceUID', ''))].append((path, header))
            for series in discover_dicom_series(paths):
                headers = grouped_headers[series.series_instance_uid]
                strength = headers[0][1].get('MagneticFieldStrength')
                frequency = float(strength) * 42.57747892 if strength else None
                positions = [tuple(header.get('ImagePositionPatient', ())) for _, header in headers]
                frame_count = ''
                if all(positions) and all(int(header.get('NumberOfFrames', 1)) == 1 for _, header in headers):
                    counts = list(Counter(positions).values())
                    if len(set(counts)) == 1:
                        frame_count = counts[0]
                add(location, sequence, 'dicom', series_uid=series.series_instance_uid,
                    frequency=frequency, frame_count=frame_count)
                entries[-1]['dicom_files'] = [path for path, _ in headers]
    return PulseqImportPlan(root, sequences, tuple(entries), tuple(issues))


def load_pulseq_import_plan(plan, options, progress=None):
    """Materialize selected images and reconstruct raw-only recordings generically."""
    from mrax.io.dicom import load_dicom_series
    from mrax.reco import reconstruct_acquisition

    experiments, datasets, models, names = [], [], [], []
    metadata = {'experiments': {}, 'import_issues': list(plan.issues)}
    selected = options.get('selected_scans')
    entries = [entry for entry in plan.entries if selected is None or entry['summary']['scan'] in selected]
    frequency = options.get('field_strength_mhz')
    for index, entry in enumerate(entries):
        key = entry['summary']['scan']
        if progress:
            progress(index, len(entries), 'Reading ' + entry['path'].name)
        try:
            override = (options.get('type_overrides') or {}).get(key)
            override = {'t1_ir': 'ir', 't1_vtr': 'vtr', 't1_vfa': 'vfa', 't2_se': 'mse', 't2star_se': 'mge'}.get(override, override)
            if entry['format'] == 'dicom':
                info = read_pulseq(entry['sequence'], calculate_trajectory=False)
                experiment, dataset, extra = load_dicom_series(entry['dicom_files'], series_instance_uid=entry['series_uid'],
                    name=key, override_type=override, metadata_overrides=_sequence_values(info))
                model = default_model_for_type(experiment.exp_type, scale=image_signal_scale(dataset.data),
                    observables=experiment.observables, image_data=dataset.data)
                detected_frequency = extra.get('magnetic_field_strength_t')
                detected_frequency = float(detected_frequency) * 42.57747892 if detected_frequency else None
            else:
                measured = read_pulseq_measurement(entry['path'], sequence_path=entry['sequence'],
                    format=entry['format'], reader_options=entry['reader_options'])
                image = measured.image_data
                if image is None:
                    if not options.get('load_raw_kspace', True):
                        raise ValueError('Raw-only measurement requires reconstruction; enable raw k-space import')
                    if progress:
                        progress(index, len(entries), 'Reconstructing ' + entry['path'].name)
                    image = np.abs(reconstruct_acquisition(measured.acquisition, method='fft', device='cpu').image)
                experiment, dataset, model, extra = build_measurement_experiment(measured,
                    image_data=image, name=key, override_type=override)
                detected_frequency = measured.metadata.get('field_strength_mhz')
            model_key = (options.get('model_overrides_by_name') or {}).get(key)
            if model_key and model_key != normalize_model_key(model.name):
                model = make_model(model_key)
            entry_frequency = frequency or detected_frequency or entry['summary']['field_strength_mhz']
            if entry_frequency:
                experiment.observables['field_strength_mhz'] = Quantity(float(entry_frequency), 'MHz')
            experiments.append(experiment)
            datasets.append(dataset)
            models.append(model)
            names.append(key)
            metadata['experiments'][key] = {**extra, 'sequence_path': str(entry['sequence'])}
        except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
            metadata['import_issues'].append({'path': str(entry['path']), 'message': str(exc)})
            warnings.warn(f'Skipped Pulseq measurement {key}: {exc}', RuntimeWarning, stacklevel=2)
    if not names:
        raise ValueError('No selected Pulseq measurements could be imported; check their metadata and reconstruction requirements')
    if not frequency:
        frequency = next((experiment.observables['field_strength_mhz'].values for experiment in experiments
            if 'field_strength_mhz' in experiment.observables), None)
    globals_map = {'field_strength_mhz': Quantity(float(frequency), 'MHz')} if frequency else {}
    study = Study(experiments, datasets, models, names=names, study_globals=globals_map)
    if progress:
        progress(len(entries), len(entries), 'Pulseq import complete')
    return study, metadata
