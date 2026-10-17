"""Pulseq measurement discovery and integration with MRax's shared data model."""
from __future__ import annotations

import warnings
from dataclasses import fields, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from mrax.data_struct import Study
from mrax.devices import device_function
from mrax.io.import_definitions import (
    NormalizedMetadata,
)
from mrax.io.measurement import (
    MeasurementData,
    build_measurement_experiment,
    measurement_entries,
    read_measurement,
    register_measurement_reader,
)


def _read_mrd(path, **options):
    from mrax.io.ismrmrd_measurement import read_ismrmrd_measurement

    return read_ismrmrd_measurement(path, **options)


def _read_twix(path, **options):
    from mrax.io.siemens_measurement import read_siemens_measurement

    return read_siemens_measurement(path, **options)


def _inspect_mrd(path):
    from mrax.io.ismrmrd_measurement import inspect_ismrmrd_measurement

    return inspect_ismrmrd_measurement(path)


register_measurement_reader('ismrmrd', ('.h5', '.hdf5', '.mrd'), _read_mrd, inspector=_inspect_mrd)
register_measurement_reader('siemens', ('.dat',), _read_twix)


def discover_pulseq_measurements(root: str | Path, *, recursive=True):
    """Discover extracted measurement files; archives and analysis files are ignored.

    MATLAB variables have no standard meaning and require an explicit reader call.
    DICOM series use MRax's existing ``discover_dicom_series/load_dicom_study``.
    """
    root = Path(root)
    paths = [root] if root.is_file() else (root.rglob('*') if recursive else root.iterdir())
    candidates = sorted(p for p in paths if p.is_file() and
        any(p.name.lower().endswith(s) for s in ('.h5', '.hdf5', '.mrd', '.dat', '.nii', '.nii.gz')))
    result = []
    for path in candidates:
        if path.suffix.lower() in ('.h5', '.hdf5'):
            try:
                _inspect_mrd(path)
            except ValueError:
                # Extracted consoles also contain auxiliary, non-MRD HDF5 files.
                continue
            except OSError:
                pass  # Corrupted candidate files are reported during import.
        result.append(path)
    return result


def _resolve_sequence(path, sequence_path, sidecar):
    if sequence_path is not None:
        return Path(sequence_path)
    if sidecar.get('sequence_path'):
        return path.parent / str(sidecar['sequence_path'])
    exact = path.with_name(path.name[:-7] + '.seq') if path.name.lower().endswith('.nii.gz') else path.with_suffix('.seq')
    if exact.exists():
        return exact
    # Filenames with a vendor prefix are deliberately not matched by substring.
    return None


def read_pulseq_measurement(path: str | Path, *, sequence_path=None, format=None,
                            reader_options: Mapping | None = None,
                            metadata_overrides: Mapping | None = None,
                            definition_paths=()) -> MeasurementData:
    """Read extracted pixels/raw data and optional Pulseq sequence metadata.

    MRD/NIfTI images and MRD encoded samples can be read without a descriptor.
    Siemens streams require a matching descriptor. Explicit overrides use the
    normalized schema: timing in ms, flip angles in degrees.
    """
    from mrax.io.pulseq_import import read_pulseq, read_pulseq_sidecar_metadata

    path = Path(path)
    sidecar = read_pulseq_sidecar_metadata(path)
    sequence_path = _resolve_sequence(path, sequence_path, sidecar)
    options = {**dict(sidecar.get('reader_options', {})), **dict(reader_options or {})}
    if format is None:
        format = sidecar.get('format')
    info = read_pulseq(sequence_path, definition_paths=definition_paths,
        definition_overrides=sidecar.get("sequence_definitions"),
        calculate_trajectory=(format in ('siemens', 'ismrmrd') or path.suffix.lower() in ('.dat', '.h5', '.hdf5', '.mrd'))
            and options.get('image_series') is None) if sequence_path is not None else None
    if info is not None and (format in ('siemens', 'ismrmrd') or path.suffix.lower() in ('.dat', '.h5', '.hdf5', '.mrd')):
        options['sequence_info'] = info
    coordinates = {}
    echo_times = None
    if info is not None and (format == 'nifti' or path.name.lower().endswith(('.nii', '.nii.gz'))):
        from mrax.io.pulseq_companions import nifti_options

        if info.definitions.get('echo_train_length', 1) > 1 and info.adc_counts:
            info = read_pulseq(sequence_path, definition_paths=definition_paths,
                definition_overrides=sidecar.get('sequence_definitions'))
        options, coordinates, echo_times = nifti_options(path, info, options)
    result = read_measurement(path, format=format, **options)
    if result.metadata.get('format') == 'nifti':
        from mrax.io.pulseq_companions import image_product_role

        result = replace(result, metadata={**result.metadata,
            'image_component': sidecar.get('image_component', image_product_role(path))})
    combined = {f.name: getattr(result.normalized_metadata, f.name) for f in fields(NormalizedMetadata)}
    if info is not None:
        from mrax.io.pulseq_companions import acquisition_frequency, sequence_frame_metadata

        contrast_count = len(np.unique(coordinates.get('contrast', []))) or result.metadata.get('source_contrast_count')
        if coordinates and options.get('frame_selection') is not None:
            coordinates = {key: np.atleast_1d(value[options['frame_selection']]) for key, value in coordinates.items()}
            if echo_times is not None:
                echo_times = np.atleast_1d(np.asarray(echo_times)[options['frame_selection']])
        coordinates = coordinates or result.metadata.get('frame_coordinates', {})
        normalized = sequence_frame_metadata(info, coordinates, result.normalized_metadata.frame_count,
            echo_times=echo_times if echo_times is not None else result.metadata.get('executed_echo_times_ms'),
            frame_indices=result.metadata.get('frame_indices'), source_frame_count=result.metadata.get('source_frame_count'),
            contrast_count=contrast_count)
        if result.metadata.get('image_component') == 'parameter_map':
            normalized = replace(result.normalized_metadata, method_name=normalized.method_name,
                sequence_name=normalized.sequence_name, protocol_name=normalized.protocol_name)
        if (np.size(normalized.echo_times) == 1 and result.normalized_metadata.echo_times is not None
                and np.size(result.normalized_metadata.echo_times) > 1):
            normalized = replace(normalized, echo_times=result.normalized_metadata.echo_times)
        # Executed sequence parameters take precedence over scanner placeholders.
        for f in fields(NormalizedMetadata):
            value = getattr(normalized, f.name)
            if f.name != 'frame_count' and value is not None and not (isinstance(value, (str, tuple)) and not value):
                combined[f.name] = value
        frequency = acquisition_frequency(sequence_path, info)
        extra = dict(result.metadata)
        if frequency is not None:
            recorded = extra.get('field_strength_mhz')
            if recorded and not np.isclose(recorded, frequency, rtol=1e-4):
                warnings.warn("MRD resonance frequency disagrees with paired acquisition metadata; using the acquisition frequency in Hz", RuntimeWarning, stacklevel=2)
            extra['field_strength_mhz'] = frequency
        if coordinates:
            extra['frame_coordinates'] = coordinates
        result = replace(result, metadata=extra)
    overrides = {**dict(sidecar.get('normalized_metadata', {})), **dict(metadata_overrides or {})}
    unknown = set(overrides) - set(combined)
    if unknown:
        raise ValueError(f"Unknown normalized metadata fields: {', '.join(sorted(unknown))}")
    combined.update(overrides)
    return replace(result, normalized_metadata=NormalizedMetadata(**combined),
        metadata={**result.metadata, 'sequence_path': None if sequence_path is None else str(sequence_path)})



@device_function
def import_pulseq_study(root: str | Path, *, sequence_map=None, image_data=None,
                        reader_options=None, type_overrides=None, metadata_overrides=None,
                        study_globals=None, device='cpu', progress_callback=None, include_dicom=True,
                        definition_paths=(), dicom_series_groups=None):
    """Import extracted images; return raw-only entries for explicit reconstruction.

    Per-file mappings use paths relative to ``root``. Invalid files warn and do
    not prevent other experiments importing. Raw-only entries remain available
    as ``metadata['raw_measurements']`` and are never fitted as dummy images.
    """
    root = Path(root)
    entries, raw, issues = [], {}, []
    combined_globals = dict(study_globals or {})
    paths = discover_pulseq_measurements(root)
    for i, path in enumerate(paths):
        source_key = path.name if root.is_file() else path.relative_to(root).as_posix()
        if progress_callback:
            progress_callback(i, len(paths), 'Reading measurement')
        try:
            variants = measurement_entries(path)
        except (OSError, ValueError) as exc:
            warnings.warn(f"Skipped measurement {source_key}: {exc}", RuntimeWarning, stacklevel=2)
            issues.append({'path': source_key, 'message': str(exc), 'error_type': type(exc).__name__})
            continue
        for variant in variants:
            options = variant['reader_options']
            suffix = '/'.join(str(options[k]) for k in ('group', 'image_series') if k in options)
            key = source_key + '::' + suffix if len(variants) > 1 else source_key
            try:
                sequence = (sequence_map or {}).get(key, (sequence_map or {}).get(source_key))
                if sequence is not None and not Path(sequence).is_absolute():
                    sequence = (root if root.is_dir() else root.parent) / sequence
                options = {**options, **(reader_options or {}).get(source_key, {}), **(reader_options or {}).get(key, {})}
                measurement = read_pulseq_measurement(path, sequence_path=sequence,
                    reader_options=options, metadata_overrides={**(metadata_overrides or {}).get(source_key, {}),
                        **(metadata_overrides or {}).get(key, {})}, definition_paths=definition_paths)
                if measurement.metadata.get('field_strength_mhz') is not None:
                    from mrax.data_struct import Quantity

                    combined_globals.setdefault('field_strength_mhz', Quantity(measurement.metadata['field_strength_mhz'], 'MHz'))
                pixels = (image_data or {}).get(key, (image_data or {}).get(source_key, measurement.image_data))
                if pixels is None:
                    raw[key] = measurement
                    continue
                entry = build_measurement_experiment(measurement, image_data=pixels,
                    name=key, override_type=(type_overrides or {}).get(key, (type_overrides or {}).get(source_key)), device=device)
                entries.append((key, entry))
            except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                warnings.warn(f"Skipped measurement {key}: {exc}", RuntimeWarning, stacklevel=2)
                issues.append({'path': key, 'message': str(exc), 'error_type': type(exc).__name__})
    dicom_paths = [root] if root.is_file() and root.suffix.lower() in ('.dcm', '.ima') else sorted(
        p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.dcm', '.ima')
        and '__MACOSX' not in p.parts and not p.name.startswith('._')) if root.is_dir() else []
    if include_dicom and dicom_paths:
        from mrax.io.dicom import load_dicom_study

        try:
            dicom_overrides = {key.removeprefix('dicom:'): dict(value)
                for key, value in (metadata_overrides or {}).items() if key.startswith('dicom:')}
            for key, sequence in (sequence_map or {}).items():
                if key.startswith('dicom:'):
                    from mrax.io.pulseq_import import read_pulseq

                    sequence = Path(sequence)
                    if not sequence.is_absolute():
                        sequence = (root if root.is_dir() else root.parent) / sequence
                    normalized = read_pulseq(sequence, definition_paths=definition_paths,
                        calculate_trajectory=False).normalized_metadata
                    values = {f.name: getattr(normalized, f.name) for f in fields(NormalizedMetadata)
                        if f.name != 'frame_count' and getattr(normalized, f.name) is not None
                        and getattr(normalized, f.name) != () and getattr(normalized, f.name) != ''}
                    name = key.removeprefix('dicom:')
                    dicom_overrides[name] = {**values, **dicom_overrides.get(name, {})}
            dicom_study, dicom_metadata = load_dicom_study(dicom_paths, series_groups=dicom_series_groups,
                type_overrides={k.removeprefix('dicom:'): v for k, v in (type_overrides or {}).items()},
                metadata_overrides=dicom_overrides)
            for name, value in dicom_study.study_globals.items():
                combined_globals.setdefault(name, value)
            for name, experiment, model, dataset in dicom_study.iter_entries():
                key = 'dicom:' + name
                entries.append((key, (experiment, dataset, model, dicom_metadata)))
        except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
            warnings.warn(f"Skipped DICOM input: {exc}", RuntimeWarning, stacklevel=2)
            issues.append({'format': 'dicom', 'message': str(exc), 'error_type': type(exc).__name__})
    study = Study(experiments=[e[0] for _, e in entries], datasets=[e[1] for _, e in entries],
        models=[e[2] for _, e in entries], names=[k for k, _ in entries], study_globals=combined_globals)
    if progress_callback:
        progress_callback(len(paths), len(paths), 'Import complete')
    return study, {'experiments': {k: e[3] for k, e in entries}, 'raw_measurements': raw, 'import_issues': issues}
