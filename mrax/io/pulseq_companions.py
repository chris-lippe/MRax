"""Conservative metadata reconciliation for paired Pulseq measurement products."""
from __future__ import annotations

import json
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields, replace

import h5py
import numpy as np


@dataclass
class _RawCompanion:
    coordinates: dict
    norms: np.ndarray | None
    echo_times: list[float] | None
    fov_m: np.ndarray


def _same_definition(a, b):
    a = a.split() if isinstance(a, str) else a
    b = b.split() if isinstance(b, str) else b
    try:
        return np.shape(a) == np.shape(b) and np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-10, atol=1e-12)
    except (ValueError, TypeError):
        return np.array_equal(a, b)


def image_product_role(path):
    """Identify conventional image components, not quantitative acquisitions."""
    name = path.name.lower().removesuffix('.gz').removesuffix('.nii')
    if name.endswith('_phase'):
        return 'phase'
    if name.endswith('_map'):
        return 'parameter_map'
    return 'magnitude'


def acquisition_frequency(sequence_path, info):
    """Read a console frequency in Hz only when its sequence definitions agree."""
    frequencies = []
    for path in sequence_path.parent.glob('*.json'):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        sequence = document.get('sequence', {})
        acquisition = document.get('acquisition_parameter', {})
        if not isinstance(sequence, dict) or not isinstance(acquisition, dict):
            continue
        definitions = sequence.get('definitions', {})
        if not isinstance(definitions, dict) or not definitions:
            continue
        if any(key not in info.definitions or not _same_definition(value, info.definitions[key])
               for key, value in definitions.items()):
            continue
        frequency = acquisition.get('larmor_frequency')
        if isinstance(frequency, (int, float)) and np.isfinite(frequency) and frequency > 0:
            frequencies.append(float(frequency) / 1e6)
    if frequencies and np.allclose(frequencies, frequencies[0], rtol=1e-8):
        return frequencies[0]
    if frequencies:
        raise ValueError("Paired acquisition metadata declares conflicting resonance frequencies")
    for key in ('FREQ', 'freq', 'FieldStrengthMHz', 'B0', 'b0'):
        value = info.definitions.get(key)
        if value is not None:
            value = float(np.asarray(value).item())
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Sequence resonance frequency or field strength must be positive and finite")
            return value * 42.57747892 if key in ('B0', 'b0') else value
    return None


def sequence_frame_metadata(info, coordinates, frame_count, *, echo_times=None,
                            frame_indices=None, source_frame_count=None, contrast_count=None):
    """Expand an echo spacing only for explicitly separate echo contrasts."""
    meta = replace(info.normalized_metadata, frame_count=frame_count)
    contrasts = np.asarray(coordinates.get('contrast', []), int)
    contrast_count = contrast_count or len(np.unique(contrasts))
    selected = {}
    for item in fields(meta):
        value = getattr(meta, item.name)
        array = np.asarray(value)
        if value is None or array.ndim == 0 or len(array) <= 1:
            continue
        if (contrast_count > 1 and len(array) == contrast_count and len(contrasts) == frame_count
                and np.all((contrasts >= 0) & (contrasts < len(array)))):
            selected[item.name] = array[contrasts].tolist()
        elif frame_indices is not None and len(array) == source_frame_count:
            selected[item.name] = array[np.asarray(frame_indices, int)].tolist()
    meta = replace(meta, **selected)
    if not len(contrasts) or meta.echo_times is None or np.size(meta.echo_times) != 1:
        return meta
    if echo_times is not None and np.size(echo_times) == frame_count:
        return replace(meta, echo_times=np.asarray(echo_times).tolist())
    train = info.definitions.get('echo_train_length')
    if (train is not None and float(train).is_integer() and int(train) > 1
            and contrast_count == int(train) and np.all((contrasts >= 0) & (contrasts < int(train)))):
        return replace(meta, echo_times=(float(np.asarray(meta.echo_times).item()) * (contrasts + 1)).tolist())
    return meta


def _raw_companion(path, info, *, energies=False):
    """Use one unambiguous same-folder MRD counter table, never filename ordering."""
    from mrax.io.ismrmrd_measurement import _FRAME_FIELDS, _ROLES, _flag

    candidates = []
    for filename in sorted(path.parent.iterdir()):
        if filename.suffix.lower() not in ('.mrd', '.h5', '.hdf5'):
            continue
        try:
            with h5py.File(filename, 'r') as file:
                for group in file.values():
                    if not isinstance(group, h5py.Group) or 'data' not in group:
                        continue
                    records = group['data']
                    if not isinstance(records, h5py.Dataset) or not records.dtype.names or 'head' not in records.dtype.names:
                        continue
                    heads = records.fields('head')[:]
                    if (not len(heads) or len(heads) != len(info.adc_counts)
                            or not np.array_equal(heads['number_of_samples'], info.adc_counts)):
                        continue
                    if 'xml' not in group:
                        continue
                    xml = ET.fromstring(group['xml'][0])
                    for node in xml.iter():
                        node.tag = node.tag.rsplit('}', 1)[-1]
                    encoding = xml.find('encoding')
                    if encoding is None or encoding.findtext('trajectory') != 'cartesian':
                        continue
                    matrix = tuple(int(encoding.findtext('reconSpace/matrixSize/' + k, '1')) for k in 'xyz')
                    declared_shape = tuple(reversed(matrix if matrix[2] > 1 else matrix[:2]))
                    if declared_shape != info.spatial_shape:
                        continue
                    imaging = heads['number_of_samples'] > 0
                    if len(np.unique(heads['encoding_space_ref'])) != 1:
                        continue
                    for flag in _ROLES.values():
                        role = _flag(heads['flags'], flag)
                        if flag == 20:
                            role &= ~_flag(heads['flags'], 21)
                        imaging &= ~role
                    indices = np.flatnonzero(imaging)
                    keys, inverse = np.unique(np.column_stack([heads['idx'][k][indices] for k in _FRAME_FIELDS]), axis=0, return_inverse=True)
                    coordinates = {k: keys[:, i] for i, k in enumerate(_FRAME_FIELDS)}
                    norms = None
                    shape = info.spatial_shape
                    # Parseval identifies contrast order only for a complete grid.
                    if energies and shape and np.all(heads['number_of_samples'][indices] == shape[-1]) and not np.any(heads['discard_pre'][indices] + heads['discard_post'][indices]):
                        slots = np.column_stack([heads['idx'][k][indices] for k in ('kspace_encode_step_1', 'kspace_encode_step_2')])
                        if all(len(np.unique(slots[inverse == i], axis=0)) == np.prod(shape[:-1]) == np.sum(inverse == i) for i in range(len(keys))):
                            norms = np.zeros(len(keys))
                            for j, frame in zip(indices, inverse):
                                values = records[int(j)]['data'].view(np.complex64)
                                norms[frame] += np.sum(np.abs(values.astype(np.complex128)) ** 2)
                            norms = np.sqrt(norms)
                    echo_times = None
                    if info.adc_echo_times_ms is not None and len(info.adc_echo_times_ms) == len(heads):
                        times = np.asarray(info.adc_echo_times_ms)[indices]
                        grouped = [times[inverse == i] for i in range(len(keys))]
                        if all(len(v) and np.allclose(v, v[0], atol=.01, rtol=1e-5) for v in grouped):
                            echo_times = [float(np.median(v)) for v in grouped]
                    fov = np.asarray([float(encoding.findtext('reconSpace/fieldOfView_mm/' + k)) / 1000 for k in 'xyz'])
                    candidates.append(_RawCompanion(coordinates, norms, echo_times, fov))
        except (OSError, ValueError, KeyError, ET.ParseError):
            continue
    return candidates[0] if len(candidates) == 1 else None


def nifti_options(path, info, options):
    """Resolve compact 2D contrast stacks with independent acquisition evidence.

    Standard NIfTI volume/time axes are retained. Nonstandard compact stacks need
    matching sequence dimensions and counters, or explicit reader options.
    """
    import nibabel as nib

    image = nib.load(path)
    options = dict(options)
    matrix = info.definitions.get('encoding_dim', info.definitions.get('Matrix'))
    companion = None
    compact = False
    if matrix is not None:
        matrix = np.asarray(matrix, int).reshape(-1)
        if (len(matrix) >= 2 and (len(matrix) < 3 or matrix[2] == 1)
                and len(image.shape) == 3 and tuple(matrix[:2]) == image.shape[:2] and image.shape[2] > 1):
            companion = _raw_companion(path, info, energies=True)
            if companion is not None and len(next(iter(companion.coordinates.values()))) == image.shape[2]:
                options.setdefault('spatial_ndim', 2)
                compact = options['spatial_ndim'] == 2
            elif 'spatial_ndim' not in options:
                raise ValueError("NIfTI third axis conflicts with the declared 2D encoding; specify spatial_ndim and contrast ordering explicitly")
    if compact and 'frame_order' not in options:
        coordinates, norms = companion.coordinates, companion.norms
        data = np.asanyarray(image.dataobj)
        if path.name.lower().endswith(('_phase.nii', '_phase.nii.gz')):
            prefix = path.name[:-len('_phase.nii.gz')] if path.name.lower().endswith('.nii.gz') else path.name[:-len('_phase.nii')]
            magnitude = [p for p in path.parent.iterdir() if p.name in
                (prefix + '_mag.nii', prefix + '_mag.nii.gz', prefix + '_magnitude.nii', prefix + '_magnitude.nii.gz')]
            if len(magnitude) == 1 and nib.load(magnitude[0]).shape == image.shape:
                data = np.asanyarray(nib.load(magnitude[0]).dataobj)
        image_norms = np.linalg.norm(data.reshape((-1, data.shape[-1])), axis=0)
        if norms is None or np.any(norms <= 0) or np.any(image_norms <= 0):
            raise ValueError("Compact NIfTI contrast order requires frame_order or a complete Cartesian raw companion")
        from scipy.optimize import linear_sum_assignment

        source, target = np.log(image_norms), np.log(norms)
        source -= source.mean()
        target -= target.mean()
        rows, columns = linear_sum_assignment(np.abs(target[:, None] - source[None]))
        if (np.max(np.abs(target[rows] - source[columns])) > 1e-4
                or np.min(np.diff(np.sort(target)), initial=np.inf) < 1e-4):
            raise ValueError("Compact NIfTI contrast order is ambiguous; supply frame_order explicitly")
        options['frame_order'] = columns.tolist()
    if image.header.get_xyzt_units()[0] == 'unknown' and 'spatial_unit' not in options:
        fov = companion.fov_m if compact else np.asarray(info.definitions.get('FOV', []), float)
        spatial_ndim = options.get('spatial_ndim', min(len(image.shape), 3))
        if len(fov) >= spatial_ndim:
            extent = np.linalg.norm(image.affine[:3, :spatial_ndim], axis=0) * image.shape[:spatial_ndim]
            if np.allclose(extent, fov[:spatial_ndim], rtol=1e-4):
                options['spatial_unit'] = 'meter'
                warnings.warn("NIfTI spatial units inferred as metres from matching sequence FOV", RuntimeWarning, stacklevel=3)
    coordinates = companion.coordinates if compact else {}
    return options, coordinates, companion.echo_times if compact else None


def measurement_overview(path, info, format, options):
    """Return normalized timing and frame count without reconstructing samples."""
    from mrax.io.ismrmrd_measurement import _FRAME_FIELDS, _ROLES, _flag

    coordinates = {}
    frame_count = 1
    if format == 'ismrmrd' and options.get('image_series') is None:
        with h5py.File(path, 'r') as file:
            heads = file[options['group']]['data'].fields('head')[:]
            keep = (heads['number_of_samples'] > 0) & (heads['encoding_space_ref'] == options.get('encoding_ref', 0))
            for flag in _ROLES.values():
                role = _flag(heads['flags'], flag)
                if flag == 20:
                    role &= ~_flag(heads['flags'], 21)
                keep &= ~role
            heads = heads[keep]
            keys = np.unique(np.column_stack([heads['idx'][k] for k in _FRAME_FIELDS]), axis=0)
            coordinates = {k: keys[:, i] for i, k in enumerate(_FRAME_FIELDS)}
            frame_count = len(keys)
    elif format == 'nifti':
        import nibabel as nib

        shape = nib.load(path).shape
        companion = _raw_companion(path, info)
        matrix = info.definitions.get('encoding_dim', info.definitions.get('Matrix'))
        if (matrix is not None and len(shape) == 3 and len(matrix) >= 2 and (len(matrix) == 2 or matrix[2] == 1)
                and tuple(matrix[:2]) == shape[:2] and companion is not None
                and len(next(iter(companion.coordinates.values()))) == shape[2]):
            coordinates = companion.coordinates
            frame_count = shape[2]
        else:
            frame_count = int(np.prod(shape[3:]))
    return sequence_frame_metadata(info, coordinates, frame_count), frame_count
