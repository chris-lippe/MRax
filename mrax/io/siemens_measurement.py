"""Siemens container adapter for measured Pulseq ADC streams."""
from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import replace

import numpy as np

from mrax.io.measurement import MeasurementData
from mrax.io.pulseq_samples import prepare_pulseq_samples


def read_siemens_measurement(path, *, sequence_info=None, scan_index=None, spatial_shape=None,
                             load_auxiliary=True, frame_selection=None, axis_order=None, density="auto", gradient_directions=None):
    """Read the selected measurement in acquisition order, retaining oversampling.

    A matching Pulseq descriptor supplies the encoding executed by the external
    interpreter; scanner protocol placeholders remain auxiliary metadata.
    """
    try:
        from mapvbvd import mapVBVD
    except ImportError as exc:
        raise ImportError("Siemens measurement import requires pymapvbvd") from exc
    if sequence_info is None:
        raise ValueError("Siemens Pulseq measurements require a matching sequence_path")
    # The reader prints progress even in quiet mode; keep library imports quiet.
    with contextlib.redirect_stdout(io.StringIO()):
        scans = mapVBVD(str(path), quiet=True, disableReflect=True, ignoreROoffcenter=True, removeOS=False, regrid=False)
    scans = scans if isinstance(scans, list) else [scans]
    candidates = [i for i, scan in enumerate(scans) if 'image' in scan]
    if scan_index is None:
        if len(candidates) != 1:
            raise ValueError("Choose scan_index explicitly when multiple image measurements exist")
        scan_index = candidates[0]
    scan = scans[scan_index]
    if 'image' not in scan:
        raise ValueError("Selected Siemens measurement contains no imaging ADC samples")
    image = scan['image']
    image.flagRemoveOS = False
    image.flagRampSampRegrid = False
    image.flagDoAverage = False
    frame_fields = ('Ave', 'Eco', 'Phs', 'Rep', 'Set', 'Sli')
    columns = int(image.NCol)
    counts = np.asarray(sequence_info.adc_counts)
    adc_counters = {}
    if counts.size and int(np.sum(counts)) == columns * int(image.NAcq) and len(counts) != int(image.NAcq):
        # Storage chunks inside a continuous ADC belong to the same readout.
        event = np.searchsorted(np.cumsum(counts), np.arange(int(image.NAcq)) * columns, side='right')
        retained = []
        starts = np.r_[0, np.flatnonzero(np.diff(event)) + 1]
        for key in frame_fields:
            values = np.asarray(getattr(image, key))
            if np.any(np.maximum.reduceat(values, starts) != np.minimum.reduceat(values, starts)):
                adc_counters[key] = values
            else:
                retained.append(key)
        frame_fields = tuple(retained)
    keys = np.column_stack([getattr(image, key) for key in frame_fields]) if frame_fields else np.zeros((int(image.NAcq), 1))
    frame_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    selected_frames = np.arange(len(frame_keys)) if frame_selection is None else np.atleast_1d(np.arange(len(frame_keys))[frame_selection])
    if not len(selected_frames):
        raise ValueError("Frame selection is empty")
    readout_indices = np.flatnonzero(np.isin(inverse, selected_frames))
    columns = int(image.NCol)
    info = sequence_info
    counts = np.asarray(info.adc_counts)
    expected = sum(info.adc_counts)
    counters = None
    if expected == columns * int(image.NAcq):
        selected_events = np.arange(int(image.NAcq))
    else:
        # Feedback/VOP records describe monitoring rather than emitted ADC events.
        counter_sets = [('image', 'phasecor', 'refscan'), ('image', 'phasecor', 'refscan', 'noise')]
        for roles in counter_sets:
            candidate = np.unique(np.concatenate([scan[role].scancounter for role in roles if role in scan]))
            if len(candidate) == len(counts):
                counters = candidate
                break
        if counters is None or len(set(counts)) != 1 or int(counts[0]) != columns:
            raise ValueError("Siemens imaging ADCs cannot be matched uniquely to the sequence events")
        selected_events = np.searchsorted(counters, image.scancounter)
        warnings.warn("Non-imaging sequence ADC events were separated using scanner scan counters", RuntimeWarning, stacklevel=2)
    # Segmented ADC events can have a different record count but the same serial sample stream.
    selected = (selected_events[readout_indices, None] * columns + np.arange(columns)).reshape(-1)
    if len(selected) != expected or frame_selection is not None:
        info = replace(info, trajectory_physical=None if info.trajectory_physical is None else info.trajectory_physical[selected],
            trajectory=None if info.trajectory is None else info.trajectory[selected], adc_counts=tuple(columns for _ in readout_indices))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = np.asarray(image.readData(image.memPos[readout_indices], cIxToRaw=readout_indices))
    serial = raw.transpose(1, 2, 0)
    auxiliary = {}
    if load_auxiliary:
        for role in scan:
            if role not in ('hdr', 'image') and hasattr(scan[role], 'unsorted'):
                obj = scan[role]
                obj.flagRemoveOS = False
                obj.flagRampSampRegrid = False
                with contextlib.redirect_stdout(io.StringIO()):
                    auxiliary[{'phasecor': 'phase_correction', 'refscan': 'calibration'}.get(role, role)] = np.asarray(obj.unsorted()).transpose(2, 1, 0)
    selected_inverse = inverse[readout_indices]
    selected_keys, selected_inverse = np.unique(keys[readout_indices], axis=0, return_inverse=True)
    frame_indices = np.repeat(selected_inverse, columns) if len(selected_keys) > 1 else None
    if axis_order is None and info.trajectory_physical is not None and len(info.adc_counts) == len(readout_indices) and len(set(info.adc_counts)) == 1:
        positions = np.asarray(info.trajectory_physical).reshape((len(readout_indices), columns, -1))
        spread = np.ptp(positions, axis=1)
        read_axis = int(np.argmax(np.median(spread, axis=0)))
        remaining = [i for i in range(positions.shape[-1]) if i != read_axis]
        line_counter = np.asarray(image.Lin)[readout_indices]
        if np.std(line_counter) > 0:
            correlations = {i: abs(np.corrcoef(positions[:, 0, i], line_counter)[0, 1]) if np.std(positions[:, 0, i]) > 0 else 0. for i in remaining}
            phase_axis = max(correlations, key=correlations.get)
            if correlations[phase_axis] > .99:
                axis_order = [read_axis, phase_axis] + [i for i in remaining if i != phase_axis]
    acquisition = prepare_pulseq_samples(info, serial, spatial_shape=spatial_shape,
        auxiliary=auxiliary, frame_indices=frame_indices, axis_order=axis_order, density=density)
    if ('phasecor' in scan and acquisition.kind == 'cartesian' and acquisition.readout_polarity is not None and
            np.any(acquisition.readout_polarity < 0) and np.any(acquisition.readout_polarity > 0)):
        acquisition = replace(acquisition, recipe=replace(acquisition.recipe, epi_phase_correction='entropy'))
    names = dict(zip(('Ave', 'Eco', 'Phs', 'Rep', 'Set', 'Sli'), ('average', 'contrast', 'phase', 'repetition', 'set', 'slice')))
    if len(selected_keys) == acquisition.frame_count:
        coordinates = {names[key]: selected_keys[:, i].astype(int) for i, key in enumerate(frame_fields)}
        if sequence_info.adc_times is not None:
            clock = np.asarray(sequence_info.adc_times, float)
            profile_times = clock[selected].reshape((len(readout_indices), columns)).mean(axis=1) - clock[0]
            coordinates['time'] = np.bincount(selected_inverse, weights=profile_times) / np.bincount(selected_inverse)
        acquisition = replace(acquisition, frame_coordinates=coordinates)
    if load_auxiliary and counters is not None and 'refscan' in scan:
        calibration = _calibration(scan['refscan'], sequence_info, auxiliary.get('calibration'), counters, axis_order, acquisition)
        if calibration is not None:
            acquisition = replace(acquisition, auxiliary_samples={**acquisition.auxiliary_samples, 'calibration_grid': calibration})
    if load_auxiliary and counters is not None and 'phasecor' in scan and acquisition.recipe.epi_phase_correction == 'entropy':
        navigators = _group_epi_navigators(scan['phasecor'], auxiliary.get('phase_correction'), sequence_info,
            counters, keys, selected_keys, frame_fields)
        if navigators is not None:
            acquisition = replace(acquisition, auxiliary_samples={**acquisition.auxiliary_samples, **navigators[0]},
                auxiliary_coordinates={**acquisition.auxiliary_coordinates, **navigators[1]},
                auxiliary_frame_axes={**acquisition.auxiliary_frame_axes, **{key: 0 for key in (*navigators[0], *navigators[1])}},
                recipe=replace(acquisition.recipe, epi_phase_correction='navigator'))
    acquisition = _attach_geometry(acquisition, image, readout_indices, selected_inverse, info, axis_order, gradient_directions)
    return MeasurementData(acquisition=acquisition, spatial_geometry=acquisition.geometry, normalized_metadata=info.normalized_metadata,
        metadata={'format': 'siemens', 'path': str(path), 'scan_index': scan_index,
                  'sequence_adc_matching': 'acquisition_order', 'adc_chunk_counters': adc_counters,
                  'frame_coordinate_units': {'time': 's'} if 'time' in acquisition.frame_coordinates else {},
                  'sequence_definitions': dict(info.definitions),
                  'scanner_poses': None if getattr(image, 'slicePos', None) is None else np.asarray(image.slicePos)[readout_indices]})


def _calibration(reference, info, raw, counters, axis_order, acquisition):
    """Prepare a proven centred ACS block using the same physical coordinates."""
    if raw is None or acquisition.kind != 'cartesian' or info.trajectory_physical is None:
        return None
    columns = int(reference.NCol)
    events = np.searchsorted(counters, reference.scancounter)
    if (len(set(info.adc_counts)) != 1 or info.adc_counts[0] != columns or
            np.any(events >= len(counters)) or not np.array_equal(counters[events], reference.scancounter)):
        return None
    from mrax.io.line_sampling import prepare_line_acquisition

    ndim = len(acquisition.encoding_shape)
    axes = list(axis_order) if axis_order is not None else list(range(ndim))
    physical = np.asarray(info.trajectory_physical)[(events[:, None] * columns + np.arange(columns)).ravel(), :ndim][:, axes]
    fov = np.asarray(info.definitions['FOV'], float)[axes].copy()
    fov[0] *= acquisition.encoding_shape[-1] / acquisition.recipe.output_shape[-1]
    phase = physical.reshape((-1, columns, ndim))[:, 0, 1:] * fov[1:]
    if np.max(np.abs(phase - np.rint(phase)), initial=0.) > 2e-3:
        return None
    lower, upper = np.rint(phase.min(0)).astype(int), np.rint(phase.max(0)).astype(int)
    size = upper - lower + 1
    if np.any(lower != -(size // 2)):
        warnings.warn("Calibration is not a centred block; raw reference records were retained", RuntimeWarning, stacklevel=3)
        return None
    shape = tuple(size[::-1]) + (columns,)
    serial = np.asarray(raw).transpose(1, 0, 2)
    prepared = prepare_line_acquisition(serial, physical, fov, shape, shape,
        (columns,) * len(events), frame_indices=np.zeros(len(physical), int))
    if prepared is None or prepared.readout_coordinates is not None or not np.all(prepared.acquired):
        warnings.warn("Calibration requires explicit reconstruction; raw reference records were retained", RuntimeWarning, stacklevel=3)
        return None
    return prepared.samples[0]


def _attach_geometry(acquisition, image, indices, frame_indices, info, axis_order, gradient_directions):
    """Use scanner planes only when their relationship to sequence axes is known."""
    from scipy.spatial.transform import Rotation

    from mrax.io.measurement_geometry import plane_geometry
    from mrax.io.measurement_layout import sequence_slice_positions, slice_stack_layout

    if acquisition.domain != 'kspace':
        return acquisition
    from mrax.data_struct import SpatialGeometry

    ndim = len(acquisition.encoding_shape)
    axes = list(axis_order) if axis_order is not None else list(range(ndim))
    shape = acquisition.recipe.output_shape or acquisition.encoding_shape
    fov = np.asarray(info.definitions.get('FOV', ()), float)
    if len(fov) >= ndim:
        spacing = fov[:ndim][axes][::-1] * 1000. / np.asarray(shape)
        acquisition = replace(acquisition, geometry=SpatialGeometry(spatial_shape=shape,
            voxel_sizes=spacing, source='siemens', status='partial'))
    poses = getattr(image, 'slicePos', None)
    if poses is None:
        return acquisition
    poses = np.asarray(poses)[indices]
    if poses.shape != (len(indices), 7) or not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-6):
        return acquisition
    if gradient_directions is None and str(info.definitions.get('OrientationMapping', 'TRA')).upper() != 'TRA':
        warnings.warn("Nonstandard sequence orientation requires explicit gradient_directions (LPS columns); scanner poses were retained without an assumed affine", RuntimeWarning, stacklevel=3)
        return acquisition
    frames = [np.flatnonzero(frame_indices == i) for i in range(acquisition.frame_count)]
    if any(not len(frame) or not np.allclose(poses[frame], poses[frame[0]], atol=1e-3) for frame in frames):
        warnings.warn("Scanner pose varies inside a frame; no single image affine was assigned", RuntimeWarning, stacklevel=3)
        return acquisition
    poses = poses[[frame[0] for frame in frames]]
    # Siemens MDH stores WXYZ, with phase/read/slice rotation columns.
    directions = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()[:, :, [1, 0, 2]]
    ndim = len(acquisition.encoding_shape)
    axes = list(axis_order) if axis_order is not None else list(range(ndim))
    if gradient_directions is not None:
        basis = np.asarray(gradient_directions, float)
        if basis.shape != (3, 3):
            raise ValueError("gradient_directions must contain three LPS direction columns")
        directions = np.broadcast_to(basis, directions.shape).copy()
    directions = directions[:, :, axes + [i for i in range(3) if i not in axes]]
    fov = np.asarray(info.definitions.get('FOV', ()), float)
    if len(fov) < ndim:
        return acquisition
    fov = np.r_[fov[:ndim][axes] * 1000., [float(info.definitions.get('SliceThickness', 0.)) * 1000.] * (3 - ndim)]
    shape = acquisition.recipe.output_shape or acquisition.encoding_shape
    all_positions = np.asarray(image.slicePos)[:, :3]
    all_normals = Rotation.from_quat(np.asarray(image.slicePos)[:, [4, 5, 6, 3]]).as_matrix()[:, :, 2]
    all_positions = sequence_slice_positions(np.asarray(getattr(image, 'Sli', np.zeros(len(all_positions), int))), all_positions, all_normals, info.definitions)
    centres = all_positions[indices][[frame[0] for frame in frames]]
    stack = slice_stack_layout(acquisition.frame_coordinates, centres, directions[:, :, 2]) if ndim == 2 else None
    if stack is not None:
        slices, groups, centres, spacing = stack
        acquisition = replace(acquisition, recipe=replace(acquisition.recipe,
            stack_slices=slices, output_stack_permutation=(2, 0, 1), output_frame_indices=tuple(int(i) for i in groups.ravel())))
        directions = directions[groups[:, 0]]
        fov[2] = spacing * slices
        shape = (slices,) + tuple(shape)
    geometry = plane_geometry(centres, directions, shape, fov, source='siemens')
    return replace(acquisition, geometry=geometry)


def _group_epi_navigators(reference, raw, info, counters, image_keys, selected_keys, frame_fields):
    """Group a proven three-echo navigator using scanner labels and ADC timing."""
    if raw is None or info.trajectory_physical is None or info.adc_times is None:
        return None
    columns = int(reference.NCol)
    if len(set(info.adc_counts)) != 1 or info.adc_counts[0] != columns:
        return None
    events = np.searchsorted(counters, reference.scancounter)
    if np.any(events >= len(counters)) or not np.array_equal(counters[events], reference.scancounter):
        return None
    varying = np.flatnonzero(np.ptp(image_keys, axis=0))
    labels = np.column_stack([getattr(reference, frame_fields[i]) for i in varying]) if len(varying) else np.zeros((len(events), 1))
    selected_labels = selected_keys[:, varying] if len(varying) else np.zeros((len(selected_keys), 1))
    lookup = {tuple(key): i for i, key in enumerate(selected_labels)}
    mapped = np.asarray([lookup.get(tuple(key), -1) for key in labels])
    chosen = np.flatnonzero(mapped >= 0)
    if not len(chosen) or not np.all(np.bincount(mapped[chosen], minlength=len(selected_keys)) == 3):
        return None
    groups = chosen[np.argsort(mapped[chosen], kind='stable')].reshape((len(selected_keys), 3))
    indices = events[groups, None] * columns + np.arange(columns)
    profiles = np.asarray(info.trajectory_physical)[indices]
    read_axis = int(np.argmax(np.median(np.ptp(profiles, axis=-2), axis=(0, 1))))
    polarity = np.sign(profiles[:, :, -1, read_axis] - profiles[:, :, 0, read_axis]).astype(np.int8)
    times = np.mean(np.asarray(info.adc_times)[indices], axis=-1)
    if (np.any(polarity[:, 0] == 0) or not np.all(polarity[:, 0] == polarity[:, 2]) or
            not np.all(polarity[:, 0] == -polarity[:, 1]) or np.any(np.diff(times, axis=1) <= 0)):
        return None
    values = np.asarray(raw)[groups].copy()
    negative = polarity < 0
    values[negative] = values[negative, :, ::-1]
    return {'epi_navigators': values}, {'epi_navigator_polarity': polarity, 'epi_navigator_times': times}
