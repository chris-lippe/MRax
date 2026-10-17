"""ISMRMRD file adapter. Sample ordering and geometry end at this boundary."""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from ismrmrd import xsd

from mrax.data_struct import AcquisitionData, ReconstructionRecipe
from mrax.io.import_definitions import NormalizedMetadata
from mrax.io.measurement import MeasurementData, sampled_reconstruction_recipe
from mrax.io.measurement_layout import sequence_slice_positions, slice_stack_layout

_FRAME_FIELDS = ("average", "contrast", "phase", "repetition", "set", "slice")
_ROLES = {"noise": 19, "calibration": 20, "navigation": 23, "phase_correction": 24,
          "feedback": 26, "dummy": 27, "realtime_feedback": 28,
          "surface_coil_correction": 29, "phase_stabilization_reference": 30, "phase_stabilization": 31}


def _read_header(xml):
    """Accept an incomplete optional information section without inventing posture."""
    xml = xml.decode() if isinstance(xml, bytes) else xml
    tree = ET.fromstring(xml)
    for section in list(tree):
        if section.tag.rsplit('}', 1)[-1] == 'measurementInformation' and not any(
                child.tag.rsplit('}', 1)[-1] == 'patientPosition' and child.text
                for child in section):
            tree.remove(section)
            warnings.warn("Ignoring incomplete optional MRD measurementInformation (missing patientPosition)",
                          RuntimeWarning, stacklevel=3)
    return xsd.CreateFromDocument(ET.tostring(tree, encoding='unicode'))


def inspect_ismrmrd_measurement(path):
    """List MRD acquisition groups and image series using headers only."""
    entries = []
    with h5py.File(path, 'r') as file:
        for name, group in file.items():
            if not isinstance(group, h5py.Group):
                continue
            if 'data' in group and isinstance(group['data'], h5py.Dataset) and group['data'].dtype.names and 'head' in group['data'].dtype.names:
                entries.append({'group': name})
            for series, image in group.items():
                if isinstance(image, h5py.Group) and {'data', 'header'} <= set(image):
                    entries.append({'group': name, 'image_series': series})
    if not entries:
        raise ValueError("HDF5 file contains no ISMRMRD measurement groups")
    return entries


def _flag(values, number):
    return (values & np.uint64(1 << (number - 1))) != 0


def _metadata_template(path, group, heads):
    """Use explicitly paired acquisition metadata without replacing sample headers."""
    with h5py.File(path, 'r') as file:
        source = file[group]
        template = source['data'].fields('head')[:]
        if len(template) != len(heads):
            raise ValueError("MRD metadata template must match the raw acquisition count and order")
        for key in ('number_of_samples', 'active_channels'):
            declared = template[key]
            if np.any((declared != 0) & (declared != heads[key])):
                raise ValueError(f"MRD metadata template has inconsistent {key}")
        for key in ('number_of_samples', 'available_channels', 'active_channels', 'channel_mask',
                    'discard_pre', 'discard_post', 'center_sample', 'trajectory_dimensions',
                    'sample_time_us', 'acquisition_time_stamp', 'physiology_time_stamp'):
            template[key] = heads[key]
        xml = source['xml'][0]
        header = _read_header(xml)
    warnings.warn("Using explicit MRD metadata template; acquisition order must match the raw samples",
                  RuntimeWarning, stacklevel=3)
    return header, template


def _phase_index(value, limit, size):
    if limit is None:
        if not 0 <= value < size:
            raise ValueError("ISMRMRD encoding indices fall outside the declared grid")
        return value
    shifted = value + size // 2 - int(limit.center)
    if 0 <= shifted < size:
        return shifted
    if not (int(limit.minimum) <= value <= int(limit.maximum)) or int(limit.maximum) - int(limit.minimum) > size:
        raise ValueError("ISMRMRD encoding indices fall outside the declared grid")
    # Fourier bins are periodic; a declared origin need not be the middle bin.
    return shifted % size


def _sequence_profiles(info, heads, indices, shape, encoding):
    """Use common sequence readout vectors; MRD indices retain phase ordering."""
    if info is None or info.trajectory_physical is None or not info.adc_counts:
        return None
    counts = np.asarray(info.adc_counts, int)
    selected = np.flatnonzero(counts == shape[-1])
    if not len(selected):
        return None
    physical = np.asarray(info.trajectory_physical)
    starts = np.r_[0, np.cumsum(counts)][selected]
    profiles = physical[starts[:, None] + np.arange(shape[-1])]
    spread = np.ptp(profiles, axis=1)
    read_axis = int(np.argmax(np.median(spread, axis=0)))
    other = [i for i in range(physical.shape[1]) if i != read_axis]
    if len(other) != len(shape) - 1 or np.max(spread[:, other], initial=0.) > 1e-2:
        return None
    signs = np.sign(profiles[:, -1, read_axis] - profiles[:, 0, read_axis])
    nonzero = signs != 0
    profiles, signs = profiles[nonzero], signs[nonzero]
    if not len(signs):
        return None
    fov = np.asarray(info.definitions.get('FOV', []), float)
    target_fov = float(encoding.reconSpace.fieldOfView_mm.x) / 1000.
    if len(fov) <= read_axis or not np.isclose(fov[read_axis], target_fov, rtol=.01, atol=1e-6):
        warnings.warn("Sequence and MRD readout FOV differ; retaining declared MRD encoding", RuntimeWarning, stacklevel=3)
        return None
    stored_signs = np.where(_flag(heads['flags'][indices], 22), -1, 1)
    orientation = stored_signs[0] * signs[0] if len(np.unique(signs)) == len(np.unique(stored_signs)) == 1 else 1.
    signs = signs * orientation
    pixel = target_fov / int(encoding.reconSpace.matrixSize.x)
    read = profiles[..., read_axis] * pixel * orientation
    read[signs < 0] = read[signs < 0, ::-1]
    vectors = []
    for sign in (1, -1):
        same = np.flatnonzero(signs == sign)
        vector = read[same[0]] if len(same) else None
        if vector is not None and not np.allclose(read[same], vector, atol=2e-6, rtol=1e-5):
            warnings.warn("Sequence readout profiles vary; supply measured trajectories for exact encoding", RuntimeWarning, stacklevel=3)
            return None
        vectors.append(vector)
    if vectors[0] is None:
        vectors[0] = vectors[1]
    if vectors[1] is None:
        vectors[1] = vectors[0]
    coordinates = np.asarray(vectors)
    if np.allclose(coordinates, np.fft.fftshift(np.fft.fftfreq(shape[-1]))[None], atol=2e-6, rtol=1e-5):
        coordinates = None
    return coordinates


def _geometry(heads, shape, fov, *, source):
    from mrax.io.measurement_geometry import plane_geometry

    directions = np.stack([np.column_stack([h[k] for k in ("read_dir", "phase_dir", "slice_dir")]) for h in heads])
    return plane_geometry(heads['position'], directions, shape, fov, source=source)


def _metadata(header, coordinates, frame_count):
    params = header.sequenceParameters
    fields = {}
    if params is not None:
        for name, key in (("echo_times", "TE"), ("repetition_times", "TR"), ("inversion_times", "TI"), ("flip_angles", "flipAngle_deg")):
            values = getattr(params, key, None)
            if values:
                values = np.asarray(values, float)
                contrast = coordinates.get("contrast")
                fields[name] = values[contrast].tolist() if contrast is not None and len(values) > 1 and np.max(contrast) < len(values) else values.tolist()
    return NormalizedMetadata(frame_count=frame_count, **fields)


def _images(group, header, *, series=None, frame_selection=None, sequence_info=None):
    choices = [name for name in group if isinstance(group[name], h5py.Group) and {'data', 'header'} <= set(group[name])]
    if series is None:
        if len(choices) != 1:
            raise ValueError(f"ISMRMRD contains {len(choices)} image series; choose image_series explicitly")
        series = choices[0]
    if series not in choices:
        raise ValueError("Requested ISMRMRD image series is absent")
    pixel_source = group[series]['data']
    heads = group[series]['header'][:]
    if not len(heads):
        raise ValueError("ISMRMRD image series is empty")
    heads['position'] = sequence_slice_positions(heads['slice'], heads['position'], heads['slice_dir'],
        {} if sequence_info is None else sequence_info.definitions)
    missing_planes = len(np.unique(heads['slice'])) > 1 and np.allclose(heads['position'], heads['position'][0], atol=1e-3)
    if len(heads) != pixel_source.shape[0]:
        raise ValueError("ISMRMRD image headers and pixel frames do not match")
    source_frame_count = len(heads)
    contrast_count = len(np.unique(heads['contrast']))
    selection = np.arange(source_frame_count)
    if frame_selection is None:
        arrays = np.asarray(pixel_source)
    else:
        selection = np.atleast_1d(np.arange(len(heads))[frame_selection])
        if not len(selection):
            raise ValueError("Frame selection is empty")
        unique, inverse = np.unique(selection, return_inverse=True)
        arrays, heads = np.asarray(pixel_source[unique])[inverse], heads[selection]
    if arrays.ndim != 5 or arrays.shape[1] != 1:
        raise ValueError("ISMRMRD images require (frame, 1, z, y, x); receiver images must be combined explicitly")
    arrays = arrays[:, 0]
    if arrays.shape[1] == 1:
        arrays = arrays[:, 0]
    coords = {k: heads[k].astype(int) for k in _FRAME_FIELDS}
    fov = np.asarray(heads[0]['field_of_view'], float)
    if arrays.ndim == 3:
        stack = slice_stack_layout(coords, heads['position'], heads['slice_dir'])
        if stack is not None:
            slices, groups, centres, spacing = stack
            arrays = arrays[groups.ravel()].reshape((len(groups), slices) + arrays.shape[1:])
            volume_heads = heads[groups[:, 0]].copy()
            volume_heads['position'] = centres
            heads = volume_heads
            coords = {k: values[groups[:, 0]] for k, values in coords.items()}
            selection = selection[groups[:, 0]]
            fov[2] = spacing * slices
    geometry = _geometry(heads, arrays.shape[1:], fov, source="ismrmrd")
    if missing_planes:
        geometry = replace(geometry, status="partial")
    meta = _metadata(header, coords, len(arrays)) if header else NormalizedMetadata(frame_count=len(arrays))
    return MeasurementData(arrays, spatial_geometry=geometry, normalized_metadata=meta,
        metadata={"format": "ismrmrd", "image_series": series, "frame_coordinates": coords,
            "frame_indices": selection, "source_frame_count": source_frame_count,
            "source_contrast_count": contrast_count,
            "field_strength_mhz": float(header.experimentalConditions.H1resonanceFrequency_Hz) / 1e6 if header else None})


def read_ismrmrd_measurement(path, *, group=None, image_series=None, encoding_ref=0,
                            frame_selection=None, trajectory_units=None, density_column=None,
                            load_auxiliary=True, sequence_info=None, density="auto", metadata_template=None,
                            trajectory_mode="auto", noise_records=(), exclude_records=(),
                            sample_window=None, center_sample=None):
    """Read MRD acquisitions or one image series without reconstructing.

    Trajectory coordinates default to cycles/pixel. Files using another convention
    must declare ``trajectory_units`` (cycles/fov, cycles/m, radians/pixel).
    An explicit ``metadata_template`` MRD file may supply XML and acquisition
    roles, counters and encoding geometry for a separately converted stream in
    exactly the same order.
    Sample sizes, receiver layout, discard windows and clock values stay with data.
    """
    if trajectory_mode not in ('auto', 'cartesian', 'sampled'):
        raise ValueError("trajectory_mode must be auto, cartesian or sampled")
    with h5py.File(path, 'r') as file:
        groups = [k for k in file if isinstance(file[k], h5py.Group) and
                  ('data' in file[k] or any(isinstance(v, h5py.Group) and 'data' in v for v in file[k].values()))]
        if group is None:
            if len(groups) != 1:
                raise ValueError("Choose an ISMRMRD group explicitly when several groups contain measurements")
            group = groups[0]
        source = file[group]
        header = _read_header(source['xml'][0]) if 'xml' in source else None
        if image_series is not None or 'data' not in source or not source['data'].dtype.names or 'head' not in source['data'].dtype.names:
            return _images(source, header, series=image_series, frame_selection=frame_selection, sequence_info=sequence_info)
        records = source['data']
        heads = records.fields('head')[:]
        if metadata_template is not None:
            template_path = Path(metadata_template).expanduser()
            if not template_path.is_absolute():
                template_path = Path(path).parent / template_path
            header, heads = _metadata_template(template_path, group, heads)
        if header is None:
            raise ValueError("ISMRMRD acquisitions require an XML encoding header")
        if not len(heads):
            raise ValueError("ISMRMRD acquisition dataset is empty")
        selected_encoding = heads['encoding_space_ref'] == encoding_ref
        flags = heads['flags']
        for selection, role in ((noise_records, 'noise'), (exclude_records, None)):
            selection = np.asarray(selection)
            if selection.size:
                if (selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer)
                        or np.any(selection < 0) or np.any(selection >= len(heads))):
                    raise ValueError("Explicit MRD record indices must be a one-dimensional integer list within the file")
                if role:
                    flags[selection] |= np.uint64(1 << (_ROLES[role] - 1))
                else:
                    selected_encoding[selection] = False
        auxiliary_indices = {}
        imaging = selected_encoding & (heads['number_of_samples'] > 0)
        for name, flag in _ROLES.items():
            role = _flag(flags, flag) & selected_encoding
            if name == 'calibration':
                role &= ~_flag(flags, 21)
            auxiliary_indices[name] = np.flatnonzero(role & (heads['number_of_samples'] > 0))
            imaging &= ~role
        indices = np.flatnonzero(imaging)
        if not len(indices):
            raise ValueError("ISMRMRD file contains no nonempty imaging acquisitions")
        if sample_window is not None:
            window = np.asarray(sample_window)
            if (window.shape != (2,) or not np.issubdtype(window.dtype, np.integer)
                    or window[0] < 0 or window[1] <= window[0]
                    or np.any(window[1] > heads['number_of_samples'][indices])):
                raise ValueError("sample_window must be a nonempty integer [start, stop] within every imaging record")
            heads['discard_pre'][indices] = window[0]
            heads['discard_post'][indices] = heads['number_of_samples'][indices] - window[1]
        if center_sample is not None:
            if (not isinstance(center_sample, (int, np.integer)) or center_sample < 0
                    or np.any(center_sample >= heads['number_of_samples'][indices])):
                raise ValueError("center_sample must be an integer within every imaging record")
            heads['center_sample'][indices] = center_sample
        if np.any(selected_encoding & (heads['number_of_samples'] == 0)):
            warnings.warn("Empty ISMRMRD records were skipped", RuntimeWarning, stacklevel=2)
        keys = np.column_stack([heads['idx'][k][indices] for k in _FRAME_FIELDS])
        frame_keys, frame_inverse = np.unique(keys, axis=0, return_inverse=True)
        source_frame_count = len(frame_keys)
        source_contrast_count = len(np.unique(frame_keys[:, _FRAME_FIELDS.index('contrast')]))
        selected_frames = np.arange(len(frame_keys)) if frame_selection is None else np.atleast_1d(np.arange(len(frame_keys))[frame_selection])
        if not len(selected_frames):
            raise ValueError("Frame selection is empty")
        keep = np.isin(frame_inverse, selected_frames)
        indices = indices[keep]
        old_inverse = frame_inverse[keep]
        if len(np.unique(selected_frames)) != len(selected_frames):
            raise ValueError("Raw frame selection must not repeat frames")
        frame_lookup = np.full(len(frame_keys), -1, dtype=int)
        frame_lookup[selected_frames] = np.arange(len(selected_frames))
        frame_inverse = frame_lookup[old_inverse]
        frame_keys = frame_keys[selected_frames]
        coordinates = {k: frame_keys[:, i] for i, k in enumerate(_FRAME_FIELDS)}
        enc = header.encoding[encoding_ref]
        encoded_xyz = [int(getattr(enc.encodedSpace.matrixSize, k)) for k in 'xyz']
        recon_xyz = [int(getattr(enc.reconSpace.matrixSize, k)) for k in 'xyz']
        ndim = 3 if encoded_xyz[2] > 1 else 2
        missing_planes = False
        if ndim == 2:
            heads['position'][imaging] = sequence_slice_positions(heads['idx']['slice'][imaging], heads['position'][imaging],
                heads['slice_dir'][imaging], {} if sequence_info is None else sequence_info.definitions)
            missing_planes = len(np.unique(heads['idx']['slice'][imaging])) > 1 and np.allclose(heads['position'][imaging], heads['position'][imaging][0], atol=1e-3)
        channels = np.unique(heads['active_channels'][indices])
        if len(channels) != 1:
            raise ValueError("Receiver count changes within the selected imaging acquisition")
        coils, frames = int(channels[0]), len(frame_keys)
        aux = {}
        if load_auxiliary:
            for name, ix in auxiliary_indices.items():
                if len(ix):
                    values = [records[int(j)]['data'].view(np.complex64).reshape(int(heads[j]['active_channels']), -1) for j in ix]
                    if len({v.shape for v in values}) == 1:
                        aux[name] = np.stack(values)
                    else:
                        warnings.warn(f"Variable-length {name} records are not attached", RuntimeWarning, stacklevel=2)
        has_trajectory = bool(np.any(heads['trajectory_dimensions'][indices]))
        trajectory_axes, physical_trajectory, cartesian_grid = None, False, False
        recovered_axes = ()
        if has_trajectory and trajectory_mode != 'cartesian':
            from mrax.io.ismrmrd_trajectory import sequence_trajectory_encoding

            trajectory_units, trajectory_axes, physical_trajectory, cartesian_grid, recovered_axes = sequence_trajectory_encoding(
                records, heads, indices, sequence_info, enc, encoded_xyz, ndim, trajectory_units)
            if recovered_axes:
                warnings.warn("Incomplete MRD trajectory axes recovered from matching Pulseq sequence and Cartesian counters",
                              RuntimeWarning, stacklevel=2)
        if trajectory_mode == 'sampled' and not has_trajectory:
            raise ValueError("Sampled trajectory_mode requires stored trajectories")
        if trajectory_mode == 'cartesian':
            if str(enc.trajectory.value if hasattr(enc.trajectory, 'value') else enc.trajectory).lower() != 'cartesian':
                raise ValueError("Cartesian trajectory_mode requires a declared Cartesian MRD encoding")
            if has_trajectory:
                warnings.warn("Using explicit Cartesian counters instead of stored MRD trajectories",
                              RuntimeWarning, stacklevel=2)
        is_sampled = has_trajectory and trajectory_mode != 'cartesian' and not (cartesian_grid and trajectory_mode == 'auto')
        if is_sampled:
            if np.any(heads['trajectory_dimensions'][indices] < ndim):
                raise ValueError("Every sampled imaging record requires spatial trajectory coordinates")
            samples, acquired, trajectory, recorded_density = _sampled(records, heads, indices, frame_inverse,
                frames, coils, encoded_xyz, enc, ndim, trajectory_units, density_column, trajectory_axes,
                sequence_info.trajectory_physical if recovered_axes else None)
            shape = tuple(reversed(encoded_xyz[:ndim]))
            if recorded_density is not None:
                resolved_density = recorded_density
            elif isinstance(density, str) and density == 'auto':
                from mrax.reco.density import triangulated_density

                resolved_density = triangulated_density(trajectory, spatial_shape=shape)
            else:
                resolved_density = None if density is None else np.asarray(density)
            recipe = sampled_reconstruction_recipe(output_shape=tuple(reversed(recon_xyz[:ndim])))
            acquisition = AcquisitionData(samples, shape, ("partition", "phase", "read")[-ndim:],
                kind='sampled', trajectory=trajectory, acquired=acquired, density=resolved_density,
                frame_coordinates=coordinates, auxiliary_samples=aux, recipe=recipe)
        else:
            read_lengths = heads['number_of_samples'][indices].astype(int)
            if sample_window is not None:
                read_lengths -= heads['discard_pre'][indices] + heads['discard_post'][indices]
            read_count = int(np.max(read_lengths))
            encoded_xyz = [max(a, b) for a, b in zip(encoded_xyz, recon_xyz)]
            encoded_xyz[0] = max(encoded_xyz[0], read_count)
            shape = tuple(reversed(encoded_xyz[:ndim]))
            readout_coordinates = _sequence_profiles(sequence_info, heads, indices, shape, enc)
            calibration_indices = np.flatnonzero(selected_encoding & (heads['number_of_samples'] > 0) & (_flag(flags, 20) | _flag(flags, 21)))
            if load_auxiliary and len(calibration_indices):
                calibration = _calibration_grid(records, heads, calibration_indices, enc, shape, coils)
                if calibration is not None:
                    aux['calibration_grid'] = calibration
            samples = np.zeros((frames, coils) + shape, np.complex64)
            counts = np.zeros((frames, 1) + shape, np.uint32)
            polarity = np.zeros((frames, 1) + shape[:-1] + (1,), np.int8)
            for j, frame in zip(indices, frame_inverse):
                head = heads[j]
                raw = records[int(j)]['data'].view(np.complex64).reshape(coils, -1)
                pre, post = int(head['discard_pre']), int(head['discard_post'])
                end = raw.shape[-1] - post
                x = np.arange(pre, end) - int(head['center_sample']) + shape[-1] // 2
                raw = raw[:, pre:end]
                reverse = bool(_flag(head['flags'], 22))
                if reverse:
                    raw = raw[:, ::-1]
                phase = [int(head['idx']['kspace_encode_step_1'])]
                if ndim == 3:
                    phase.insert(0, int(head['idx']['kspace_encode_step_2']))
                for dim, field_name in enumerate(('kspace_encoding_step_2', 'kspace_encoding_step_1')[-len(phase):]):
                    limit = getattr(enc.encodingLimits, field_name, None) if enc.encodingLimits else None
                    phase[dim] = _phase_index(phase[dim], limit, shape[dim])
                if any(not 0 <= p < n for p, n in zip(phase, shape)) or np.any(x < 0) or np.any(x >= shape[-1]):
                    raise ValueError("ISMRMRD encoding indices or discard window fall outside the declared grid")
                key = (int(frame), slice(None), *phase)
                samples[key][..., x] += raw
                counts[(int(frame), 0, *phase)][x] += 1
                polarity[(int(frame), 0, *phase, 0)] = -1 if reverse else 1
            np.divide(samples, np.maximum(counts, 1), out=samples)
            undersampled = np.any(counts == 0)
            if undersampled:
                warnings.warn("Cartesian acquisition is undersampled; FFT gives a zero-filled preview. Use calibrated iterative reconstruction for unfolding", RuntimeWarning, stacklevel=2)
            epi = bool(np.any(polarity < 0))
            recipe = ReconstructionRecipe(output_shape=tuple(reversed(recon_xyz[:ndim])),
                epi_phase_correction='entropy' if epi and len(auxiliary_indices['phase_correction']) else 'none',
                output_scale=float(np.sqrt(np.prod(recon_xyz[:ndim]) / np.prod(shape))))
            acquisition = AcquisitionData(samples, shape, ("partition", "phase", "read")[-ndim:],
                acquired=counts > 0, statistical_weights=counts if np.any(counts > 1) else None,
                frame_coordinates=coordinates, auxiliary_samples=aux,
                readout_polarity=polarity if epi or readout_coordinates is not None else None,
                readout_coordinates=readout_coordinates, recipe=recipe)
        fov = [float(getattr(enc.reconSpace.fieldOfView_mm, k)) for k in 'xyz']
        slice_count = len(np.unique(heads['idx']['slice'][imaging]))
        if ndim == 2 and slice_count > 1:
            fov[2] /= slice_count
        frame_heads = [heads[indices[np.flatnonzero(frame_inverse == i)[0]]] for i in range(frames)]
        frame_heads = np.asarray(frame_heads)
        sequence_directions = False
        if (physical_trajectory and header.measurementInformation is None
                and trajectory_axes is not None and trajectory_axes != list(range(ndim))):
            directions = np.stack([np.column_stack([h[k] for k in ('read_dir', 'phase_dir', 'slice_dir')]) for h in frame_heads])
            if np.allclose(directions, np.eye(3), atol=1e-6):
                # Placeholder MRD directions cannot establish patient posture.
                # The matched sequence still establishes scanner gradient axes.
                order = list(trajectory_axes) + list(range(ndim, 3))
                for i, key in enumerate(('read_dir', 'phase_dir', 'slice_dir')):
                    frame_heads[key] = np.eye(3)[:, order[i]]
                sequence_directions = True
        output_shape = tuple(reversed(recon_xyz[:ndim]))
        if ndim == 2:
            stack = slice_stack_layout(coordinates, frame_heads['position'], frame_heads['slice_dir'])
            if stack is not None:
                slices, groups, centres, spacing = stack
                acquisition = replace(acquisition, recipe=replace(acquisition.recipe,
                    stack_slices=slices, output_stack_permutation=(2, 0, 1),
                    output_frame_indices=tuple(int(i) for i in groups.ravel())))
                frame_heads = frame_heads[groups[:, 0]].copy()
                frame_heads['position'] = centres
                fov[2] = spacing * slices
                output_shape = (slices,) + output_shape
        geometry = _geometry(frame_heads, output_shape, fov, source='ismrmrd')
        if missing_planes:
            geometry = replace(geometry, status='partial')
        if sequence_directions:
            geometry = replace(geometry, status='partial', source='ismrmrd-sequence')
        acquisition = replace(acquisition, geometry=geometry)
        meta = _metadata(header, coordinates, frames)
        executed_echo_times = None
        if (sequence_info is not None and sequence_info.adc_echo_times_ms is not None
                and len(sequence_info.adc_echo_times_ms) == len(heads)):
            values = np.asarray(sequence_info.adc_echo_times_ms)[indices]
            grouped = [values[frame_inverse == i] for i in range(frames)]
            if all(np.allclose(v, v[0], atol=.01, rtol=1e-5) for v in grouped):
                executed_echo_times = [float(np.median(v)) for v in grouped]
        return MeasurementData(acquisition=acquisition, spatial_geometry=geometry, normalized_metadata=meta,
            metadata={'format': 'ismrmrd', 'path': str(path), 'group': group,
                      'field_strength_mhz': float(header.experimentalConditions.H1resonanceFrequency_Hz) / 1e6,
                      'encoding_ref': encoding_ref, 'frame_coordinates': coordinates,
                      'frame_indices': selected_frames, 'source_frame_count': source_frame_count,
                      'source_contrast_count': source_contrast_count,
                      'trajectory_units': trajectory_units, 'trajectory_axes': trajectory_axes,
                      'sequence_trajectory_matched': physical_trajectory,
                      'cartesian_trajectory_validated': cartesian_grid,
                      'trajectory_axes_from_sequence': recovered_axes,
                      'executed_echo_times_ms': executed_echo_times,
                      'metadata_template_used': metadata_template is not None,
                      'sequence_parameters_available': header.sequenceParameters is not None})


def _sampled(records, heads, indices, frame_inverse, frames, coils, xyz, encoding, ndim, units, density_column, axes=None,
             sequence_trajectory=None):
    from mrax.io.ismrmrd_trajectory import convert_coordinates

    if units is None:
        largest = 0.
        for j in indices:
            coords = records[int(j)]['traj'].reshape(-1, int(heads[j]['trajectory_dimensions']))
            largest = max(largest, float(np.max(np.abs(coords[:, :ndim]))))
        units = 'cycles/pixel' if largest <= .75 else 'cycles/fov'
        if units == 'cycles/fov':
            warnings.warn("ISMRMRD trajectory units inferred as cycles/fov; set trajectory_units explicitly to confirm", RuntimeWarning, stacklevel=3)
    # A trajectory slot is shared only when both encoding counters and coordinates agree.
    slots = sorted({(int(heads[j]['idx']['kspace_encode_step_2']), int(heads[j]['idx']['kspace_encode_step_1']), int(heads[j]['idx']['segment'])) for j in indices})
    slot_lookup = {key: i for i, key in enumerate(slots)}
    lengths = {}
    for j in indices:
        key = tuple(int(heads[j]['idx'][k]) for k in ('kspace_encode_step_2', 'kspace_encode_step_1', 'segment'))
        n = int(heads[j]['number_of_samples']) - int(heads[j]['discard_pre']) - int(heads[j]['discard_post'])
        if key in lengths and lengths[key] != n:
            raise ValueError("ADC length changes for a shared trajectory slot")
        lengths[key] = n
    offsets = np.r_[0, np.cumsum([lengths[key] for key in slots])]
    samples = np.zeros((frames, coils, offsets[-1]), np.complex64)
    counts = np.zeros((frames, 1, offsets[-1]), np.uint32)
    trajectory = np.full((offsets[-1], ndim), np.nan, np.float32)
    density = np.ones(offsets[-1], np.float32) if density_column is not None else None
    sequence_starts = np.r_[0, np.cumsum(heads['number_of_samples'])] if sequence_trajectory is not None else None
    for j, frame in zip(indices, frame_inverse):
        head = heads[j]
        key = tuple(int(head['idx'][k]) for k in ('kspace_encode_step_2', 'kspace_encode_step_1', 'segment'))
        slot = slot_lookup[key]
        target = slice(int(offsets[slot]), int(offsets[slot + 1]))
        row = records[int(j)]
        pre, post = int(head['discard_pre']), int(head['discard_post'])
        end = int(head['number_of_samples']) - post
        values = row['data'].view(np.complex64).reshape(coils, -1)[:, pre:end]
        coords = row['traj'].reshape(-1, int(head['trajectory_dimensions']))[pre:end]
        if sequence_trajectory is not None:
            coords = sequence_trajectory[sequence_starts[j] + pre:sequence_starts[j] + end]
        if coords.shape[-1] < ndim:
            raise ValueError("ISMRMRD trajectory has too few spatial coordinates")
        coords = convert_coordinates(coords, units, xyz, encoding, ndim, axes)
        coords = coords[:, ::-1]
        if np.isfinite(trajectory[target]).all() and not np.allclose(trajectory[target], coords, atol=2e-6, rtol=1e-5):
            raise ValueError("Trajectory changes between frames; import individual frames explicitly")
        trajectory[target] = coords
        samples[int(frame), :, target] += values
        counts[int(frame), :, target] += 1
        if density is not None:
            density[target] = row['traj'].reshape(-1, int(head['trajectory_dimensions']))[pre:end, density_column]
    np.divide(samples, np.maximum(counts, 1), out=samples)
    return samples, counts > 0, trajectory, density


def _calibration_grid(records, heads, indices, encoding, shape, coils):
    ndim = len(shape)
    phase_columns = ('kspace_encode_step_2', 'kspace_encode_step_1')[-(ndim - 1):]
    phase = np.column_stack([heads['idx'][key][indices] for key in phase_columns]).astype(int)
    for dim, name in enumerate(('kspace_encoding_step_2', 'kspace_encoding_step_1')[-(ndim - 1):]):
        limit = getattr(encoding.encodingLimits, name, None) if encoding.encodingLimits else None
        try:
            phase[:, dim] = [_phase_index(int(value), limit, shape[dim]) for value in phase[:, dim]]
        except ValueError:
            warnings.warn("Calibration indices exceed declared encoding limits; raw records were retained",
                          RuntimeWarning, stacklevel=3)
            return None
    pre = heads['discard_pre'][indices].astype(int)
    end = heads['number_of_samples'][indices].astype(int) - heads['discard_post'][indices].astype(int)
    start_x = pre - heads['center_sample'][indices].astype(int) + shape[-1] // 2
    end_x = end - heads['center_sample'][indices].astype(int) + shape[-1] // 2
    lower = np.r_[phase.min(0), start_x.min()]
    upper = np.r_[phase.max(0) + 1, end_x.max()]
    size = upper - lower
    if np.any(lower < 0) or np.any(upper > np.asarray(shape)) or not np.array_equal(lower, np.asarray(shape) // 2 - size // 2):
        warnings.warn("Calibration is not a centred Cartesian block; raw calibration records were retained", RuntimeWarning, stacklevel=3)
        return None
    grid = np.zeros((coils,) + tuple(size), np.complex64)
    counts = np.zeros(tuple(size), np.uint16)
    for row, j in enumerate(indices):
        raw = records[int(j)]['data'].view(np.complex64).reshape(int(heads[j]['active_channels']), -1)
        if raw.shape[0] != coils:
            raise ValueError("Calibration receiver count differs from imaging")
        raw = raw[:, pre[row]:end[row]]
        if bool(_flag(heads[j]['flags'], 22)):
            raw = raw[:, ::-1]
        key = tuple(phase[row] - lower[:-1])
        target = slice(int(start_x[row] - lower[-1]), int(end_x[row] - lower[-1]))
        grid[(slice(None),) + key + (target,)] += raw
        counts[key + (target,)] += 1
    if np.any(counts == 0):
        warnings.warn("Calibration block is incomplete; raw calibration records were retained", RuntimeWarning, stacklevel=3)
        return None
    grid /= counts[None]
    return grid
