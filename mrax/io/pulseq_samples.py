"""Prepare ADC streams using sequence coordinates, without reconstructing images."""
from __future__ import annotations

import numpy as np

from mrax.data_struct import AcquisitionData, ReconstructionRecipe
from mrax.io.measurement import sampled_reconstruction_recipe


def prepare_pulseq_samples(info, samples, *, spatial_shape=None, frame_indices=None, auxiliary=None, axis_order=None, density="auto"):
    """Map coil-first serial ADC samples to explicit Fourier coordinates.

    Repeated visits to a Cartesian location remain distinct frames. Coordinates
    that cannot be represented on the declared lattice remain sampled data.
    """
    samples = np.asarray(samples)
    samples = samples.astype(np.result_type(samples.dtype, np.complex64), copy=False)
    if samples.ndim not in (2, 3):
        raise ValueError("Pulseq samples must have (coil, sample) or (coil, ADC, sample) axes")
    sample_count = int(np.prod(samples.shape[1:]))
    physical = info.trajectory_physical
    fov = np.asarray(info.definitions.get('FOV', info.definitions.get('fov', [])), float).reshape(-1).copy()
    shape = tuple(spatial_shape or info.spatial_shape or ())
    if physical is None or not shape or len(fov) < len(shape):
        # ADC recordings without a proven spatial encoding are not MRI k-space.
        return AcquisitionData(samples.reshape((samples.shape[0], -1))[None], (sample_count,), ('sample',), domain='adc')
    if axis_order is not None and sorted(axis_order) != list(range(len(shape))):
        raise ValueError("axis_order must permute the physical encoding axes")
    if np.any(fov[:len(shape)] <= 0) or not np.isfinite(fov[:len(shape)]).all():
        raise ValueError("Pulseq FOV must contain positive finite spatial lengths")
    physical = np.asarray(physical)[:, :len(shape)]
    if len(physical) != sample_count:
        raise ValueError("Measured ADC count differs from the sequence trajectory; provide the matching sequence or explicit ADC selection")
    output_shape = shape
    if info.adc_counts and len(set(info.adc_counts)) == 1:
        n = int(info.adc_counts[0])
        if len(physical) % n == 0:
            lines = physical.reshape((-1, n, len(shape)))
            spread = np.ptp(lines, axis=1)
            read_axis = int(np.argmax(np.median(spread, axis=0)))
            others = [i for i in range(len(shape)) if i != read_axis]
            if np.max(spread[:, others], initial=0.) < 1e-2 and len(lines) > 1:
                phase_axis = max(others, key=lambda i: np.count_nonzero(np.diff(lines[:, 0, i])))
                axes = list(axis_order) if axis_order is not None else [read_axis, phase_axis] + [i for i in others if i != phase_axis]
                physical = physical[:, axes]
                fov = fov[axes]
                shape = tuple(np.asarray(shape[::-1])[axes][::-1])
                output_shape = shape
                steps = np.diff(physical[:n, 0]) * fov[0]
                step = float(np.median(np.abs(steps)))
                oversampling = int(round(1. / step)) if step > 0 else 1
                if oversampling > 1 and np.allclose(np.abs(steps) * oversampling, 1., atol=2e-3):
                    fov[0] *= oversampling
                    shape = shape[:-1] + (shape[-1] * oversampling,)
    from mrax.io.line_sampling import prepare_line_acquisition

    line_acquisition = prepare_line_acquisition(samples, physical, fov, shape, output_shape,
        info.adc_counts, frame_indices=frame_indices, auxiliary=auxiliary)
    if line_acquisition is not None:
        return line_acquisition
    samples = samples.reshape((samples.shape[0], -1))
    lattice = physical * fov[None, :len(shape)]
    if np.any(np.ptp(lattice, axis=0) < 1):
        return AcquisitionData(samples.reshape((samples.shape[0], -1))[None], (sample_count,), ('sample',), domain='adc')
    normalized = (lattice / np.asarray(shape[::-1])[None])[:, ::-1].astype(np.float32)
    integer = np.rint(lattice)
    if np.max(np.abs(lattice - integer)) < 2e-3:
        indices = integer.astype(int)[:, ::-1] + np.asarray(shape)[None] // 2
        # Positive and negative Nyquist points are the same discrete Fourier bin.
        indices = np.where(indices == np.asarray(shape), 0, indices)
        valid = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
        if not np.all(valid):
            raise ValueError("Sequence trajectory extends outside the encoding matrix")
        linear = np.ravel_multi_index(indices.T, shape)
        if frame_indices is None and info.adc_counts and len(set(info.adc_counts)) == 1:
            n = int(info.adc_counts[0])
            lines = indices.reshape((-1, n, len(shape)))
            if np.all(np.ptp(lines[..., :-1], axis=1) == 0):
                _, inverse, visits = np.unique(lines[:, 0, :-1], axis=0, return_inverse=True, return_counts=True)
                if len(set(visits)) != 1:
                    raise ValueError("Nonuniform Cartesian repetitions require explicit frame labels")
                counters = np.zeros(len(visits), int)
                line_frames = []
                for key in inverse:
                    line_frames.append(counters[key])
                    counters[key] += 1
                frame_indices = np.repeat(line_frames, n)
        if frame_indices is None:
            # Stable ranks distinguish repeated images, independent of line ordering.
            order = np.argsort(linear, kind='stable')
            starts = np.r_[0, np.flatnonzero(np.diff(linear[order])) + 1]
            ranks = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
            frame_indices = np.empty(len(linear), int)
            frame_indices[order] = ranks
        frame_indices = np.asarray(frame_indices, int)
        frame_count = int(np.max(frame_indices)) + 1
        values = np.zeros((frame_count, samples.shape[0], int(np.prod(shape))), samples.dtype)
        counts = np.zeros((frame_count, 1, int(np.prod(shape))), np.uint16)
        for coil in range(samples.shape[0]):
            np.add.at(values[:, coil], (frame_indices, linear), samples[coil])
        np.add.at(counts[:, 0], (frame_indices, linear), 1)
        np.divide(values, np.maximum(counts, 1), out=values)
        return AcquisitionData(values.reshape((frame_count, samples.shape[0]) + shape), shape,
            ('partition', 'phase', 'read')[-len(shape):], acquired=counts.reshape((frame_count, 1) + shape) > 0,
            auxiliary_samples=auxiliary or {}, statistical_weights=counts.reshape((frame_count, 1) + shape) if np.any(counts > 1) else None,
            recipe=ReconstructionRecipe(output_shape=output_shape, output_scale=float(np.sqrt(np.prod(output_shape) / np.prod(shape)))))
    # Do not round ramp sampling, half-sample shifts, radial or spiral coordinates.
    if frame_indices is None and len(set(info.adc_counts)) == 1:
        n = int(info.adc_counts[0])
        readouts = normalized.reshape((-1, n, len(shape)))
        if np.all(np.ptp(readouts[..., :-1], axis=1) < 2e-5):
            phase = np.round(readouts[:, 0, :-1], 5)
            _, inverse, counts = np.unique(phase, axis=0, return_inverse=True, return_counts=True)
            if len(set(counts)) == 1 and counts[0] > 1:
                visits = np.zeros(len(counts), int)
                indices = []
                for key in inverse:
                    indices.append(visits[key])
                    visits[key] += 1
                frame_indices = np.repeat(indices, n)
    if frame_indices is None:
        values, coords, acquired = samples[None], normalized, None
    else:
        frame_indices = np.asarray(frame_indices, int)
        frames, counts = np.unique(frame_indices, return_counts=True)
        if len(set(counts)) != 1:
            raise ValueError("Frame sample counts differ; select a frame explicitly")
        grouped = np.argsort(frame_indices, kind='stable').reshape((len(frames), int(counts[0])))
        values = np.empty((len(frames), samples.shape[0], int(counts[0])), samples.dtype)
        coords = None
        for index, selected in enumerate(grouped):
            k = normalized[selected]
            order = np.lexsort(tuple(np.round(k[:, i], 6) for i in range(k.shape[1])))
            k = k[order]
            if coords is not None and not np.allclose(coords, k, atol=2e-6):
                raise ValueError("Pulseq trajectory changes between frames; select a frame explicitly")
            coords = k
            values[index] = samples[:, selected[order]]
        acquired = None
    if isinstance(density, str) and density == 'auto':
        from mrax.reco.density import triangulated_density

        density = triangulated_density(coords, spatial_shape=shape)
    return AcquisitionData(values, shape, ('partition', 'phase', 'read')[-len(shape):],
        kind='sampled', trajectory=coords, acquired=acquired, auxiliary_samples=auxiliary or {},
        density=density, recipe=sampled_reconstruction_recipe(output_shape=output_shape))
